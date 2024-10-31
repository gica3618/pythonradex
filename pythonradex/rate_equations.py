#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:03:13 2024

@author: gianni
"""

import numba as nb
import numpy as np
from pythonradex import atomic_transition
from scipy import constants


class RateEquations():

    def __init__(self,molecule,collider_densities,Tkin,average_over_line_profile):
        self.molecule = molecule
        self.collider_densities = collider_densities
        self.Tkin = Tkin
        self.average_over_line_profile = average_over_line_profile
        def is_requested(collider):
            return collider in collider_densities
        self.collider_selection = [is_requested(collider) for collider in
                                   self.molecule.ordered_colliders]
        self.collider_selection = nb.typed.List(self.collider_selection)
        self.collider_densities_list = nb.typed.List([])
        #important to iterate over ordered_colliders, not collider_densities.items()
        for collider in self.molecule.ordered_colliders:
            if is_requested(collider):
                #need to convert to float, otherwise numba can get confused
                self.collider_densities_list.append(float(collider_densities[collider]))
            else:
                self.collider_densities_list.append(np.inf)
        #precompute matrices
        self.GammaC = self.construct_GammaC(
                    Tkin=np.atleast_1d(self.Tkin),
                    collider_selection=self.collider_selection,
                    collider_densities_list=self.collider_densities_list,
                    n_levels=self.molecule.n_levels,
                    coll_trans_low_up_number=self.molecule.coll_trans_low_up_number,
                    Tkin_data = self.molecule.coll_Tkin_data,
                    K21_data=self.molecule.coll_K21_data,gups=self.molecule.coll_gups,
                    glows=self.molecule.coll_glows,DeltaEs=self.molecule.coll_DeltaEs)
        if not self.average_over_line_profile:
            n_levels=self.molecule.n_levels
            trans_low_number = self.molecule.nlow_rad_transitions
            trans_up_number = self.molecule.nup_rad_transitions
            nu0 = self.molecule.nu0
            A21 = self.molecule.A21
            B21 = self.molecule.B21
            B12 = self.molecule.B12
            self.U_nu0 = self.U_matrix_nu0(
                               n_levels=n_levels,trans_low_number=trans_low_number,
                               trans_up_number=trans_up_number,nu0=nu0,A21=A21)
            self.V_nu0 = self.V_matrix_nu0(
                              n_levels=n_levels,trans_low_number=trans_low_number,
                              trans_up_number=trans_up_number,nu0=nu0,B21=B21,B12=B12)
        #equation of steady state: A*x=b, x=fractional population that we search
        self.b = np.zeros(self.molecule.n_levels)
        self.b[0] = 1

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def construct_GammaC(
              Tkin,collider_selection,collider_densities_list,n_levels,
              coll_trans_low_up_number,Tkin_data,K21_data,gups,glows,DeltaEs):
        '''pre-calculate the matrix with the collsional rates, so that it can
        easily be added during solving the radiative transfer'''
        GammaC = np.zeros((n_levels,n_levels))
        for i in range(len(collider_selection)):
            if not collider_selection[i]:
                continue
            coll_density = collider_densities_list[i]
            n_transitions = len(DeltaEs[i])
            for j in range(n_transitions):
                K12,K21 = atomic_transition.fast_coll_coeffs(
                             Tkin=Tkin,Tkin_data=Tkin_data[i][j],
                             K21_data=K21_data[i][j],gup=gups[i][j],glow=glows[i][j],
                             Delta_E=DeltaEs[i][j])
                #K12 and K21 are 1D arrays because Tkin is a 1D array
                K12 = K12[0]
                K21 = K21[0]
                n_low = coll_trans_low_up_number[i][j,0]
                n_up = coll_trans_low_up_number[i][j,1]
                GammaC[n_up,n_low] += K12*coll_density
                GammaC[n_low,n_low] += -K12*coll_density
                GammaC[n_low,n_up] += K21*coll_density
                GammaC[n_up,n_up] += -K21*coll_density
        assert np.all(np.isfinite(GammaC))
        return GammaC

    #### case 1: "nu0 case" (everything evaluated at nu0, no overlap allowed
    # (because no averaging))

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def U_matrix_nu0(n_levels,trans_low_number,trans_up_number,nu0,A21):
        U = np.zeros((n_levels,n_levels))
        n_transitions = len(nu0)
        for i in range(n_transitions):
            n_low = trans_low_number[i]
            n_up = trans_up_number[i]
            U[n_up,n_low] = constants.h*nu0[i]/(4*np.pi)*A21[i]
        return U

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def V_matrix_nu0(n_levels,trans_low_number,trans_up_number,nu0,B21,B12):
        V = np.zeros((n_levels,n_levels))
        n_transitions = len(nu0)
        for i in range(n_transitions):
            n_low = trans_low_number[i]
            n_up = trans_up_number[i]
            const = constants.h*nu0[i]/(4*np.pi)
            V[n_up,n_low] = const*B21[i]
            V[n_low,n_up] = const*B12[i]
        return V

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def Ieff_nu0(n_levels,Iext_nu0,beta_nu0,Sdust_nu0,trans_low_number,trans_up_number,
                 tau_dust_nu0,tau_tot_nu0):
        #Ieff is a function of nu, but for the "nu0" case, the line profile is a delta
        #function, so Ieff also gets evaluated at nu0 via the delta function present in V
        Ieff = np.zeros((n_levels,n_levels))
        n_transitions = len(Iext_nu0)
        for i in range(n_transitions):
            n_low = trans_low_number[i]
            n_up = trans_up_number[i]
            tau_tot = tau_tot_nu0[i]
            if tau_tot == 0:
                psistar_eta_c = 0
            else:
                psistar_eta_c = (1-beta_nu0[i])*Sdust_nu0[i]*tau_dust_nu0[i]/tau_tot_nu0[i]
            Ieff[n_low,n_up] = beta_nu0[i]*Iext_nu0[i] + psistar_eta_c
            Ieff[n_up,n_low] = Ieff[n_low,n_up]
        return Ieff

    def GammaR_nu0(self,Iext_nu0,beta_nu0,Sdust_nu0,tau_dust_nu0,tau_tot_nu0):
        n_levels=self.molecule.n_levels
        trans_low_number = self.molecule.nlow_rad_transitions
        trans_up_number = self.molecule.nup_rad_transitions
        nu0 = self.molecule.nu0
        Ieff_nu0 = self.Ieff_nu0(
                      n_levels=n_levels,Iext_nu0=Iext_nu0,beta_nu0=beta_nu0,
                      Sdust_nu0=Sdust_nu0,trans_low_number=trans_low_number,
                      trans_up_number=trans_up_number,tau_dust_nu0=tau_dust_nu0,
                      tau_tot_nu0=tau_tot_nu0)
        GammaR = 4*np.pi/(constants.h*nu0) * (self.U_nu0+self.V_nu0*Ieff_nu0)
        #the diagonal of GammaR is the negative row sum over the non-diagonal terms
        diagonal = -(np.sum(GammaR,axis=0)-GammaR.diagonal())
        np.fill_diagonal(a=GammaR,val=diagonal)
        return GammaR

    #### case 2: averaging, but no overlap
    def GammaR_average_no_overlap(self):
        

    @staticmethod
    #@nb.jit(nopython=True,cache=True) #doesn't help
    def rad_rates_LI(Jbar,A21,B12,B21):
        uprate = B12*Jbar
        downrate = A21+B21*Jbar
        return uprate,downrate

    @staticmethod
    def rad_rates_ALI(A21,B12,B21,A21_factor,B21_factor):
        #note that in the case without dust and without overlapping lines,
        #this reduces to the standard ALI scheme shown in section 7.10 of the Dullemond
        #radiative transfer lectures, that is
        # uprate = B12*beta*Iext
        # and
        # downrate = A21*beta + B21*beta*Iext
        uprate = B12*B21_factor
        downrate = A21*A21_factor + B21*B21_factor
        return uprate,downrate

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def add_rad_rates(matrix,rad_rates,nlow_rad_transitions,nup_rad_transitions):
        uprates,downrates = rad_rates
        for i in range(nlow_rad_transitions.size):
            ln = nlow_rad_transitions[i]
            un = nup_rad_transitions[i]
            down = downrates[i]
            up = uprates[i]
            #production of low level from upper level
            matrix[ln,un] +=  down
            #destruction of upper level towards lower level
            matrix[un,un] += -down
            #production of upper level from lower level
            matrix[un,ln] += up
            #destruction of lower level towards upper level
            matrix[ln,ln] += -up
        return matrix

    @staticmethod
    #@nb.jit(nopython=True,cache=True) #doesn't help
    def add_coll_rates(matrix,GammaC):
        return matrix + GammaC

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def apply_normalisation_condition(matrix,n_levels):
        # the system of equations is not linearly independent
        #thus, I replace one equation by the normalisation condition,
        #i.e. x1+...+xn=1, where xi is the fractional population of level i
        #I replace the first equation (arbitrary choice):
        matrix[0,:] = np.ones(n_levels)
        return matrix

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_solve(matrix,b):
        return np.linalg.solve(matrix,b)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def assert_frac_pop_positive(fractional_population):
        assert np.all(fractional_population >= 0),\
                  'negative level population, potentially due to high column'\
                  +'density and/or low collider density'

    def solve(self,**kwargs):
        matrix = np.zeros((self.molecule.n_levels,self.molecule.n_levels))
        rad_rates = self.rad_rates(**kwargs)
        matrix = self.add_rad_rates(
                     matrix=matrix,rad_rates=rad_rates,
                     nlow_rad_transitions=self.molecule.nlow_rad_transitions,
                     nup_rad_transitions=self.molecule.nup_rad_transitions)
        matrix = self.add_coll_rates(matrix=matrix,GammaC=self.GammaC)
        matrix = self.apply_normalisation_condition(
                                         matrix=matrix,n_levels=self.molecule.n_levels)
        fractional_population = self.fast_solve(matrix=matrix,b=self.b)
        self.assert_frac_pop_positive(fractional_population=fractional_population)
        return fractional_population