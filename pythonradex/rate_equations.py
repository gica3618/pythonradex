#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:03:13 2024

@author: gianni
"""

import numba as nb
import numpy as np
from pythonradex import atomic_transition,helpers
import numbers


class RateEquations():

    def __init__(self,geometry,molecule,collider_densities,Tkin,N,
                 treat_line_overlap,ext_background,T_dust,tau_dust):
        self.geometry = geometry
        self.molecule = molecule
        self.treat_line_overlap = treat_line_overlap
        self.N = N
        self.set_collision_rates(Tkin=Tkin,collider_densities=collider_densities)
        self.set_ext_background(ext_background=ext_background)
        self.set_dust(T_dust=T_dust,tau_dust=tau_dust)
        self.compute_Unu0_Vnu0()
        #equation of steady state: A*x=b, x=fractional population that we search
        self.b = np.zeros(self.molecule.n_levels)
        # the system of equations is not linearly independent
        #thus, I replace one equation by the normalisation condition,
        #i.e. x1+...+xn=1, where xi is the fractional population of level i
        #I replace the first equation (arbitrary choice):
        self.b[0] = 1
        if not treat_line_overlap:
            self.GammaR = self.GammaR_nu0
        else:
            self.GammaR = self.GammaR_averaged

    def set_N(self,N):
        self.N = N

    def collider_is_requested(self,collider):
        return collider in self.collider_densities

    def get_collider_selection(self):
        collider_selection = [self.collider_is_requested(collider) for collider
                              in self.molecule.ordered_colliders]
        return nb.typed.List(collider_selection)

    def get_collider_densities_list(self):
        collider_densities_list = nb.typed.List([])
        #important to iterate over ordered_colliders, not collider_densities.items()
        for collider in self.molecule.ordered_colliders:
            if self.collider_is_requested(collider):
                #need to convert to float, otherwise numba can get confused
                collider_densities_list.append(float(self.collider_densities[collider]))
            else:
                collider_densities_list.append(np.inf)
        return collider_densities_list

    def set_collision_rates(self,Tkin,collider_densities):
        self.Tkin = Tkin
        self.collider_densities = collider_densities
        collider_selection = self.get_collider_selection()
        collider_densities_list = self.get_collider_densities_list()
        self.GammaC = self.construct_GammaC(
                    Tkin=np.atleast_1d(self.Tkin),
                    collider_selection=collider_selection,
                    collider_densities_list=collider_densities_list,
                    n_levels=self.molecule.n_levels,
                    coll_trans_low_up_number=self.molecule.coll_trans_low_up_number,
                    Tkin_data = self.molecule.coll_Tkin_data,
                    K21_data=self.molecule.coll_K21_data,gups=self.molecule.coll_gups,
                    glows=self.molecule.coll_glows,DeltaEs=self.molecule.coll_DeltaEs)

    @staticmethod
    def generate_constant_func(const_value):
        def const_func(nu):
            return np.ones_like(nu)*const_value
        return const_func

    def assign_func_nu(self,func_name,argument):
        if isinstance(argument, numbers.Number):
            assert argument >= 0, 'ext_background, T_dust or tau_dust cannot be negative'
            setattr(self,func_name,self.generate_constant_func(const_value=argument))
        else:
            setattr(self,func_name,argument)

    def set_ext_background(self,ext_background):
        self.assign_func_nu(func_name='ext_background',argument=ext_background)
        if not self.treat_line_overlap:
            self.Iext_nu0 = np.array([self.ext_background(nu0) for nu0 in
                                      self.molecule.nu0])

    def set_dust(self,T_dust,tau_dust):
        for func_name,func in {'T_dust':T_dust,'tau_dust':tau_dust}.items():
            self.assign_func_nu(func_name=func_name,argument=func)
        if not self.treat_line_overlap:
            self.tau_dust_nu0 = np.array([self.tau_dust(nu0) for nu0 in
                                          self.molecule.nu0])
            self.S_dust_nu0 = np.array([self.S_dust(nu=nu0) for nu0 in
                                        self.molecule.nu0])

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

    def S_dust(self,nu):
        T = np.atleast_1d(self.T_dust(nu))
        return np.squeeze(helpers.B_nu(nu=nu,T=T))

    # We need to consider three cases:
    #1. no averaging over line profile (i.e. evaluate everything at nu0);
    #this cannot be used for overlapping lines, but dust is ok
    #2. with averaging, no treatment of overlapping lines
    #3. with averaging, with treatement of overlapping lines
    #all three cases need to include dust continuum


    #### case 1: "nu0 case" (everything evaluated at nu0, no overlapping lines allowed
    # (because no averaging))
    #basically, instead of averaging over the line profile, everything evaluated
    #at nu0, so the averaging function is a delta func instead of line profile

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def U_matrix_nu0(n_levels,trans_low_number,trans_up_number,A21):
        #already multiply by 4*pi/h*nu0 here
        U = np.zeros((n_levels,n_levels))
        n_transitions = len(trans_low_number)
        for i in range(n_transitions):
            n_low = trans_low_number[i]
            n_up = trans_up_number[i]
            U[n_up,n_low] = A21[i]
        return U

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def V_matrix_nu0(n_levels,trans_low_number,trans_up_number,B21,B12):
        #already multiply by 4*pi/h*nu0 here
        V = np.zeros((n_levels,n_levels))
        n_transitions = len(trans_low_number)
        for i in range(n_transitions):
            n_low = trans_low_number[i]
            n_up = trans_up_number[i]
            V[n_up,n_low] = B21[i]
            V[n_low,n_up] = B12[i]
        return V

    def compute_Unu0_Vnu0(self):
        n_levels=self.molecule.n_levels
        trans_low_number = self.molecule.nlow_rad_transitions
        trans_up_number = self.molecule.nup_rad_transitions
        A21 = self.molecule.A21
        B21 = self.molecule.B21
        B12 = self.molecule.B12
        self.U_nu0 = self.U_matrix_nu0(
                           n_levels=n_levels,trans_low_number=trans_low_number,
                           trans_up_number=trans_up_number,A21=A21)
        self.V_nu0 = self.V_matrix_nu0(
                          n_levels=n_levels,trans_low_number=trans_low_number,
                          trans_up_number=trans_up_number,B21=B21,B12=B12)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def tau_line_nu0(level_population,nlow_rad_transitions,nup_rad_transitions,N,
                     A21,phi_nu0,gup_rad_transitions,glow_rad_transitions,nu0):
        n_lines = nlow_rad_transitions.size
        tau_nu0 = np.empty(n_lines)
        for i in range(n_lines):
            N1 = N * level_population[nlow_rad_transitions[i]]
            N2 = N * level_population[nup_rad_transitions[i]]
            tau_nu0[i] = atomic_transition.fast_tau_nu(
                                 A21=A21[i],phi_nu=phi_nu0[i],g_up=gup_rad_transitions[i],
                                 g_low=glow_rad_transitions[i],N1=N1,N2=N2,nu=nu0[i])
        return tau_nu0

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def Ieff_nu0(n_levels,Iext_nu0,beta_nu0,S_dust_nu0,trans_low_number,trans_up_number,
                 tau_dust_nu0,tau_tot_nu0):
        Ieff = np.zeros((n_levels,n_levels))
        n_transitions = len(Iext_nu0)
        for i in range(n_transitions):
            n_low = trans_low_number[i]
            n_up = trans_up_number[i]
            tau_tot = tau_tot_nu0[i]
            if tau_tot == 0:
                psistar_eta_c = 0
            else:
                psistar_eta_c = (1-beta_nu0[i])*S_dust_nu0[i]*tau_dust_nu0[i]/tau_tot
            Ieff[n_low,n_up] = beta_nu0[i]*Iext_nu0[i] + psistar_eta_c
            Ieff[n_up,n_low] = Ieff[n_low,n_up]
        return Ieff

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def mixed_term_nu0(n_levels,beta_nu0,trans_low_number,trans_up_number,
                       tau_tot_nu0,tau_line_nu0,A21):
        #this is the term (Sum_l'' Chi^{dagger}_{l''l}) * Psi^*_nu * (Sum_l''' U_{l'l'''})
        #if there is no overlap, only the term l''=l' and l'''=l survives:
        #Psi^*_nu * Chi^{dagger}_l'l * U_l'l
        #note that U_l'l is 0 if l'<l
        term = np.zeros((n_levels,n_levels))
        n_transitions = len(beta_nu0)
        for i in range(n_transitions):
            n_low = trans_low_number[i]
            n_up = trans_up_number[i]
            tau_tot = tau_tot_nu0[i]
            if tau_tot == 0:
                continue
            #already multiply by 4*pi/h*nu0 here
            term[n_up,n_low] = (1-beta_nu0[i])*tau_line_nu0[i]/tau_tot*A21[i]
        return term

    def GammaR_nu0(self,level_population):
        tau_line_nu0 = self.tau_line_nu0(
                           level_population=level_population,
                           nlow_rad_transitions=self.molecule.nlow_rad_transitions,
                           nup_rad_transitions=self.molecule.nup_rad_transitions,
                           N=self.N,A21=self.molecule.A21,
                           phi_nu0=self.molecule.phi_nu0,
                           gup_rad_transitions=self.molecule.gup_rad_transitions,
                           glow_rad_transitions=self.molecule.glow_rad_transitions,
                           nu0=self.molecule.nu0)
        tau_tot_nu0 = tau_line_nu0 + self.tau_dust_nu0
        beta_nu0 = self.geometry.beta(tau_tot_nu0)
        n_levels = self.molecule.n_levels
        trans_low_number = self.molecule.nlow_rad_transitions
        trans_up_number = self.molecule.nup_rad_transitions
        Ieff_nu0 = self.Ieff_nu0(
                      n_levels=n_levels,Iext_nu0=self.Iext_nu0,beta_nu0=beta_nu0,
                      S_dust_nu0=self.S_dust_nu0,trans_low_number=trans_low_number,
                      trans_up_number=trans_up_number,tau_dust_nu0=self.tau_dust_nu0,
                      tau_tot_nu0=tau_tot_nu0)
        mixed_term_nu0  = self.mixed_term_nu0(
                             n_levels=n_levels,beta_nu0=beta_nu0,
                             trans_low_number=trans_low_number,
                             trans_up_number=trans_up_number,
                             tau_tot_nu0=tau_tot_nu0,tau_line_nu0=tau_line_nu0,
                             A21=self.molecule.A21)
        #GammaR_ll' = U_l'l+... i.e. indices are interchanged, so I have to transpose
        #see eq. 2.19 in Rybicki & Hummer (1992)
        #note that the factor h*nu/4pi is already taken into account in all terms
        GammaR = np.transpose(self.U_nu0+self.V_nu0*Ieff_nu0-mixed_term_nu0)
        diag = self.get_diagonal_GammaR(GammaR=GammaR)
        np.fill_diagonal(a=GammaR,val=diag)
        return GammaR

    @staticmethod
    def get_diagonal_GammaR(GammaR):
        #the diagonal of GammaR is the negative row sum over the non-diagonal terms
        return -(np.sum(GammaR,axis=0)-GammaR.diagonal())

    #### cases 2: averaging over line profile
    #note that U is proportional to phi_nu (when divided by h*nu), so we don't
    #need to average the U term

    def get_tau_tot_functions(self,level_population):
        kwargs = {'level_population':level_population,'N':self.N,
                  'tau_dust':self.tau_dust}
        return [self.molecule.get_tau_tot_nu(line_index=i,**kwargs)
                for i in range(self.molecule.n_rad_transitions)]

    def get_tau_line_functions(self,level_population):
        kwargs = {'level_population':level_population,'N':self.N}
        return [self.molecule.get_tau_line_nu(line_index=i,**kwargs) for i in
                range(self.molecule.n_rad_transitions)]

    def V_Ieff_averaged(self,tau_tot_functions):
        V_Ieff = np.zeros((self.molecule.n_levels,)*2)
        for i,trans in enumerate(self.molecule.rad_transitions):
            def Ieff(nu):
                tau_tot = tau_tot_functions[i](nu)
                beta = self.geometry.beta(tau_tot)
                betaIext = beta*self.ext_background(nu)
                return np.where(tau_tot==0,betaIext,betaIext
                                +(1-beta)*self.S_dust(nu)*self.tau_dust(nu)/tau_tot)
            Ieff_averaged = trans.line_profile.average_over_phi_nu(Ieff)
            n_low = trans.low.number
            n_up = trans.up.number
            V_Ieff[n_low,n_up] = self.V_nu0[n_low,n_up]*Ieff_averaged
            V_Ieff[n_up,n_low] = self.V_nu0[n_up,n_low]*Ieff_averaged
        return V_Ieff

    def mixed_term_averaged(self,tau_tot_functions,tau_line_functions):
        mixed_term = np.zeros((self.molecule.n_levels,)*2)
        #I choose that trans corresponds to the term U_lprime_lprimeprimeprime
        #while the pairing transition corresponds to chi_lprimeprime_l. This makes
        #it easier because I know that U_ab is zero if a<b, so I can simply iterate
        #over all transitions to have all posibilities for the U term covered
        for i,trans in enumerate(self.molecule.rad_transitions):
            #add the transition itself to the list of pairing transitions
            #(this is for the term lprimeprime=lprime and lprimeprimeprime=l):
            pairing_transitions = self.molecule.overlapping_lines[i] + [i,]
            for j in pairing_transitions:
                def mixed(nu):
                    #for tau_tot it doesn't matter which transition I take,
                    #because they are overlapping
                    tau_tot = tau_tot_functions[i](nu)
                    beta = self.geometry.beta(tau_tot)
                    #need to be careful here: need to take tau_line of the
                    #lprimeprime -> l transition
                    tau_line = tau_line_functions[j](nu)
                    #A21 (U matrix) needs to be from the lprime -> lprimeprimeprime
                    #transition
                    A21 = trans.A21
                    return np.where(tau_tot==0,0,(1-beta)*tau_line/tau_tot*A21)
                #need to average over the line profile of the
                #lprime -> lprimeprimeprime transition!
                mixed_averaged = trans.line_profile.average_over_phi_nu(mixed)
                pairing_trans = self.molecule.rad_transitions[j]
                #only non-diagonal terms need to be calculated, as the diagonal
                #of Gamma will be calculated from the non-diagonal terms directly
                if trans.up.number != pairing_trans.low.number:
                    mixed_term[trans.up.number,pairing_trans.low.number] += mixed_averaged
                if trans.up.number != pairing_trans.up.number:
                    mixed_term[trans.up.number,pairing_trans.up.number] += -mixed_averaged
        return mixed_term

    def GammaR_averaged(self,level_population):
        tau_line_functions = self.get_tau_line_functions(
                                             level_population=level_population)
        tau_tot_functions = self.get_tau_tot_functions(
                                              level_population=level_population)
        V_Ieff = self.V_Ieff_averaged(tau_tot_functions=tau_tot_functions)
        mixed_term = self.mixed_term_averaged(tau_tot_functions=tau_tot_functions,
                                              tau_line_functions=tau_line_functions)
        GammaR = np.transpose(self.U_nu0+V_Ieff-mixed_term)
        diag = self.get_diagonal_GammaR(GammaR=GammaR)
        np.fill_diagonal(a=GammaR,val=diag)
        return GammaR

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def apply_normalisation_condition(Gamma,n_levels):
        #first equation is replace by normalisation condition
        Gamma[0,:] = np.ones(n_levels)
        return Gamma

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_solve(Gamma,b):
        return np.linalg.solve(Gamma,b)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def assert_frac_pop_positive(fractional_population):
        assert np.all(fractional_population >= 0),\
                  'negative level population, potentially due to high column'\
                  +'density and/or low collider density'

    def solve(self,level_population):
        Gamma = self.GammaR(level_population=level_population) + self.GammaC
        Gamma = self.apply_normalisation_condition(
                                         Gamma=Gamma,n_levels=self.molecule.n_levels)
        fractional_population = self.fast_solve(Gamma=Gamma,b=self.b)
        self.assert_frac_pop_positive(fractional_population=fractional_population)
        return fractional_population