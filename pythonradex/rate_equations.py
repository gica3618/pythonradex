#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:03:13 2024

@author: gianni
"""

import numba as nb
import numpy as np
from pythonradex import atomic_transition


class RateEquations():

    def __init__(self,molecule,collider_densities,Tkin,mode):
        self.molecule = molecule
        self.collider_densities = collider_densities
        self.Tkin = Tkin
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
        self.coll_rate_matrix = self.construct_coll_rate_matrix(
                    Tkin=np.atleast_1d(self.Tkin),
                    collider_selection=self.collider_selection,
                    collider_densities_list=self.collider_densities_list,
                    n_levels=self.molecule.n_levels,
                    coll_trans_low_up_number=self.molecule.coll_trans_low_up_number,
                    Tkin_data = self.molecule.coll_Tkin_data,
                    K21_data=self.molecule.coll_K21_data,gups=self.molecule.coll_gups,
                    glows=self.molecule.coll_glows,DeltaEs=self.molecule.coll_DeltaEs)
        if not mode in ('LI','ALI'):
            raise ValueError(f'unknown rate equation mode "{mode}"')
        self.mode = mode
        if self.mode == 'LI':
            self.rad_rates = self.rad_rates_LI
        elif self.mode == 'ALI':
            self.rad_rates = self.rad_rates_ALI
        #equation of steady state: A*x=b, x=fractional population that we search
        self.b = np.zeros(self.molecule.n_levels)
        self.b[0] = 1

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def construct_coll_rate_matrix(
              Tkin,collider_selection,collider_densities_list,n_levels,
              coll_trans_low_up_number,Tkin_data,K21_data,gups,glows,DeltaEs):
        '''pre-calculate the matrix with the collsional rates, so that it can
        easily be added during solving the radiative transfer'''
        coll_rate_matrix = np.zeros((n_levels,n_levels))
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
                coll_rate_matrix[n_up,n_low] += K12*coll_density
                coll_rate_matrix[n_low,n_low] += -K12*coll_density
                coll_rate_matrix[n_low,n_up] += K21*coll_density
                coll_rate_matrix[n_up,n_up] += -K21*coll_density
        assert np.all(np.isfinite(coll_rate_matrix))
        return coll_rate_matrix

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
            #descruction of upper level towards lower level
            matrix[un,un] += -down
            #production of upper level from lower level
            matrix[un,ln] += up
            #descruction of lower level towards upper level
            matrix[ln,ln] += -up
        return matrix

    @staticmethod
    #@nb.jit(nopython=True,cache=True) #doesn't help
    def add_coll_rates(matrix,coll_rate_matrix):
        return matrix + coll_rate_matrix

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
        matrix = self.add_coll_rates(matrix=matrix,coll_rate_matrix=self.coll_rate_matrix)
        matrix = self.apply_normalisation_condition(
                                         matrix=matrix,n_levels=self.molecule.n_levels)
        fractional_population = self.fast_solve(matrix=matrix,b=self.b)
        self.assert_frac_pop_positive(fractional_population=fractional_population)
        return fractional_population