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
import time

#TODO remove unnecessary nb.jit


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

    def set_collision_rates(self,Tkin,collider_densities):
        #start_set = time.time()
        self.Tkin = Tkin
        self.collider_densities = collider_densities
        #start_GammaC = time.time()
        self.GammaC = self.molecule.get_GammaC(
                              Tkin=self.Tkin,collider_densities=collider_densities)
        #end_GammaC = time.time()
        #print(f'GammaC: {end_GammaC-start_GammaC}')
        #end_set = time.time()
        #print(f'set coll rates: {end_set-start_set:.3g}')

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
        # start = time.time()
        self.assign_func_nu(func_name='ext_background',argument=ext_background)
        if not self.treat_line_overlap:
            if isinstance(ext_background,numbers.Number):
                #faster than evaluating the function at each nu0
                self.Iext_nu0 = np.full(shape=self.molecule.nu0.shape,
                                        fill_value=ext_background)
            else:
                self.Iext_nu0 = np.array([self.ext_background(nu0) for nu0 in
                                          self.molecule.nu0])
        # end = time.time()
        # print(f'set ext bg: {end-start:.3g}')

    def set_dust(self,T_dust,tau_dust):
        # start = time.time()
        self.no_dust = T_dust == 0
        for func_name,func in {'T_dust':T_dust,'tau_dust':tau_dust}.items():
            self.assign_func_nu(func_name=func_name,argument=func)
        if not self.treat_line_overlap:
            # start_nu0 = time.time()
            if isinstance(tau_dust,numbers.Number):
                #fast:
                self.tau_dust_nu0 = np.full(shape=self.molecule.nu0.shape,
                                            fill_value=tau_dust)
            else:
                # self.tau_dust_nu0 = np.array([self.tau_dust(nu0) for nu0 in
                #                               self.molecule.nu0])
                self.tau_dust_nu0 = self.tau_dust(self.molecule.nu0)
            self.S_dust_nu0 = self.S_dust(nu=self.molecule.nu0)
            # end_nu0 = time.time()
            # print(f'dust nu0: {end_nu0-start_nu0:.3g}')
        # end = time.time()
        # print(f'set dust: {end-start:.3g}')

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

    def U_matrix_nu0(self):
        #already multiply by 4*pi/h*nu0 here
        U = np.zeros((self.molecule.n_levels,)*2)
        n_low = self.molecule.nlow_rad_transitions
        n_up = self.molecule.nup_rad_transitions
        U[n_up,n_low] = self.molecule.A21
        return U

    def V_matrix_nu0(self):
        #already multiply by 4*pi/h*nu0 here
        V = np.zeros((self.molecule.n_levels,)*2)
        n_low = self.molecule.nlow_rad_transitions
        n_up = self.molecule.nup_rad_transitions
        V[n_up,n_low] = self.molecule.B21
        V[n_low,n_up] = self.molecule.B12
        return V

    def compute_Unu0_Vnu0(self):
        self.U_nu0 = self.U_matrix_nu0()
        self.V_nu0 = self.V_matrix_nu0()

    # @staticmethod
    # @nb.jit(nopython=True,cache=True)
    # def fast_tau_line_nu0(level_population,N,nlow_rad_transitions,nup_rad_transitions,
    #                       A21,phi_nu0,glow_rad_transitions,gup_rad_transitions,nu0):
    #     N1 = N * level_population[nlow_rad_transitions]
    #     N2 = N * level_population[nup_rad_transitions]
    #     tau_nu0 = atomic_transition.tau_nu(
    #                    A21=A21,phi_nu=phi_nu0,g_low=glow_rad_transitions,
    #                    g_up=gup_rad_transitions,N1=N1,N2=N2,nu=nu0)
    #     return tau_nu0

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_tau_line_nu0(level_population,N,nlow_rad_transitions,nup_rad_transitions,
                          A21,phi_nu0,glow_rad_transitions,gup_rad_transitions,nu0):
        tau_nu0 = np.empty_like(A21)
        for i in range(len(nlow_rad_transitions)):
            N1 = N * level_population[nlow_rad_transitions[i]]
            N2 = N * level_population[nup_rad_transitions[i]]
            tau_nu0[i] = atomic_transition.tau_nu(
                               A21=A21[i],phi_nu=phi_nu0[i],
                               g_low=glow_rad_transitions[i],
                               g_up=gup_rad_transitions[i],N1=N1,N2=N2,nu=nu0[i])
        return tau_nu0

    def tau_line_nu0(self,level_population,N):
        return self.fast_tau_line_nu0(
                   level_population=level_population,N=N,
                   nlow_rad_transitions=self.molecule.nlow_rad_transitions,
                   nup_rad_transitions=self.molecule.nup_rad_transitions,
                   A21=self.molecule.A21,phi_nu0=self.molecule.phi_nu0,
                   glow_rad_transitions=self.molecule.glow_rad_transitions,
                   gup_rad_transitions=self.molecule.gup_rad_transitions,
                   nu0=self.molecule.nu0)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_Ieff_nu0(beta_nu0,tau_tot_nu0,no_dust,S_dust_nu0,tau_dust_nu0,
                      n_levels,nlow_rad_transitions,nup_rad_transitions,
                      Iext_nu0):
        if no_dust:
            psistar_eta_c = np.zeros_like(beta_nu0)
        else:
            psistar_eta_c = np.where(tau_tot_nu0 == 0,0,
                         (1-beta_nu0)*S_dust_nu0*tau_dust_nu0/tau_tot_nu0)
        Ieff = np.zeros((n_levels,n_levels))
        for i in range(len(nlow_rad_transitions)):
            nlow = nlow_rad_transitions[i]
            nup = nup_rad_transitions[i]
            I = beta_nu0[i]*Iext_nu0[i] + psistar_eta_c[i]
            Ieff[nlow,nup] = I
            Ieff[nup,nlow] = I
        return Ieff

    def Ieff_nu0(self,beta_nu0,tau_tot_nu0):
        return self.fast_Ieff_nu0(
                     beta_nu0=beta_nu0,tau_tot_nu0=tau_tot_nu0,no_dust=self.no_dust,
                     S_dust_nu0=self.S_dust_nu0,tau_dust_nu0=self.tau_dust_nu0,
                     n_levels=self.molecule.n_levels,
                     nlow_rad_transitions=self.molecule.nlow_rad_transitions,
                     nup_rad_transitions=self.molecule.nup_rad_transitions,
                     Iext_nu0=self.Iext_nu0)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_mixed_term_nu0(beta_nu0,tau_tot_nu0,tau_line_nu0,n_levels,
                            nlow_rad_transitions,nup_rad_transitions,A21):
        #this is the term (Sum_l'' Chi^{dagger}_{l''l}) * Psi^*_nu * (Sum_l''' U_{l'l'''})
        #if there is no overlap, only the term l''=l' and l'''=l survives:
        #Psi^*_nu * Chi^{dagger}_l'l * U_l'l
        #note that U_l'l is 0 if l'<l
        term = np.zeros((n_levels,n_levels))
        #already multiply by 4*pi/h*nu0 here
        for i in range(len(nlow_rad_transitions)):
            n_low = nlow_rad_transitions[i]
            n_up = nup_rad_transitions[i]
            tau_tot = tau_line_nu0[i]
            if tau_tot == 0:
                term[n_up,n_low] = 0
            else:
                term[n_up,n_low] = (1-beta_nu0[i])*tau_line_nu0[i]/tau_tot*A21[i]
        return term

    def mixed_term_nu0(self,beta_nu0,tau_tot_nu0,tau_line_nu0):
        return self.fast_mixed_term_nu0(
                   beta_nu0=beta_nu0,tau_tot_nu0=tau_tot_nu0,
                   tau_line_nu0=tau_line_nu0,n_levels=self.molecule.n_levels,
                   nlow_rad_transitions=self.molecule.nlow_rad_transitions,
                   nup_rad_transitions=self.molecule.nup_rad_transitions,
                   A21=self.molecule.A21)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def transposed_GammaR(U_nu0,V_nu0,Ieff_nu0,mixed_term_nu0):
        return np.transpose(U_nu0+V_nu0*Ieff_nu0-mixed_term_nu0)

    def GammaR_nu0(self,level_population):
        # start = time.time()
        tau_line_nu0 = self.tau_line_nu0(
                           level_population=level_population,N=self.N)
        tau_tot_nu0 = tau_line_nu0 + self.tau_dust_nu0
        # end = time.time()
        # print(f'tau: {end-start:.3g}')
        # start = time.time()
        beta_nu0 = self.geometry.beta(tau_tot_nu0)
        # end = time.time()
        # print(f'beta: {end-start:.3g}')
        # start = time.time()
        Ieff_nu0 = self.Ieff_nu0(beta_nu0=beta_nu0,tau_tot_nu0=tau_tot_nu0)
        # end = time.time()
        # print(f'Ieff: {end-start:.3g}')
        # start = time.time()
        mixed_term_nu0  = self.mixed_term_nu0(
                            beta_nu0=beta_nu0,tau_tot_nu0=tau_tot_nu0,
                            tau_line_nu0=tau_line_nu0)
        # end = time.time()
        # print(f'mixed term: {end-start:.3g}')
        #GammaR_ll' = U_l'l+... i.e. indices are interchanged, so I have to transpose
        #see eq. 2.19 in Rybicki & Hummer (1992)
        #note that the factor h*nu/4pi is already taken into account in all terms
        # start = time.time()
        #GammaR = np.transpose(self.U_nu0+self.V_nu0*Ieff_nu0-mixed_term_nu0)
        GammaR = self.transposed_GammaR(
                          U_nu0=self.U_nu0,V_nu0=self.V_nu0,Ieff_nu0=Ieff_nu0,
                          mixed_term_nu0=mixed_term_nu0)
        # end = time.time()
        # print(f'transpose: {end-start:.3g}')
        # start = time.time()
        diag = self.get_diagonal_GammaR(GammaR=GammaR)
        # end = time.time()
        # print(f'compute diag: {end-start:.3g}')
        # start = time.time()
        np.fill_diagonal(a=GammaR,val=diag)
        # end = time.time()
        # print(f'fill diagonal: {end-start:.3g}')
        return GammaR

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def get_diagonal_GammaR(GammaR):
        #the diagonal of GammaR is the negative row sum over the non-diagonal terms
        return -(np.sum(GammaR,axis=0)-np.diag(GammaR))

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

    def solve(self,level_population):
        # start = time.time()
        Gamma = self.GammaR(level_population=level_population) + self.GammaC
        Gamma[0,:] = np.ones(self.molecule.n_levels)
        # end = time.time()
        # print(f'Gamma: {end-start:.3g}')
        # start = time.time()
        fractional_population = np.linalg.solve(Gamma,b=self.b)
        assert np.all(fractional_population >= 0),\
                  'negative level population, potentially due to high column'\
                  +'density and/or low collider density'
        # end = time.time()
        # print(f'solve for frac pop: {end-start:.3g}')
        return fractional_population