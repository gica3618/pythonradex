#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:04:44 2024

@author: gianni
"""

import os
from scipy import constants
from pythonradex import rate_equations,molecule,helpers,escape_probability
import numpy as np
import itertools
import pytest

here = os.path.dirname(os.path.abspath(__file__))
cmb = helpers.generate_CMB_background()
geometry = escape_probability.UniformSphere()

Tkin = 150
N = 1e15/constants.centi**-2

def T_dust(nu):
    return np.ones_like(nu)*200

def tau_dust(nu):
    return np.ones_like(nu)*0.5


class RateEqGenerator():

    def __init__(self,molecule,collider_densities):
        self.molecule = molecule
        self.collider_densities = collider_densities

    def generate_rate_eq(self,treat_line_overlap,ext_background,T_dust,tau_dust):
        return rate_equations.RateEquations(
                          molecule=self.molecule,
                          collider_densities=self.collider_densities,
                          Tkin=Tkin,treat_line_overlap=treat_line_overlap,
                          geometry=geometry,N=N,ext_background=ext_background,
                          T_dust=T_dust,tau_dust=tau_dust)

    def rate_eq_iterator(self):
        for treat_line_overlap,ext_bg in\
                                  itertools.product([True,False],[0,cmb]):
            for Td,taud in zip((0,T_dust),(0,tau_dust)):
                yield self.generate_rate_eq(
                           treat_line_overlap=treat_line_overlap,
                           ext_background=ext_bg,T_dust=T_dust,tau_dust=tau_dust)

    def get_zero_tau_rate_eq(self,ext_background):
        return rate_equations.RateEquations(
                      molecule=self.molecule,collider_densities=self.collider_densities,
                      Tkin=Tkin,treat_line_overlap=True,
                      geometry=geometry,N=0,ext_background=ext_background,
                      T_dust=0,tau_dust=0)

class TestGeneral():

    #take a molecule with many colliders on purpose
    test_molecule = molecule.EmittingMolecule(
                          datafilepath=os.path.join(here,'LAMDA_files/o.dat'),
                          line_profile_type='Gaussian',
                          width_v=1*constants.kilo)
    LTE_level_pop = test_molecule.LTE_level_pop(T=123)
    collider_densities = {'para-H2':1e4/constants.centi**3,
                          'e':1e2/constants.centi**3,
                          'H':1.2e4/constants.centi**3}
    rate_eq_generator = RateEqGenerator(molecule=test_molecule,
                                        collider_densities=collider_densities)

    def test_collider_is_requested(self):
        for rate_eq in self.rate_eq_generator.rate_eq_iterator():
            for coll in self.collider_densities.keys():
                assert rate_eq.collider_is_requested(coll)
            for coll in self.test_molecule.ordered_colliders:
                if not coll in self.collider_densities:
                    assert not rate_eq.collider_is_requested(coll)

    def test_get_collider_selection(self):
        expected_ordered_colliders = ['H','H+','e','ortho-H2','para-H2']
        assert self.test_molecule.ordered_colliders\
                                           == expected_ordered_colliders
        expected_collider_selection = np.array((True,False,True,False,True))
        for rate_eq in self.rate_eq_generator.rate_eq_iterator():
            assert np.all(expected_collider_selection==rate_eq.get_collider_selection())

    def test_get_collider_densities_list(self):
        for rate_eq in self.rate_eq_generator.rate_eq_iterator():
            expected_collider_densities_list = []
            for coll in rate_eq.molecule.ordered_colliders:
                if coll in self.collider_densities:
                    expected_collider_densities_list.append(self.collider_densities[coll])
                else:
                    expected_collider_densities_list.append(np.inf)
            assert np.all(expected_collider_densities_list
                          ==list(rate_eq.get_collider_densities_list()))

    def test_coll_rate_matrix(self):
        #write a slow "for" loop to calculate the rate matrix and compare to the fast
        #loop used in the code
        expected_GammaC = np.zeros((self.test_molecule.n_levels,)*2)
        for collider,coll_density in self.collider_densities.items():
            coll_transitions = self.test_molecule.coll_transitions[collider]
            for trans in coll_transitions:
                n_up = trans.up.number
                n_low = trans.low.number
                K12,K21 = trans.coeffs(Tkin=Tkin)
                expected_GammaC[n_up,n_low] += K12*coll_density
                expected_GammaC[n_low,n_low] += -K12*coll_density
                expected_GammaC[n_low,n_up] += K21*coll_density
                expected_GammaC[n_up,n_up] += -K21*coll_density
        for rate_eq in self.rate_eq_generator.rate_eq_iterator():
            assert np.allclose(expected_GammaC,rate_eq.GammaC,atol=0,rtol=1e-10)

    def test_nu_functions(self):
        def non_zero(nu):
            return (nu/(100*constants.giga))**2
        rate_eq = rate_equations.RateEquations(
                       molecule=self.test_molecule,
                       collider_densities=self.collider_densities,
                       Tkin=Tkin,treat_line_overlap=False,geometry=geometry,
                       N=N,ext_background=non_zero,T_dust=non_zero,tau_dust=non_zero)
        test_nu0 = rate_eq.molecule.nu0
        for func in ('ext_background','T_dust','tau_dust'):
            assert np.all(getattr(rate_eq,func)(test_nu0)==non_zero(test_nu0))

    def test_const_nu_functions(self):
        kwargs = {'molecule':self.test_molecule,
                  'collider_densities':self.collider_densities,'Tkin':123,
                  'treat_line_overlap':False,'geometry':'uniform sphere',
                  'N':1e12/constants.centi**2}
        valid_const_values = [0,1.2,3]
        for const_value in valid_const_values:
            rate_eq = rate_equations.RateEquations(
                           **kwargs,ext_background=const_value,T_dust=const_value,
                           tau_dust=const_value)
            test_nu0 = rate_eq.molecule.nu0
            for func in ('ext_background','T_dust','tau_dust'):
                assert np.all(getattr(rate_eq,func)(test_nu0)==const_value)
        invalid_const_values = [-1.2,-1]
        for const_value in invalid_const_values:
            with pytest.raises(AssertionError):
                rate_eq = rate_equations.RateEquations(
                               **kwargs,ext_background=const_value,T_dust=const_value,
                               tau_dust=const_value)
                
    def test_U_nu0(self):
        expected_U = np.zeros((self.test_molecule.n_levels,)*2)
        for trans in self.test_molecule.rad_transitions:
            expected_U[trans.up.number,trans.low.number] = trans.A21
        for rate_eq in self.rate_eq_generator.rate_eq_iterator():
            assert np.all(expected_U==rate_eq.U_nu0)

    def test_V_nu0(self):
        expected_V = np.zeros((self.test_molecule.n_levels,)*2)
        for trans in self.test_molecule.rad_transitions:
            expected_V[trans.up.number,trans.low.number] = trans.B21
            expected_V[trans.low.number,trans.up.number] = trans.B12
        for rate_eq in self.rate_eq_generator.rate_eq_iterator():
            assert np.all(expected_V==rate_eq.V_nu0)

    def test_tau_line_nu0(self):
        level_pop = self.LTE_level_pop
        for rate_eq in self.rate_eq_generator.rate_eq_iterator():
            tau_line_nu0 = rate_eq.tau_line_nu0(
                               level_population=level_pop,
                               nlow_rad_transitions=rate_eq.molecule.nlow_rad_transitions,
                               nup_rad_transitions=rate_eq.molecule.nup_rad_transitions,
                               N=rate_eq.N,A21=rate_eq.molecule.A21,
                               phi_nu0=rate_eq.molecule.phi_nu0,
                               gup_rad_transitions=rate_eq.molecule.gup_rad_transitions,
                               glow_rad_transitions=rate_eq.molecule.glow_rad_transitions,
                               nu0=rate_eq.molecule.nu0)
            expected_tau_line_nu0 = rate_eq.molecule.get_tau_nu0_lines(
                                       N=rate_eq.N,level_population=level_pop)
            assert np.all(tau_line_nu0==expected_tau_line_nu0)

    def test_Ieff_nu0(self):
        general_kwargs = {'n_levels':self.test_molecule.n_levels,
                          'trans_low_number':self.test_molecule.nlow_rad_transitions,
                          'trans_up_number':self.test_molecule.nup_rad_transitions}
        level_pop = self.LTE_level_pop
        for rate_eq in self.rate_eq_generator.rate_eq_iterator():
            Iext_nu0 = rate_eq.ext_background(self.test_molecule.nu0)
            tau_tot_kwargs = {'level_population':level_pop,'N':N,
                              'tau_dust':rate_eq.tau_dust}
            tau_tot_nu0 = [self.test_molecule.get_tau_tot_nu(line_index=i,**tau_tot_kwargs)(line.nu0)
                           for i,line in enumerate(self.test_molecule.rad_transitions)]
            tau_tot_nu0 = np.array(tau_tot_nu0)
            beta_nu0 = geometry.beta(tau_tot_nu0)
            S_dust_nu0 = helpers.B_nu(nu=self.test_molecule.nu0,
                                      T=T_dust(self.test_molecule.nu0))
            tau_dust_nu0 = tau_dust(self.test_molecule.nu0)
            Ieff = rate_eq.Ieff_nu0(**general_kwargs,Iext_nu0=Iext_nu0,beta_nu0=beta_nu0,
                                    S_dust_nu0=S_dust_nu0,tau_dust_nu0=tau_dust_nu0,
                                    tau_tot_nu0=tau_tot_nu0)
            expected_Ieff = np.zeros((self.test_molecule.n_levels,)*2)
            for t,trans in enumerate(self.test_molecule.rad_transitions):
                nup,nlow = trans.up.number,trans.low.number
                expected_Ieff[nup,nlow] = expected_Ieff[nlow,nup]\
                      = beta_nu0[t]*Iext_nu0[t]\
                            + (1-beta_nu0[t])*S_dust_nu0[t]*tau_dust_nu0[t]/tau_tot_nu0[t]
            assert np.all(Ieff==expected_Ieff)

    def test_Ieff_nu0_zero_tau(self):
        general_kwargs = {'n_levels':self.test_molecule.n_levels,
                          'trans_low_number':self.test_molecule.nlow_rad_transitions,
                          'trans_up_number':self.test_molecule.nup_rad_transitions}
        rate_eq = self.rate_eq_generator.get_zero_tau_rate_eq(ext_background=cmb)
        Iext_nu0 = rate_eq.ext_background(self.test_molecule.nu0)
        tau_tot_nu0 = np.zeros(rate_eq.molecule.n_rad_transitions)
        beta_nu0 = geometry.beta(tau_tot_nu0)
        S_dust_nu0 = np.zeros_like(tau_tot_nu0)
        tau_dust_nu0 = np.zeros_like(tau_tot_nu0)
        Ieff = rate_eq.Ieff_nu0(
                  **general_kwargs,Iext_nu0=Iext_nu0,beta_nu0=beta_nu0,
                  S_dust_nu0=S_dust_nu0,tau_dust_nu0=tau_dust_nu0,tau_tot_nu0=tau_tot_nu0)
        expected_Ieff = np.zeros((self.test_molecule.n_levels,)*2)
        for line in rate_eq.molecule.rad_transitions:
            nup,nlow = line.up.number,line.low.number
            expected_Ieff[nup,nlow] = cmb(line.nu0)
            expected_Ieff[nlow,nup] = expected_Ieff[nup,nlow]
        assert np.all(Ieff==expected_Ieff)

    def test_mixed_term_nu0(self):
        A21 = self.test_molecule.A21
        level_pop = self.LTE_level_pop
        general_kwargs = {'n_levels':self.test_molecule.n_levels,'A21':A21,
                          'trans_low_number':self.test_molecule.nlow_rad_transitions,
                          'trans_up_number':self.test_molecule.nup_rad_transitions}
        for rate_eq in self.rate_eq_generator.rate_eq_iterator():
            tau_tot_kwargs = {'level_population':level_pop,'N':N,
                              'tau_dust':rate_eq.tau_dust}
            tau_tot_nu0 = [self.test_molecule.get_tau_tot_nu(line_index=i,**tau_tot_kwargs)(line.nu0)
                           for i,line in enumerate(self.test_molecule.rad_transitions)]
            tau_tot_nu0 = np.array(tau_tot_nu0)
            beta_nu0 = geometry.beta(tau_tot_nu0)
            tau_line_nu0 = self.test_molecule.get_tau_nu0_lines(
                                           N=N,level_population=level_pop)
            mixed_term = rate_eq.mixed_term_nu0(
                                             **general_kwargs,tau_tot_nu0=tau_tot_nu0,
                                             beta_nu0=beta_nu0,tau_line_nu0=tau_line_nu0)
            expected_mixed_term = np.zeros((self.test_molecule.n_levels,)*2)
            for t,trans in enumerate(self.test_molecule.rad_transitions):
                nup,nlow = trans.up.number,trans.low.number
                expected_mixed_term[nup,nlow] = (1-beta_nu0[t])/tau_tot_nu0[t]\
                                                        *tau_line_nu0[t]*A21[t]
            assert np.all(mixed_term==expected_mixed_term)
            assert np.all(mixed_term.diagonal()==0)

    def test_mixed_term_nu0_zero_tau(self):
        rate_eq = self.rate_eq_generator.get_zero_tau_rate_eq(ext_background=cmb)
        A21 = self.test_molecule.A21
        general_kwargs = {'n_levels':self.test_molecule.n_levels,'A21':A21,
                          'trans_low_number':self.test_molecule.nlow_rad_transitions,
                          'trans_up_number':self.test_molecule.nup_rad_transitions}
        tau_tot_nu0 = np.zeros((self.test_molecule.n_rad_transitions)*2)
        beta_nu0 = geometry.beta(tau_tot_nu0)
        tau_line_nu0 = tau_tot_nu0.copy()
        mixed_term = rate_eq.mixed_term_nu0(
                                         **general_kwargs,tau_tot_nu0=tau_tot_nu0,
                                         beta_nu0=beta_nu0,tau_line_nu0=tau_line_nu0)
        assert np.all(mixed_term==0)

    def test_tau_line_functions(self):
        for rate_eq in self.rate_eq_generator.rate_eq_iterator():
            tau_line_funcs = rate_eq.get_tau_line_functions(
                                            level_population=self.LTE_level_pop)
            for i,line in enumerate(rate_eq.molecule.rad_transitions):
                width_nu = rate_eq.molecule.width_v/constants.c*line.nu0
                nu = np.linspace(line.nu0-width_nu,line.nu0+width_nu,100)
                expected_tau_line = self.test_molecule.get_tau_line_nu(
                                    line_index=i,level_population=self.LTE_level_pop,
                                    N=N)(nu)
                assert np.all(tau_line_funcs[i](nu)==expected_tau_line)
        zero_tau_rate_eq = self.rate_eq_generator.get_zero_tau_rate_eq(ext_background=cmb)
        tau_line_funcs = zero_tau_rate_eq.get_tau_line_functions(
                                                 level_population=self.LTE_level_pop)
        for i,line in enumerate(zero_tau_rate_eq.molecule.rad_transitions):
            width_nu = rate_eq.molecule.width_v/constants.c*line.nu0
            nu = np.linspace(line.nu0-width_nu,line.nu0+width_nu,100)
            assert np.all(tau_line_funcs[i](nu)==0)

    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_V_Ieff_averaged_zero_N(self):
        level_pop = self.LTE_level_pop
        rate_eq = self.rate_eq_generator.get_zero_tau_rate_eq(ext_background=cmb)
        tau_tot_functions = rate_eq.get_tau_tot_functions(level_population=level_pop)
        V_Ieff = rate_eq.V_Ieff_averaged(tau_tot_functions=tau_tot_functions)
        expected_V_Ieff = np.zeros((self.test_molecule.n_levels,)*2)
        for line in self.test_molecule.rad_transitions:
            nup,nlow = line.up.number,line.low.number
            Ieff = line.line_profile.average_over_phi_nu(cmb)
            expected_V_Ieff[nup,nlow] = Ieff*line.B21
            expected_V_Ieff[nlow,nup] = Ieff*line.B12
        assert np.all(V_Ieff==expected_V_Ieff)

    def test_V_Ieff_averaged(self):
        level_pop = self.LTE_level_pop
        for rate_eq in self.rate_eq_generator.rate_eq_iterator():
            tau_tot_functions = rate_eq.get_tau_tot_functions(level_population=level_pop)
            V_Ieff = rate_eq.V_Ieff_averaged(tau_tot_functions=tau_tot_functions)
            expected_V_Ieff = np.zeros((rate_eq.molecule.n_levels,)*2)
            for i,trans in enumerate(rate_eq.molecule.rad_transitions):
                def Ieff(nu):
                    tau_tot = tau_tot_functions[i](nu)
                    beta = geometry.beta(tau_tot)
                    betaIext = beta*rate_eq.ext_background(nu)
                    return np.where(tau_tot==0,betaIext,betaIext
                                    +(1-beta)*rate_eq.S_dust(nu)*rate_eq.tau_dust(nu)/tau_tot)
                Ieff_averaged = trans.line_profile.average_over_phi_nu(Ieff)
                n_low = trans.low.number
                n_up = trans.up.number
                expected_V_Ieff[n_low,n_up] = trans.B12*Ieff_averaged
                expected_V_Ieff[n_up,n_low] = trans.B21*Ieff_averaged
            assert np.all(V_Ieff==expected_V_Ieff)


class TestOverlapStuff():

    #transitions 8 and 10 of CN are separated by ~650 km/s
    test_molecule = molecule.EmittingMolecule(
                              datafilepath=os.path.join(here,'LAMDA_files/cn.dat'),
                              line_profile_type='Gaussian',
                              width_v=1000*constants.kilo)
    assert test_molecule.has_overlapping_lines
    LTE_level_pop = test_molecule.LTE_level_pop(T=102)
    collider_densities = {'He':1e4/constants.centi**3,
                          'e':1e2/constants.centi**3}
    rate_eq_generator = RateEqGenerator(molecule=test_molecule,
                                        collider_densities=collider_densities)

    def test_tau_tot_functions(self):
        for rate_eq in self.rate_eq_generator.rate_eq_iterator():
            tau_tot_funcs = rate_eq.get_tau_tot_functions(
                                            level_population=self.LTE_level_pop)
            for i,line in enumerate(rate_eq.molecule.rad_transitions):
                width_nu = rate_eq.molecule.width_v/constants.c*line.nu0
                nu = np.linspace(line.nu0-width_nu,line.nu0+width_nu,100)
                expected_tau_tot = self.test_molecule.get_tau_tot_nu(
                                    line_index=i,level_population=self.LTE_level_pop,
                                    N=N,tau_dust=rate_eq.tau_dust)(nu)
                assert np.all(tau_tot_funcs[i](nu)==expected_tau_tot)
        zero_tau_rate_eq = self.rate_eq_generator.get_zero_tau_rate_eq(ext_background=cmb)
        tau_tot_funcs = zero_tau_rate_eq.get_tau_tot_functions(
                                          level_population=self.LTE_level_pop)
        for i,line in enumerate(zero_tau_rate_eq.molecule.rad_transitions):
            width_nu = rate_eq.molecule.width_v/constants.c*line.nu0
            nu = np.linspace(line.nu0-width_nu,line.nu0+width_nu,100)
            assert np.all(tau_tot_funcs[i](nu)==0)

    def get_mixed_term_transition_pairs(self,l,lprime):
        #for each level, the rad transitions going downward:
        downward_rad_transitions = []
        #for each level, the rad transitions involving that level: 
        level_transitions = []
        for i in range(self.test_molecule.n_levels):
            down_transitions = []
            level_transitions_i = []
            for j,trans in enumerate(self.test_molecule.rad_transitions):
                if trans.up.number == i:
                    down_transitions.append(j)
                if trans.up.number == i or trans.low.number == i:
                    level_transitions_i.append(j)
            downward_rad_transitions.append(down_transitions)
            level_transitions.append(level_transitions_i)
        #look at equation 2.19 in Rybicki & Hummer (1992)
        #the left side of the double sum is over transitions involving l
        transitions_involving_l = level_transitions[l]
        #the right side of the double sum is over transitions involving lprime,
        #but since it contains U_lprime_lprimeprimeprime, I only need to consider
        #downward transitions from lprime (since U is zero for upward transitions)
        downward_transitions_from_lprime = downward_rad_transitions[lprime]
        #a list containing pairs of transitions to consider, where the first element
        #of the pair is the transitions involving l, and the second the one involving
        #lprime
        transition_pairs_to_consider = []
        for downward_trans in downward_transitions_from_lprime:
            trans_overlapping_with_downward_trans\
                              = self.test_molecule.overlapping_lines[downward_trans]
            for overlap_trans in trans_overlapping_with_downward_trans:
                if overlap_trans in transitions_involving_l:
                    transition_pairs_to_consider.append([overlap_trans,downward_trans])
        #add the transition lprime -> l if it exists (this corresponds to the
        #term where the transition in the lprimeprime and lprimeprimeprime is
        #the same)
        for downward_trans in downward_transitions_from_lprime:
            if self.test_molecule.rad_transitions[downward_trans].low.number == l:
                transition_pairs_to_consider.append([downward_trans,]*2)
        return transition_pairs_to_consider

    def mixed_term_averaged(self,tau_tot_functions,tau_line_functions):
        mixed_term = np.zeros((self.test_molecule.n_levels,)*2)
        #rather than iterating over transitions as in the production code,
        #here I iterate over every element of the matrix (probably slower, but
        #easier to understand)
        for l,lprime in itertools.product(range(self.test_molecule.n_levels),repeat=2):
            if l==lprime:
                #no need to calculate diagonal term, as diagonal of Gamma will be
                #calculated from the non-diagonal terms
                continue
            transition_pairs_to_consider = self.get_mixed_term_transition_pairs(
                                                               l=l,lprime=lprime)
            mixed_lprime_l = 0
            for trans_pair in transition_pairs_to_consider:
                def mixed(nu):
                    #for tau_tot it doesn't matter which transition I take,
                    #becuase they are overlapping
                    tau_tot = tau_tot_functions[trans_pair[0]](nu)
                    beta = geometry.beta(tau_tot)
                    #need to be careful here: need to take tau_line of the
                    #lprimeprime -> l transition
                    tau_line = tau_line_functions[trans_pair[0]](nu)
                    #A21 (U matrix) needs to be from the lprime -> lprimeprimeprime
                    #transition
                    A21 = self.test_molecule.rad_transitions[trans_pair[1]].A21
                    return np.where(tau_tot==0,0,(1-beta)*tau_line/tau_tot*A21)
                trans0 = self.test_molecule.rad_transitions[trans_pair[0]]
                if trans0.up.number == l:
                    sign = -1
                else:
                    assert trans0.low.number == l
                    sign = 1
                #need to average over the line profile of the
                #lprime -> lprimeprimeprime transition!
                mixed_lprime_l += sign*self.test_molecule.rad_transitions[trans_pair[1]]\
                                        .line_profile.average_over_phi_nu(mixed)
            mixed_term[lprime,l] = mixed_lprime_l
        return mixed_term

    def test_mixed_term_averaged(self):
        level_pop = self.LTE_level_pop
        for rate_eq in self.rate_eq_generator.rate_eq_iterator():
            if not rate_eq.treat_line_overlap:
                continue
            tau_tot_functions = rate_eq.get_tau_tot_functions(level_population=level_pop)
            tau_line_functions = rate_eq.get_tau_line_functions(level_population=level_pop)
            mixed_term_averaged = rate_eq.mixed_term_averaged(
                                      tau_tot_functions=tau_tot_functions,
                                      tau_line_functions=tau_line_functions)
            expected = self.mixed_term_averaged(tau_tot_functions=tau_tot_functions,
                                                tau_line_functions=tau_line_functions)
            assert np.allclose(mixed_term_averaged,expected,atol=0,rtol=1e-10)
            assert np.all(mixed_term_averaged.diagonal()==0)


class TestGammaR():
    
    def all_rate_eqs_iterator(self):
        return itertools.chain(TestGeneral.rate_eq_generator.rate_eq_iterator(),
                               TestOverlapStuff.rate_eq_generator.rate_eq_iterator())

    def test_Gammar_diag(self):
        for rate_eq in self.all_rate_eqs_iterator():
            level_pop = rate_eq.molecule.LTE_level_pop(T=102)
            GammaR = rate_eq.GammaR(level_population=level_pop)
            expected_diag = -(GammaR.sum(axis=0)-GammaR.diagonal())
            assert np.allclose(GammaR.diagonal(),expected_diag,atol=0,rtol=1e-10)

    def test_GammaR(self):
        for rate_eq in self.all_rate_eqs_iterator():
            level_pop = rate_eq.molecule.LTE_level_pop(T=102)
            if not rate_eq.treat_line_overlap:
                tau_line_nu0 = rate_eq.tau_line_nu0(
                                   level_population=level_pop,
                                   nlow_rad_transitions=rate_eq.molecule.nlow_rad_transitions,
                                   nup_rad_transitions=rate_eq.molecule.nup_rad_transitions,
                                   N=rate_eq.N,A21=rate_eq.molecule.A21,
                                   phi_nu0=rate_eq.molecule.phi_nu0,
                                   gup_rad_transitions=rate_eq.molecule.gup_rad_transitions,
                                   glow_rad_transitions=rate_eq.molecule.glow_rad_transitions,
                                   nu0=rate_eq.molecule.nu0)
                tau_tot_nu0 = tau_line_nu0 + rate_eq.tau_dust_nu0
                n_levels = rate_eq.molecule.n_levels
                beta_nu0 = rate_eq.geometry.beta(tau_tot_nu0)
                trans_low_number = rate_eq.molecule.nlow_rad_transitions
                trans_up_number = rate_eq.molecule.nup_rad_transitions
                Ieff_nu0 = rate_eq.Ieff_nu0(
                              n_levels=n_levels,Iext_nu0=rate_eq.Iext_nu0,
                              beta_nu0=beta_nu0,S_dust_nu0=rate_eq.S_dust_nu0,
                              trans_low_number=trans_low_number,
                              trans_up_number=trans_up_number,
                              tau_dust_nu0=rate_eq.tau_dust_nu0,
                              tau_tot_nu0=tau_tot_nu0)
                mixed_term_nu0  = rate_eq.mixed_term_nu0(
                                     n_levels=n_levels,beta_nu0=beta_nu0,
                                     trans_low_number=trans_low_number,
                                     trans_up_number=trans_up_number,
                                     tau_tot_nu0=tau_tot_nu0,tau_line_nu0=tau_line_nu0,
                                     A21=rate_eq.molecule.A21)
                expected_GammaR = rate_eq.U_nu0+rate_eq.V_nu0*Ieff_nu0-mixed_term_nu0
            else:
                tau_line_functions = rate_eq.get_tau_line_functions(
                                                     level_population=level_pop)
                tau_tot_functions = rate_eq.get_tau_tot_functions(
                                                      level_population=level_pop)
                VIeff = rate_eq.V_Ieff_averaged(tau_tot_functions=tau_tot_functions)
                mixed = rate_eq.mixed_term_averaged(tau_tot_functions=tau_tot_functions,
                                                    tau_line_functions=tau_line_functions)
                expected_GammaR = rate_eq.U_nu0+VIeff-mixed
            expected_GammaR = np.transpose(expected_GammaR)
            expected_diag = -(np.sum(expected_GammaR,axis=0)-expected_GammaR.diagonal())
            np.fill_diagonal(a=expected_GammaR,val=expected_diag)
            assert np.all(expected_GammaR==rate_eq.GammaR(level_population=level_pop))


def test_square_line_profile_averaging():
    #for a molecule without overlap, nu0 and averaging should give same results
    test_molecule = molecule.EmittingMolecule(
                          datafilepath=os.path.join(here,'LAMDA_files/co.dat'),
                          line_profile_type='rectangular',
                          width_v=1*constants.kilo)
    collider_densities = {'ortho-H2':1e3/constants.centi**3,
                          'para-H2':1e4/constants.centi**3}
    def T_dust(nu):
        return np.ones_like(nu)*200
    def tau_dust(nu):
        return np.ones_like(nu)*0.01
    kwargs = {'molecule':test_molecule,'collider_densities':collider_densities,
              'Tkin':100,'geometry':geometry,'N':1e16/constants.centi**2,
              'ext_background':cmb,'T_dust':T_dust,'tau_dust':tau_dust}
    rate_eq_nu0 = rate_equations.RateEquations(**kwargs,treat_line_overlap=False)
    rate_eq_avg = rate_equations.RateEquations(**kwargs,treat_line_overlap=True)
    level_pop = test_molecule.LTE_level_pop(20)
    tau_line_nu0 = rate_eq_nu0.tau_line_nu0(
                       level_population=level_pop,
                       nlow_rad_transitions=rate_eq_nu0.molecule.nlow_rad_transitions,
                       nup_rad_transitions=rate_eq_nu0.molecule.nup_rad_transitions,
                       N=rate_eq_nu0.N,A21=rate_eq_nu0.molecule.A21,
                       phi_nu0=rate_eq_nu0.molecule.phi_nu0,
                       gup_rad_transitions=rate_eq_nu0.molecule.gup_rad_transitions,
                       glow_rad_transitions=rate_eq_nu0.molecule.glow_rad_transitions,
                       nu0=rate_eq_nu0.molecule.nu0)
    tau_tot_nu0 = tau_line_nu0 + rate_eq_nu0.tau_dust_nu0
    beta_nu0 = rate_eq_nu0.geometry.beta(tau_tot_nu0)
    n_levels = rate_eq_nu0.molecule.n_levels
    trans_low_number = rate_eq_nu0.molecule.nlow_rad_transitions
    trans_up_number = rate_eq_nu0.molecule.nup_rad_transitions
    Ieff_nu0 = rate_eq_nu0.Ieff_nu0(
                  n_levels=n_levels,Iext_nu0=rate_eq_nu0.Iext_nu0,beta_nu0=beta_nu0,
                  S_dust_nu0=rate_eq_nu0.S_dust_nu0,trans_low_number=trans_low_number,
                  trans_up_number=trans_up_number,tau_dust_nu0=rate_eq_nu0.tau_dust_nu0,
                  tau_tot_nu0=tau_tot_nu0)
    V_Ieff_nu0 = rate_eq_nu0.V_nu0*Ieff_nu0
    tau_tot_functions = rate_eq_avg.get_tau_tot_functions(
                                          level_population=level_pop)
    V_Ieff_averaged = rate_eq_avg.V_Ieff_averaged(tau_tot_functions=tau_tot_functions)
    mixed_term_nu0  = rate_eq_nu0.mixed_term_nu0(
                         n_levels=n_levels,beta_nu0=beta_nu0,
                         trans_low_number=trans_low_number,
                         trans_up_number=trans_up_number,
                         tau_tot_nu0=tau_tot_nu0,tau_line_nu0=tau_line_nu0,
                         A21=rate_eq_nu0.molecule.A21)
    tau_line_functions = rate_eq_avg.get_tau_line_functions(
                                         level_population=level_pop)
    mixed_term_averaged = rate_eq_avg.mixed_term_averaged(
                            tau_tot_functions=tau_tot_functions,
                            tau_line_functions=tau_line_functions)
    GammaR_nu0 = rate_eq_nu0.GammaR(level_population=level_pop)
    GammaR_averaged = rate_eq_avg.GammaR(level_population=level_pop)
    for X_nu0,X_averaged in [(V_Ieff_nu0,V_Ieff_averaged),
                             (mixed_term_nu0,mixed_term_averaged),
                             (GammaR_nu0,GammaR_averaged)]:
        assert np.allclose(X_nu0,X_averaged,atol=0,rtol=1e-3)