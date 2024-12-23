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

here = os.path.dirname(os.path.abspath(__file__))
dummy_line_profile_type = 'Gaussian'
dummy_width_v = 1*constants.kilo
geometry = escape_probability.UniformSphere()

#take a molecule with many colliders on purpose
test_molecule = molecule.EmittingMolecule(
                      datafilepath=os.path.join(here,'LAMDA_files/o.dat'),
                      line_profile_type=dummy_line_profile_type,
                      width_v=dummy_width_v)
LTE_level_pop = test_molecule.LTE_level_pop(T=123)
collider_densities = {'para-H2':1e4/constants.centi**3,
                      'e':1e2/constants.centi**3,
                      'H':1.2e4/constants.centi**3}
Tkin = 150
N = 1e15/constants.centi**-2


def generate_rate_eq(treat_line_overlap,ext_background,T_dust,tau_dust):
    return rate_equations.RateEquations(
                      molecule=test_molecule,collider_densities=collider_densities,
                      Tkin=Tkin,treat_line_overlap=treat_line_overlap,
                      geometry=geometry,N=N,ext_background=ext_background,
                      T_dust=T_dust,tau_dust=tau_dust)

def T_dust(nu):
    return np.ones_like(nu)*200

def tau_dust(nu):
    return np.ones_like(nu)*0.5

cmb = helpers.generate_CMB_background()

def rate_eq_iterator():
    for treat_line_overlap,ext_bg in\
                              itertools.product([True,False],['zero',cmb]):
        for Td,taud in zip(('zero',T_dust),('zero',tau_dust)):
            yield generate_rate_eq(treat_line_overlap=treat_line_overlap,
                                    ext_background=ext_bg,T_dust=T_dust,tau_dust=tau_dust)

def get_zero_tau_rate_eq(ext_background):
    return rate_equations.RateEquations(
                  molecule=test_molecule,collider_densities=collider_densities,
                  Tkin=Tkin,treat_line_overlap=True,
                  geometry=geometry,N=0,ext_background=ext_background,
                  T_dust='zero',tau_dust='zero')

def test_collider_is_requested():
    for rate_eq in rate_eq_iterator():
        for coll in collider_densities.keys():
            assert rate_eq.collider_is_requested(coll)
        for coll in test_molecule.ordered_colliders:
            if not coll in collider_densities:
                assert not rate_eq.collider_is_requested(coll)

def test_get_collider_selection():
    expected_ordered_colliders = ['H','H+','e','ortho-H2','para-H2']
    assert test_molecule.ordered_colliders\
                                       == expected_ordered_colliders
    expected_collider_selection = np.array((True,False,True,False,True))
    for rate_eq in rate_eq_iterator():
        assert np.all(expected_collider_selection==rate_eq.get_collider_selection())

def test_get_collider_densities_list():
    for rate_eq in rate_eq_iterator():
        expected_collider_densities_list = []
        for coll in rate_eq.molecule.ordered_colliders:
            if coll in collider_densities:
                expected_collider_densities_list.append(collider_densities[coll])
            else:
                expected_collider_densities_list.append(np.inf)
        assert np.all(expected_collider_densities_list
                      ==list(rate_eq.get_collider_densities_list()))

def test_coll_rate_matrix():
    #write a slow "for" loop to calculate the rate matrix and compare to the fast
    #loop used in the code
    expected_GammaC = np.zeros((test_molecule.n_levels,test_molecule.n_levels))
    for collider,coll_density in collider_densities.items():
        coll_transitions = test_molecule.coll_transitions[collider]
        for trans in coll_transitions:
            n_up = trans.up.number
            n_low = trans.low.number
            K12,K21 = trans.coeffs(Tkin=Tkin)
            expected_GammaC[n_up,n_low] += K12*coll_density
            expected_GammaC[n_low,n_low] += -K12*coll_density
            expected_GammaC[n_low,n_up] += K21*coll_density
            expected_GammaC[n_up,n_up] += -K21*coll_density
    for rate_eq in rate_eq_iterator():
        assert np.allclose(expected_GammaC,rate_eq.GammaC,atol=0,rtol=1e-10)

def test_zero_functions():
    def non_zero(nu):
        return np.ones_like(nu)
    rate_eq = rate_equations.RateEquations(
                   molecule=test_molecule,collider_densities=collider_densities,
                   Tkin=Tkin,treat_line_overlap=False,geometry=geometry,N=N,
                   ext_background=non_zero,T_dust=non_zero,tau_dust=non_zero)
    test_nu0 = rate_eq.molecule.nu0
    for func in ('ext_background','T_dust','tau_dust'):
        assert np.all(getattr(rate_eq,func)(test_nu0)==1)
    rate_eq.set_ext_background(ext_background='zero')
    assert np.all(rate_eq.ext_background(test_nu0)==0)
    rate_eq.set_dust(T_dust='zero',tau_dust='zero')
    for func in ('T_dust','tau_dust'):
        assert np.all(getattr(rate_eq,func)(test_nu0)==0)

def test_U_nu0():
    expected_U = np.zeros((test_molecule.n_levels,test_molecule.n_levels))
    for trans in test_molecule.rad_transitions:
        expected_U[trans.up.number,trans.low.number] = trans.A21
    for rate_eq in rate_eq_iterator():
        assert np.all(expected_U==rate_eq.U_nu0)

def test_V_nu0():
    expected_V = np.zeros((test_molecule.n_levels,test_molecule.n_levels))
    for trans in test_molecule.rad_transitions:
        expected_V[trans.up.number,trans.low.number] = trans.B21
        expected_V[trans.low.number,trans.up.number] = trans.B12
    for rate_eq in rate_eq_iterator():
        assert np.all(expected_V==rate_eq.V_nu0)

def test_Ieff_nu0():
    general_kwargs = {'n_levels':test_molecule.n_levels,
                      'trans_low_number':test_molecule.nlow_rad_transitions,
                      'trans_up_number':test_molecule.nup_rad_transitions}
    level_pop = LTE_level_pop
    for rate_eq in rate_eq_iterator():
        Iext_nu0 = rate_eq.ext_background(test_molecule.nu0)
        tau_tot_kwargs = {'level_population':level_pop,'N':N,'tau_dust':rate_eq.tau_dust}
        tau_tot_nu0 = [test_molecule.get_tau_tot_nu(line_index=i,**tau_tot_kwargs)(line.nu0)
                       for i,line in enumerate(test_molecule.rad_transitions)]
        tau_tot_nu0 = np.array(tau_tot_nu0)
        beta_nu0 = geometry.beta(tau_tot_nu0)
        S_dust_nu0 = helpers.B_nu(nu=test_molecule.nu0,T=T_dust(test_molecule.nu0))
        tau_dust_nu0 = tau_dust(test_molecule.nu0)
        Ieff = rate_eq.Ieff_nu0(**general_kwargs,Iext_nu0=Iext_nu0,beta_nu0=beta_nu0,
                                S_dust_nu0=S_dust_nu0,tau_dust_nu0=tau_dust_nu0,
                                tau_tot_nu0=tau_tot_nu0)
        expected_Ieff = np.zeros((test_molecule.n_levels,)*2)
        for t,trans in enumerate(test_molecule.rad_transitions):
            nup,nlow = trans.up.number,trans.low.number
            expected_Ieff[nup,nlow] = expected_Ieff[nlow,nup]\
                  = beta_nu0[t]*Iext_nu0[t]\
                        + (1-beta_nu0[t])*S_dust_nu0[t]*tau_dust_nu0[t]/tau_tot_nu0[t]
        assert np.all(Ieff==expected_Ieff)

def test_Ieff_nu0_zero_tau():
    general_kwargs = {'n_levels':test_molecule.n_levels,
                      'trans_low_number':test_molecule.nlow_rad_transitions,
                      'trans_up_number':test_molecule.nup_rad_transitions}
    rate_eq = get_zero_tau_rate_eq(ext_background=cmb)
    Iext_nu0 = rate_eq.ext_background(test_molecule.nu0)
    tau_tot_nu0 = np.zeros(rate_eq.molecule.n_rad_transitions)
    beta_nu0 = geometry.beta(tau_tot_nu0)
    S_dust_nu0 = np.zeros_like(tau_tot_nu0)
    tau_dust_nu0 = np.zeros_like(tau_tot_nu0)
    Ieff = rate_eq.Ieff_nu0(
              **general_kwargs,Iext_nu0=Iext_nu0,beta_nu0=beta_nu0,
              S_dust_nu0=S_dust_nu0,tau_dust_nu0=tau_dust_nu0,tau_tot_nu0=tau_tot_nu0)
    expected_Ieff = np.zeros((test_molecule.n_levels,)*2)
    for line in rate_eq.molecule.rad_transitions:
        nup,nlow = line.up.number,line.low.number
        expected_Ieff[nup,nlow] = cmb(line.nu0)
        expected_Ieff[nlow,nup] = expected_Ieff[nup,nlow]
    assert np.all(Ieff==expected_Ieff)

def test_mixed_term_nu0():
    A21 = test_molecule.A21
    level_pop = LTE_level_pop
    general_kwargs = {'n_levels':test_molecule.n_levels,'A21':A21,
                      'trans_low_number':test_molecule.nlow_rad_transitions,
                      'trans_up_number':test_molecule.nup_rad_transitions}
    for rate_eq in rate_eq_iterator():
        tau_tot_kwargs = {'level_population':level_pop,'N':N,'tau_dust':rate_eq.tau_dust}
        tau_tot_nu0 = [test_molecule.get_tau_tot_nu(line_index=i,**tau_tot_kwargs)(line.nu0)
                       for i,line in enumerate(test_molecule.rad_transitions)]
        tau_tot_nu0 = np.array(tau_tot_nu0)
        beta_nu0 = geometry.beta(tau_tot_nu0)
        tau_line_nu0 = test_molecule.get_tau_nu0(N=N,level_population=level_pop)
        mixed_term = rate_eq.mixed_term_nu0(
                                         **general_kwargs,tau_tot_nu0=tau_tot_nu0,
                                         beta_nu0=beta_nu0,tau_line_nu0=tau_line_nu0)
        expected_mixed_term = np.zeros((test_molecule.n_levels,)*2)
        for t,trans in enumerate(test_molecule.rad_transitions):
            nup,nlow = trans.up.number,trans.low.number
            expected_mixed_term[nup,nlow] = (1-beta_nu0[t])/tau_tot_nu0[t]\
                                                    *tau_line_nu0[t]*A21[t]
        assert np.all(mixed_term==expected_mixed_term)

def test_mixed_term_nu0_zero_tau():
    rate_eq = get_zero_tau_rate_eq(ext_background=cmb)
    A21 = test_molecule.A21
    general_kwargs = {'n_levels':test_molecule.n_levels,'A21':A21,
                      'trans_low_number':test_molecule.nlow_rad_transitions,
                      'trans_up_number':test_molecule.nup_rad_transitions}
    tau_tot_nu0 = np.zeros((test_molecule.n_rad_transitions)*2)
    beta_nu0 = geometry.beta(tau_tot_nu0)
    tau_line_nu0 = tau_tot_nu0.copy()
    mixed_term = rate_eq.mixed_term_nu0(
                                     **general_kwargs,tau_tot_nu0=tau_tot_nu0,
                                     beta_nu0=beta_nu0,tau_line_nu0=tau_line_nu0)
    assert np.all(mixed_term==0)

def test_GammaR_nu0():
    for rate_eq in rate_eq_iterator():
        if rate_eq.treat_line_overlap:
            continue
        GammaR = rate_eq.GammaR_nu0(level_population=LTE_level_pop)
        expected_diag = -(GammaR.sum(axis=0)-GammaR.diagonal())
        assert np.all(GammaR.diagonal()==expected_diag)

def test_tau_tot_functions():
    for rate_eq in rate_eq_iterator():
        tau_tot_funcs = rate_eq.get_tau_tot_functions(level_population=LTE_level_pop)
        for i,line in enumerate(rate_eq.molecule.rad_transitions):
            width_nu = rate_eq.molecule.width_v/constants.c*line.nu0
            nu = np.linspace(line.nu0-width_nu,line.nu0+width_nu,100)
            expected_tau_tot = test_molecule.get_tau_tot_nu(
                                line_index=i,level_population=LTE_level_pop,N=N,
                                tau_dust=rate_eq.tau_dust)(nu)
            assert np.all(tau_tot_funcs[i](nu)==expected_tau_tot)
    zero_tau_rate_eq = get_zero_tau_rate_eq(ext_background=cmb)
    tau_tot_funcs = zero_tau_rate_eq.get_tau_tot_functions(level_population=LTE_level_pop)
    for i,line in enumerate(zero_tau_rate_eq.molecule.rad_transitions):
        width_nu = rate_eq.molecule.width_v/constants.c*line.nu0
        nu = np.linspace(line.nu0-width_nu,line.nu0+width_nu,100)
        assert np.all(tau_tot_funcs[i](nu)==0)

def test_tau_line_functions():
    for rate_eq in rate_eq_iterator():
        tau_line_funcs = rate_eq.get_tau_line_functions(level_population=LTE_level_pop)
        for i,line in enumerate(rate_eq.molecule.rad_transitions):
            width_nu = rate_eq.molecule.width_v/constants.c*line.nu0
            nu = np.linspace(line.nu0-width_nu,line.nu0+width_nu,100)
            expected_tau_line = test_molecule.get_tau_line_nu(
                                line_index=i,level_population=LTE_level_pop,N=N)(nu)
            assert np.all(tau_line_funcs[i](nu)==expected_tau_line)
    zero_tau_rate_eq = get_zero_tau_rate_eq(ext_background=help)
    tau_line_funcs = zero_tau_rate_eq.get_tau_line_functions(
                                             level_population=LTE_level_pop)
    for i,line in enumerate(zero_tau_rate_eq.molecule.rad_transitions):
        width_nu = rate_eq.molecule.width_v/constants.c*line.nu0
        nu = np.linspace(line.nu0-width_nu,line.nu0+width_nu,100)
        assert np.all(tau_line_funcs[i](nu)==0)

def test_V_Ieff_averaged_zero_N():
    level_pop = LTE_level_pop
    rate_eq = get_zero_tau_rate_eq(ext_background=cmb)
    tau_tot_functions = rate_eq.get_tau_tot_functions(level_population=level_pop)
    V_Ieff = rate_eq.V_Ieff_averaged(tau_tot_functions=tau_tot_functions)
    expected_V_Ieff = np.zeros((test_molecule.n_levels,)*2)
    for line in test_molecule.rad_transitions:
        nup,nlow = line.up.number,line.low.number
        Ieff = line.line_profile.average_over_phi_nu(cmb)
        expected_V_Ieff[nup,nlow] = Ieff*line.B21
        expected_V_Ieff[nlow,nup] = Ieff*line.B12
    assert np.all(V_Ieff==expected_V_Ieff)

def test_V_Ieff_averaged():
    level_pop = LTE_level_pop
    for rate_eq in rate_eq_iterator():
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

                  # for rate_eq in rate_eq_iterator():
    #     tau_tot_funcs = rate_eq.get_tau_tot_functions(level_population=level_pop)
        
        

#TODO write test for physics (see below)

'''
A21 = np.array([line.A21 for line in test_molecule.rad_transitions])
B21 = np.array([line.B21 for line in test_molecule.rad_transitions])
B12 = np.array([line.B12 for line in test_molecule.rad_transitions])
Tkin = 100
LTE_level_pop_Tkin = test_molecule.LTE_level_pop(Tkin)

collider_densities_large = {'para-H2':1e10/constants.centi**3,
                            'ortho-H2':1e10/constants.centi**3}                                  
collider_densities_0 = {'para-H2':0,'ortho-H2':0}
Jbar_0 = np.zeros(test_molecule.n_rad_transitions)
A21_factor_thin = np.ones(test_molecule.n_rad_transitions)
B21_factor_thin = np.zeros(test_molecule.n_rad_transitions)
beta_thin = np.ones(test_molecule.n_rad_transitions)
T_dust = 123
Bnu_dust = np.array([helpers.B_nu(nu=trans.nu0,T=T_dust) for trans in
                 test_molecule.rad_transitions])
LTE_level_pop_Tdust = test_molecule.LTE_level_pop(T_dust)
T_I_ext = 234
LTE_level_pop_T_I_ext = test_molecule.LTE_level_pop(T_I_ext)
Iext = np.array([helpers.B_nu(nu=trans.nu0,T=T_I_ext) for trans in
                 test_molecule.rad_transitions])

#following functions test the physics of RateEquations

def solve_for_level_pops(collider_densities,Jbar,A21_factor,B21_factor):
    kwargs = {'molecule':test_molecule,'collider_densities':collider_densities,
              'Tkin':Tkin}
    Einstein_kwargs = {'A21':A21,'B12':B12,'B21':B21}
    rate_equations_LI = rate_equations.RateEquations(mode='LI',**kwargs)
    level_pop_LI = rate_equations_LI.solve(Jbar=Jbar,**Einstein_kwargs)
    rate_equations_ALI = rate_equations.RateEquations(mode='ALI',**kwargs)
    level_pop_ALI = rate_equations_ALI.solve(
                          A21_factor=A21_factor,B21_factor=B21_factor,**Einstein_kwargs)
    return level_pop_LI,level_pop_ALI

def test_compute_level_populations_no_excitation():
    level_pops = solve_for_level_pops(collider_densities=collider_densities_0,
                                      Jbar=Jbar_0,
                                      A21_factor=A21_factor_thin,
                                      B21_factor=B21_factor_thin)
    expected_level_pop = np.zeros(len(test_molecule.levels))
    expected_level_pop[0] = 1
    for level_pop in level_pops:
        assert np.all(level_pop==expected_level_pop)

def wrapper_for_testing(collider_densities,Jbar,A21_factor,
                        B21_factor,expected_level_pop,rtol=1e-3):
    level_pops = solve_for_level_pops(collider_densities=collider_densities,
                                      Jbar=Jbar,A21_factor=A21_factor,
                                      B21_factor=B21_factor)
    for level_pop in level_pops:
        assert np.allclose(level_pop,expected_level_pop,rtol=rtol,atol=0)

def test_compute_level_populations_LTE_from_coll():
    wrapper_for_testing(collider_densities=collider_densities_large,
                        Jbar=Jbar_0,A21_factor=A21_factor_thin,
                        B21_factor=B21_factor_thin,
                        expected_level_pop=LTE_level_pop_Tkin)

def test_compute_level_populations_LTE_from_external_rad():
    Jbar = beta_thin*Iext
    B21_factor = beta_thin*Iext #no overlapping lines
    wrapper_for_testing(collider_densities=collider_densities_0,Jbar=Jbar,
                        A21_factor=A21_factor_thin,
                        B21_factor=B21_factor,
                        expected_level_pop=LTE_level_pop_T_I_ext)

def test_compute_level_populations_LTE_external_rad_and_coll():
    #assume external field has same temperature as dust, otherwise I don't know
    #what the LTE temperature should be...
    Iext_Tkin = np.array([helpers.B_nu(nu=trans.nu0,T=Tkin) for trans in
                          test_molecule.rad_transitions])
    Jbar = beta_thin*Iext_Tkin
    B21_factor = beta_thin*Iext_Tkin #no overlapping lines
    wrapper_for_testing(collider_densities=collider_densities_large,
                        Jbar=Jbar,A21_factor=A21_factor_thin,
                        B21_factor=B21_factor,
                        expected_level_pop=LTE_level_pop_Tkin)

def test_LTE_from_thick_internal_dust():
    Jbar = Bnu_dust
    A21_factor = A21_factor_thin #assume gas is thin
    #for B21 factor, assume dust is completely dominating tau
    B21_factor = Bnu_dust
    wrapper_for_testing(collider_densities=collider_densities_0,Jbar=Jbar,
                        A21_factor=A21_factor,B21_factor=B21_factor,
                        expected_level_pop=LTE_level_pop_Tdust)

def test_LTE_thick_dust_and_coll():
    Bnu_dust_Tkin = np.array([helpers.B_nu(nu=trans.nu0,T=Tkin) for trans in
                              test_molecule.rad_transitions])
    Jbar = Bnu_dust_Tkin
    A21_factor = A21_factor_thin
    B21_factor = Bnu_dust_Tkin
    wrapper_for_testing(collider_densities=collider_densities_large,Jbar=Jbar,
                        A21_factor=A21_factor,B21_factor=B21_factor,
                        expected_level_pop=LTE_level_pop_Tkin)
'''