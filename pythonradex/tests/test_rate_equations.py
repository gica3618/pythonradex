#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:04:44 2024

@author: gianni
"""

import os
from scipy import constants
from pythonradex import radiative_transfer,molecule,helpers
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
dummy_line_profile_type = 'Gaussian'
dummy_width_v = 1*constants.kilo

##############  Tests for RateEquations ###################

#take a molecule with many colliders on purpose
test_molecule = molecule.EmittingMolecule(
                      datafilepath=os.path.join(here,'LAMDA_files/o.dat'),
                      line_profile_type=dummy_line_profile_type,
                      width_v=dummy_width_v)
A21_lines = np.array([line.A21 for line in test_molecule.rad_transitions])
B21_lines = np.array([line.B21 for line in test_molecule.rad_transitions])
B12_lines = np.array([line.B12 for line in test_molecule.rad_transitions])
Tkin = 100
expected_LTE_level_pop = test_molecule.LTE_level_pop(Tkin)
collider_densities = {'para-H2':1e4/constants.centi**3,
                      'e':1e2/constants.centi**3,
                      'H':1.2e4/constants.centi**3}
collider_densities_large = {'para-H2':1e10/constants.centi**3,
                            'ortho-H2':1e10/constants.centi**3}                                  
collider_densities_0 = {'para-H2':0,'ortho-H2':0}
Jbar_lines_0 = np.zeros(test_molecule.n_rad_transitions)
Jbar_lines_B_nu = np.array([helpers.B_nu(nu=trans.nu0,T=Tkin) for trans in
                            test_molecule.rad_transitions])
beta_lines_thin = np.ones(test_molecule.n_rad_transitions)
betaIext_lines_0 = np.zeros(test_molecule.n_rad_transitions)


def test_RateEquations_constructor():
    expected_ordered_colliders = ['H','H+','e','ortho-H2','para-H2']
    assert test_molecule.ordered_colliders\
                                   == expected_ordered_colliders
    expected_collider_selection = np.array((True,False,True,False,True))
    expected_collider_densities_list = []
    for coll in expected_ordered_colliders:
        if coll in collider_densities:
            expected_collider_densities_list.append(collider_densities[coll])
        else:
            expected_collider_densities_list.append(np.inf)
    for mode in ('LI','ALI'):
        rate_eq = radiative_transfer.RateEquations(
                            molecule=test_molecule,
                            collider_densities=collider_densities,
                            Tkin=50,mode=mode)
        assert np.all(np.array(rate_eq.collider_selection) == expected_collider_selection)
        assert np.all(np.array(rate_eq.collider_densities_list)
                      ==expected_collider_densities_list)
        
def test_coll_rate_matrix():
    #write a slow for loop to calculate the rate matrix and compare to the fast
    #loop used in the code
    T = 100
    expected_coll_rate_matrix = np.zeros((test_molecule.n_levels,test_molecule.n_levels))
    for collider,coll_density in collider_densities.items():
        coll_transitions = test_molecule.coll_transitions[collider]
        for trans in coll_transitions:
            n_up = trans.up.number
            n_low = trans.low.number
            K12,K21 = trans.coeffs(Tkin=T)
            expected_coll_rate_matrix[n_up,n_low] += K12*coll_density
            expected_coll_rate_matrix[n_low,n_low] += -K12*coll_density
            expected_coll_rate_matrix[n_low,n_up] += K21*coll_density
            expected_coll_rate_matrix[n_up,n_up] += -K21*coll_density
    for mode in ('LI','ALI'):
        rate_eq = radiative_transfer.RateEquations(
                            molecule=test_molecule,
                            collider_densities=collider_densities,
                            Tkin=T,mode=mode)
        assert np.allclose(expected_coll_rate_matrix,rate_eq.coll_rate_matrix,
                           atol=0,rtol=1e-10)

#following functions test the physics of RateEquations

def solve_for_level_pops(collider_densities,Jbar_lines,beta_lines,betaIext_lines):
    kwargs = {'molecule':test_molecule,'collider_densities':collider_densities,
              'Tkin':Tkin}
    Einstein_kwargs = {'A21_lines':A21_lines,'B12_lines':B12_lines,
                       'B21_lines':B21_lines}
    rate_equations_LI = radiative_transfer.RateEquations(mode='LI',**kwargs)
    level_pop_LI = rate_equations_LI.solve(Jbar_lines=Jbar_lines,**Einstein_kwargs)
    rate_equations_ALI = radiative_transfer.RateEquations(mode='ALI',**kwargs)
    level_pop_ALI = rate_equations_ALI.solve(
                          beta_lines=beta_lines,betaIext_lines=betaIext_lines,
                          **Einstein_kwargs)
    return level_pop_LI,level_pop_ALI

def test_compute_level_populations_no_excitation():
    level_pops = solve_for_level_pops(collider_densities=collider_densities_0,
                                      Jbar_lines=Jbar_lines_0,beta_lines=beta_lines_thin,
                                      betaIext_lines=betaIext_lines_0)
    expected_level_pop = np.zeros(len(test_molecule.levels))
    expected_level_pop[0] = 1
    for level_pop in level_pops:
        assert np.all(level_pop==expected_level_pop)

def test_compute_level_populations_LTE_from_coll():
    level_pops = solve_for_level_pops(collider_densities=collider_densities_large,
                                      Jbar_lines=Jbar_lines_0,beta_lines=beta_lines_thin,
                                      betaIext_lines=betaIext_lines_0)
    for level_pop in level_pops:
        assert np.allclose(level_pop,expected_LTE_level_pop,rtol=1e-2,atol=0)

def test_compute_level_populations_LTE_from_rad():
    betaIext_lines = beta_lines_thin*Jbar_lines_B_nu
    level_pops = solve_for_level_pops(collider_densities=collider_densities_0,
                                      Jbar_lines=Jbar_lines_B_nu,beta_lines=beta_lines_thin,
                                      betaIext_lines=betaIext_lines)
    for level_pop in level_pops:
        assert np.allclose(level_pop,expected_LTE_level_pop,rtol=1e-3,atol=0)

def test_compute_level_populations_LTE_rad_and_coll():
    betaIext_lines = beta_lines_thin*Jbar_lines_B_nu
    level_pops = solve_for_level_pops(collider_densities=collider_densities_large,
                                      Jbar_lines=Jbar_lines_B_nu,beta_lines=beta_lines_thin,
                                      betaIext_lines=betaIext_lines)
    for level_pop in level_pops:
        assert np.allclose(level_pop,expected_LTE_level_pop,rtol=1e-3,atol=0)