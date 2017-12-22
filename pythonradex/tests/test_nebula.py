# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:30:44 2017

@author: gianni
"""
import os
from pythonradex import nebula,molecule,helpers
from scipy import constants
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
lamda_filepath = os.path.join(here,'co.dat')

test_molecule = molecule.Molecule.from_LAMDA_datafile(data_filepath=lamda_filepath)
Tkin = 100
expected_LTE_level_pop = test_molecule.LTE_level_pop(Tkin)
coll_partner_densities_default = {'para-H2':1e1/constants.centi**3,
                                  'ortho-H2':1e1/constants.centi**3}
coll_partner_densities_large = {'para-H2':1e10/constants.centi**3,
                                'ortho-H2':1e10/constants.centi**3}                                  
coll_partner_densities_0 = {'para-H2':0,'ortho-H2':0}
Jbar_lines_0 = np.zeros(test_molecule.n_rad_transitions)
Jbar_lines_B_nu = [helpers.B_nu(nu=trans.nu0,T=Tkin) for trans in
                   test_molecule.rad_transitions]
beta_lines_thin = np.ones(test_molecule.n_rad_transitions)
I_ext_lines_0 = np.zeros(test_molecule.n_rad_transitions)

def get_level_pops(coll_partner_densities,Jbar_lines,beta_lines,I_ext_lines):
    kwargs = {'molecule':test_molecule,
              'coll_partner_densities':coll_partner_densities,'Tkin':Tkin}
    rate_equations_std = nebula.RateEquations(mode='std',**kwargs)
    level_pop_std = rate_equations_std.solve(Jbar_lines=Jbar_lines)
    rate_equations_ALI = nebula.RateEquations(mode='ALI',**kwargs)
    level_pop_ALI = rate_equations_ALI.solve(
                          beta_lines=beta_lines,I_ext_lines=I_ext_lines)
    return level_pop_std,level_pop_ALI

def test_compute_level_populations_no_excitation():
    level_pops = get_level_pops(coll_partner_densities=coll_partner_densities_0,
                                Jbar_lines=Jbar_lines_0,beta_lines=beta_lines_thin,
                                I_ext_lines=I_ext_lines_0)
    expected_level_pop = np.zeros(len(test_molecule.levels))
    expected_level_pop[0] = 1
    for level_pop in level_pops:
        assert np.all(level_pop==expected_level_pop)

def test_compute_level_populations_LTE_from_coll():
    level_pops = get_level_pops(coll_partner_densities=coll_partner_densities_large,
                                Jbar_lines=Jbar_lines_0,beta_lines=beta_lines_thin,
                                I_ext_lines=I_ext_lines_0)
    for level_pop in level_pops:
        assert np.allclose(level_pop,expected_LTE_level_pop,rtol=1e-2,atol=0)

def test_compute_level_populations_LTE_from_rad():
    level_pops = get_level_pops(coll_partner_densities=coll_partner_densities_0,
                                Jbar_lines=Jbar_lines_B_nu,beta_lines=beta_lines_thin,
                                I_ext_lines=Jbar_lines_B_nu)
    for level_pop in level_pops:
        assert np.allclose(level_pop,expected_LTE_level_pop,rtol=1e-3,atol=0)

def test_compute_level_populations_LTE_rad_and_coll():
    level_pops = get_level_pops(coll_partner_densities=coll_partner_densities_large,
                                Jbar_lines=Jbar_lines_B_nu,beta_lines=beta_lines_thin,
                                I_ext_lines=Jbar_lines_B_nu)
    for level_pop in level_pops:
        assert np.allclose(level_pop,expected_LTE_level_pop,rtol=1e-3,atol=0)


geometry = 'uniform sphere'
Ntot = 1e15/constants.centi**2
line_profile = 'square'
width_v = 2*constants.kilo
ext_background = helpers.CMB_background
general_test_nebula = nebula.Nebula(
                         data_filepath=lamda_filepath,geometry=geometry,
                         ext_background=ext_background,Tkin=Tkin,
                         coll_partner_densities=coll_partner_densities_default,
                         Ntot=Ntot,line_profile=line_profile,width_v=width_v)
LTE_test_nebula = nebula.Nebula(
                         data_filepath=lamda_filepath,geometry=geometry,
                         ext_background=ext_background,Tkin=Tkin,
                         coll_partner_densities=coll_partner_densities_large,
                         Ntot=Ntot,line_profile=line_profile,width_v=width_v)
thin_LTE_test_nebula = nebula.Nebula(
                         data_filepath=lamda_filepath,geometry=geometry,
                         ext_background=ext_background,Tkin=Tkin,
                         coll_partner_densities=coll_partner_densities_large,
                         Ntot=Ntot/1e6,line_profile=line_profile,width_v=width_v)
thick_LTE_test_nebula = nebula.Nebula(
                         data_filepath=lamda_filepath,geometry=geometry,
                         ext_background=ext_background,Tkin=Tkin,
                         coll_partner_densities=coll_partner_densities_large,
                         Ntot=Ntot*1e10,line_profile=line_profile,width_v=width_v)

def test_solve_radiative_transfer():
    LTE_test_nebula.solve_radiative_transfer()
    assert np.allclose(LTE_test_nebula.level_pop,expected_LTE_level_pop,rtol=1e-2)
    assert np.allclose(LTE_test_nebula.Tex,Tkin,rtol=1e-2)
    thin_LTE_test_nebula.solve_radiative_transfer()
    assert np.allclose(thin_LTE_test_nebula.level_pop,expected_LTE_level_pop,rtol=1e-2)
    assert np.allclose(thin_LTE_test_nebula.Tex,Tkin,rtol=1e-2)
    assert np.allclose(thin_LTE_test_nebula.tau_nu0,0,atol=1e-2)

r = 1
def test_compute_line_fluxes():
    n = thin_LTE_test_nebula.Ntot/(2*r)
    tot_particles = n*4/3*r**3*np.pi
    up_level_particles = np.array([tot_particles*expected_LTE_level_pop[trans.up.number]
                                   for trans in thin_LTE_test_nebula.emitting_molecule.rad_transitions])
    A21 = [trans.A21 for trans in thin_LTE_test_nebula.emitting_molecule.rad_transitions]
    Delta_E = [trans.Delta_E for trans in thin_LTE_test_nebula.emitting_molecule.rad_transitions]
    expected_LTE_fluxes_thin = up_level_particles*A21*Delta_E/(4*np.pi*r**2)
    assert np.allclose(thin_LTE_test_nebula.line_fluxes,expected_LTE_fluxes_thin,
                       atol=0,rtol=1e-2)
    thick_LTE_test_nebula.solve_radiative_transfer()
    #the higher transitions might not be optically thick, so just test the lowest transition:
    lowest_trans = thick_LTE_test_nebula.emitting_molecule.rad_transitions[0]
    expected_thick_LTE_flux_lowest_trans = np.pi*helpers.B_nu(nu=lowest_trans.nu0,T=Tkin)\
                                           * lowest_trans.line_profile.width_nu
    assert np.isclose(thick_LTE_test_nebula.line_fluxes[0],
                       expected_thick_LTE_flux_lowest_trans,atol=0,rtol=1e-2)

def test_determine_observed_fluxes():
    general_test_nebula.solve_radiative_transfer()
    line_fluxes = general_test_nebula.line_fluxes
    surface = 4*np.pi*r**2
    obs_fluxes = general_test_nebula.observed_fluxes(
                       source_surface=surface,d_observer=r)
    assert np.allclose(obs_fluxes,line_fluxes,atol=0)

def test_print_results():
    general_test_nebula.print_results()