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


geometries = list(nebula.Nebula.geometries.keys())
Ntot = 1e15/constants.centi**2
line_profile = 'square'
width_v = 2*constants.kilo
ext_background = helpers.CMB_background
nebulae = {}
for geo in geometries:
    nebulae[geo] = {}
    nebulae[geo]['general'] = nebula.Nebula(
                             data_filepath=lamda_filepath,geometry=geo,
                             ext_background=ext_background,Tkin=Tkin,
                             coll_partner_densities=coll_partner_densities_default,
                             Ntot=Ntot,line_profile=line_profile,width_v=width_v)
    nebulae[geo]['LTE'] = nebula.Nebula(
                             data_filepath=lamda_filepath,geometry=geo,
                             ext_background=ext_background,Tkin=Tkin,
                             coll_partner_densities=coll_partner_densities_large,
                             Ntot=Ntot,line_profile=line_profile,width_v=width_v)
    nebulae[geo]['thin_LTE'] = nebula.Nebula(
                             data_filepath=lamda_filepath,geometry=geo,
                             ext_background=ext_background,Tkin=Tkin,
                             coll_partner_densities=coll_partner_densities_large,
                             Ntot=Ntot/1e6,line_profile=line_profile,width_v=width_v)
    nebulae[geo]['thick_LTE'] = nebula.Nebula(
                             data_filepath=lamda_filepath,geometry=geo,
                             ext_background=ext_background,Tkin=Tkin,
                             coll_partner_densities=coll_partner_densities_large,
                             Ntot=Ntot*1e10,line_profile=line_profile,width_v=width_v)

for neb in nebulae.values():
    for n in neb.values():
        n.solve_radiative_transfer()

def test_solve_radiative_transfer():
    for geo,neb in nebulae.items():
        for mode in ('LTE','thin_LTE','thick_LTE'):
            assert np.allclose(neb[mode].level_pop,expected_LTE_level_pop,rtol=1e-2)
            assert np.allclose(neb[mode].Tex,Tkin,rtol=1e-2)
        assert np.allclose(neb['thin_LTE'].tau_nu0,0,atol=1e-2)

r = 1
S = 1 #surface of the slab
def get_surface(geo):
    if 'sphere' in geo:
        return 4*np.pi*r**2
    elif 'slab' in geo:
        return 2*S #the slab has two sides

def test_compute_line_fluxes():
    for geo,neb in nebulae.items():
        if 'RADEX' in geo:
            continue
        Ntot = neb['thin_LTE'].Ntot
        if 'sphere' in geo:
            volume = 4/3*r**3*np.pi
            surface = get_surface(geo)
            n = Ntot/(2*r)
        elif 'slab' in geo:
            volume = r*S
            surface = get_surface(geo)
            n = Ntot/r
        tot_particles = n*volume
        up_level_particles = np.array(
                              [tot_particles*expected_LTE_level_pop[trans.up.number]
                              for trans in neb['thin_LTE'].emitting_molecule.rad_transitions])
        A21 = [trans.A21 for trans in neb['thin_LTE'].emitting_molecule.rad_transitions]
        Delta_E = [trans.Delta_E for trans in neb['thin_LTE'].emitting_molecule.rad_transitions]
        expected_LTE_fluxes_thin = up_level_particles*A21*Delta_E/surface
        assert np.allclose(neb['thin_LTE'].line_fluxes,expected_LTE_fluxes_thin,
                           atol=0,rtol=1e-2)
        #the higher transitions might not be optically thick, so just test the
        #lowest transition:
        lowest_trans = neb['thick_LTE'].emitting_molecule.rad_transitions[0]
        expected_thick_LTE_flux_lowest_trans = np.pi*helpers.B_nu(nu=lowest_trans.nu0,T=Tkin)\
                                               * lowest_trans.line_profile.width_nu
        assert np.isclose(neb['thick_LTE'].line_fluxes[0],
                           expected_thick_LTE_flux_lowest_trans,atol=0,rtol=1e-2)

def test_determine_observed_fluxes():
    geo = 'uniform sphere'
    neb = nebulae[geo]
    line_fluxes = neb['general'].line_fluxes
    surface = get_surface(geo)
    obs_fluxes = neb['general'].observed_fluxes(
                       source_surface=surface,d_observer=r)
    assert np.allclose(obs_fluxes,line_fluxes,atol=0)

def test_energy_conservation():
    distance = 1
    for geo,neb in nebulae.items():
        surface = get_surface(geo)
        line_fluxes = neb['general'].line_fluxes
        emitted_energy = np.array(line_fluxes)*surface
        obs_fluxes = neb['general'].observed_fluxes(
                       source_surface=surface,d_observer=distance)
        assert np.allclose(emitted_energy,obs_fluxes*4*np.pi*distance**2,atol=0)

def test_print_results():
    for neb in nebulae.values():
        neb['general'].print_results()