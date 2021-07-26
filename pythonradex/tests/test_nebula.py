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

test_molecule = molecule.Molecule.from_LAMDA_datafile(datafilepath=lamda_filepath)
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
ext_background = helpers.generate_CMB_background()
nebulae = {}

r = 10*constants.au
slab_surface = (10*constants.au)**2 #surface of the slab
slab_depth = 0.1*constants.au
d = 1*constants.parsec
def get_solid_angle(geo):
    if 'sphere' in geo:
        Omega = np.pi*r**2/d**2
    elif 'slab' in geo:
        Omega = slab_surface/d**2
    else:
        raise ValueError('geo: {:s}'.format(geo))
    return Omega

for geo in geometries:
    nebulae[geo] = {}
    nebulae[geo]['general'] = nebula.Nebula(
                             datafilepath=lamda_filepath,geometry=geo,
                             ext_background=ext_background,Tkin=Tkin,
                             coll_partner_densities=coll_partner_densities_default,
                             Ntot=Ntot,line_profile=line_profile,width_v=width_v)
    nebulae[geo]['LTE'] = nebula.Nebula(
                             datafilepath=lamda_filepath,geometry=geo,
                             ext_background=ext_background,Tkin=Tkin,
                             coll_partner_densities=coll_partner_densities_large,
                             Ntot=Ntot,line_profile=line_profile,width_v=width_v)
    nebulae[geo]['thin_LTE'] = nebula.Nebula(
                             datafilepath=lamda_filepath,geometry=geo,
                             ext_background=ext_background,Tkin=Tkin,
                             coll_partner_densities=coll_partner_densities_large,
                             Ntot=Ntot/1e6,line_profile=line_profile,width_v=width_v)
    nebulae[geo]['thick_LTE'] = nebula.Nebula(
                             datafilepath=lamda_filepath,geometry=geo,
                             ext_background=ext_background,Tkin=Tkin,
                             coll_partner_densities=coll_partner_densities_large,
                             Ntot=Ntot*1e10,line_profile=line_profile,width_v=width_v)

for geo,neb in nebulae.items():
    for n in neb.values():
        n.solve_radiative_transfer()
        n.compute_line_fluxes(solid_angle=get_solid_angle(geo))

def test_solve_radiative_transfer():
    for geo,neb in nebulae.items():
        for mode in ('LTE','thin_LTE','thick_LTE'):
            assert np.allclose(neb[mode].level_pop,expected_LTE_level_pop,rtol=1e-2)
            assert np.allclose(neb[mode].Tex,Tkin,rtol=1e-2)
        assert np.allclose(neb['thin_LTE'].tau_nu0,0,atol=1e-2)

def test_compute_line_fluxes():
    for geo,neb in nebulae.items():
        if 'RADEX' in geo or 'slab' in geo:
            continue
        Ntot = neb['thin_LTE'].Ntot
        if 'sphere' in geo:
            volume = 4/3*r**3*np.pi
            n = Ntot/(2*r)
        elif 'slab' in geo:
            volume = slab_surface*slab_depth
            n = Ntot/slab_depth
        tot_particles = n*volume
        thin_LTE_rad_transitions = neb['thin_LTE'].emitting_molecule.rad_transitions
        up_level_particles = np.array(
                              [tot_particles*expected_LTE_level_pop[trans.up.number]
                              for trans in thin_LTE_rad_transitions])
        A21 = np.array([trans.A21 for trans in thin_LTE_rad_transitions])
        Delta_E = np.array([trans.Delta_E for trans in thin_LTE_rad_transitions])
        energy_flux = up_level_particles*A21*Delta_E #W
        if 'sphere' in geo:
            expected_LTE_fluxes_thin = energy_flux/(4*np.pi*d**2)
        elif 'slab' in geo:
            #not isotropic, so I can't do the same thing as for sphere
            expected_LTE_fluxes_thin = energy_flux/(4*np.pi*slab_surface)\
                                        *get_solid_angle(geo)
        line_profile_nu0 = np.array([trans.line_profile.phi_nu(trans.nu0) for
                                     trans in thin_LTE_rad_transitions])
        expected_LTE_fluxes_nu_thin = expected_LTE_fluxes_thin*line_profile_nu0
        assert np.allclose(neb['thin_LTE'].obs_line_fluxes,expected_LTE_fluxes_thin,
                           atol=0,rtol=1e-2)
        line_fluxes_nu_thin = [np.max(fluxes_nu) for fluxes_nu in
                               neb['thin_LTE'].obs_line_spectra]
        assert np.allclose(line_fluxes_nu_thin,expected_LTE_fluxes_nu_thin,
                           atol=0,rtol=1e-2)
        #the higher transitions might not be optically thick, so just test the
        #lowest transition:
        lowest_trans = neb['thick_LTE'].emitting_molecule.rad_transitions[0]
        expected_thick_LTE_flux_nu = helpers.B_nu(nu=lowest_trans.nu0,T=Tkin)\
                                      *get_solid_angle(geo)
        expected_thick_LTE_flux = expected_thick_LTE_flux_nu\
                                       * lowest_trans.line_profile.width_nu
        assert np.isclose(neb['thick_LTE'].obs_line_fluxes[0],
                           expected_thick_LTE_flux,atol=0,rtol=1e-2)

def test_print_results():
    for neb in nebulae.values():
        neb['general'].print_results()