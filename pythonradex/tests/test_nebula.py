# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:30:44 2017

@author: gianni
"""
import os
from pythonradex import nebula,helpers
from scipy import constants
import numpy as np
import itertools


here = os.path.dirname(os.path.abspath(__file__))
datafilepath = os.path.join(here,'LAMDA_files/co.dat')
Ntot = 1e15/constants.centi**2
width_v = 2*constants.kilo
ext_background = helpers.generate_CMB_background()

line_profiles = ('rectangular','Gaussian')
geometries = tuple(nebula.Nebula.geometries.keys())
iteration_modes = ('ALI','std')
use_ng_options = (True,False)
average_beta_options = (True,False)

def nebula_params_iterator():
    return itertools.product(geometries,line_profiles,iteration_modes,
                             use_ng_options,average_beta_options)

def generate_new_nebulae():
    return [nebula.Nebula(datafilepath=datafilepath,geometry=geo,
                          line_profile=lp,width_v=width_v,iteration_mode=mode,
                          use_NG_acceleration=use_ng,
                          average_beta_over_line_profile=average_beta)
            for geo,lp,mode,use_ng,average_beta in nebula_params_iterator()]


def test_set_cloud_parameters():
    nebulae = generate_new_nebulae()
    collider_densities = {'para-H2':1}
    standard_params = {'ext_background':ext_background,'Tkin':20,
                       'collider_densities':collider_densities,'Ntot':Ntot}
    def verify_nebula_params(neb,params):
        for p in ('Ntot','collider_densities','Ntot'):
            assert getattr(neb,p) == params[p]
        I_ext_lines = np.array([params['ext_background'](line.nu0) for line in
                                neb.emitting_molecule.rad_transitions])
        assert np.all(I_ext_lines==neb.I_ext_lines)
    for neb in nebulae:
        for i in range(2):
            #two times the because the first time is special (initial setting of params)
            neb.set_cloud_parameters(**standard_params)
            verify_nebula_params(neb,standard_params)
        new_params = {'ext_background':lambda nu: ext_background(nu)/2,'Ntot':4*Ntot,
                      'Tkin':4*standard_params['Tkin']}
        for param_name,new_value in new_params.items():
            changed_params = standard_params.copy()
            changed_params[param_name] = new_value
            neb.set_cloud_parameters(**changed_params)
            verify_nebula_params(neb,changed_params)
        new_collider_densities = [standard_params['collider_densities'] | {'ortho-H2':200},
                                  {'para-H2':standard_params['collider_densities']['para-H2']*2},
                                  {'ortho-H2':300}]
        for new_coll_densities in new_collider_densities:
            changed_params = standard_params.copy()
            changed_params['collider_densities'] = new_coll_densities
            neb.set_cloud_parameters(**changed_params)
            verify_nebula_params(neb,changed_params)
            #test also change of colliders and Tkin at the same time:
            changed_params = standard_params.copy()
            changed_params['collider_densities'] = new_coll_densities
            changed_params['Tkin'] = 2*standard_params['Tkin']
            neb.set_cloud_parameters(**changed_params)
            verify_nebula_params(neb,changed_params)

Ntot_values = np.logspace(12,16,4)/constants.centi**2
ext_backgrounds = [ext_background,lambda nu: 0,lambda nu: ext_background(nu)/2]
Tkins = np.linspace(20,200,4)
coll_density_values = np.logspace(2,7,4)/constants.centi**3
collider_cases = [['ortho-H2'],['para-H2'],['ortho-H2','para-H2']]
def param_iterator():
    return itertools.product(Ntot_values,ext_backgrounds,Tkins,
                                       coll_density_values,collider_cases)

def test_set_cloud_parameters_with_physics():
    neb_kwargs = {'datafilepath':datafilepath,'geometry':'uniform sphere',
                  'line_profile':'rectangular','width_v':width_v,'iteration_mode':'ALI',
                  'use_NG_acceleration':True,'average_beta_over_line_profile':False}
    nebula_to_modify = nebula.Nebula(**neb_kwargs)
    def generate_new_parametrised_nebula(params):
        neb = nebula.Nebula(**neb_kwargs)
        neb.set_cloud_parameters(**params)
        return neb
    for Ntot,ext_b,Tkin,coll_dens,colliders in param_iterator():
        params = {'Ntot':Ntot,'Tkin':Tkin,'ext_background':ext_b}
        #put different values in case more than one collider (not really necessary...)
        coll_densities = {collider:coll_dens*(i+1) for i,collider in
                          enumerate(colliders)}
        params['collider_densities'] = coll_densities
        nebula_to_modify.set_cloud_parameters(**params)
        reference_neb = generate_new_parametrised_nebula(params=params)
        nebula_to_modify.solve_radiative_transfer()
        reference_neb.solve_radiative_transfer()
        assert np.all(nebula_to_modify.level_pop==reference_neb.level_pop)

def test_beta_alllines():
    T = 45
    N = 1e14/constants.centi**2
    for geo,avg_beta,lp in itertools.product(geometries,average_beta_options,line_profiles):
        neb = nebula.Nebula(datafilepath=datafilepath,geometry=geo,
                            line_profile=lp,width_v=width_v,iteration_mode='ALI',
                            use_NG_acceleration=True,
                            average_beta_over_line_profile=avg_beta)
        neb.set_cloud_parameters(ext_background=ext_background,Ntot=N,Tkin=T,
                                 collider_densities={'para-H2':1e4/constants.centi**3})
        level_pop = neb.emitting_molecule.LTE_level_pop(T=T)
        expected_beta = []
        for line in neb.emitting_molecule.rad_transitions:
            n_low = line.low.number
            n_up = line.up.number
            N1 = N*level_pop[n_low]
            N2 = N*level_pop[n_up]
            if avg_beta:
                nu = line.line_profile.coarse_nu_array
                phi_nu = line.line_profile.coarse_phi_nu_array
                tau_nu = line.tau_nu(N1=N1,N2=N2,nu=nu)
                beta = neb.geometry.beta(tau_nu)
                beta = np.trapz(beta*phi_nu,nu)/np.trapz(phi_nu,nu)
                expected_beta.append(beta)
            else:
                tau_nu0 = line.tau_nu0(N1=N1,N2=N2)
                beta = neb.geometry.beta(np.array((tau_nu0,)))
                expected_beta.append(beta[0])
        assert np.allclose(expected_beta,neb.beta_alllines(level_populations=level_pop),
                           atol=0,rtol=1e-4)
    
def test_sourcefunction_alllines():
    T = 123
    for neb in generate_new_nebulae():
        level_pop = neb.emitting_molecule.LTE_level_pop(T=T)
        expected_source_func = []
        for line in neb.emitting_molecule.rad_transitions:
            n_low = line.low.number
            x1 = level_pop[n_low]
            n_up = line.up.number
            x2 = level_pop[n_up]
            s = line.A21*x2/(x1*line.B12-x2*line.B21)
            expected_source_func.append(s)
        neb_source_func = neb.source_function_alllines(level_populations=level_pop)
        assert np.all(expected_source_func==neb_source_func)

def test_ng_acceleration():
    neb_kwargs = {'datafilepath':datafilepath,'geometry':'uniform sphere',
                  'line_profile':'rectangular','width_v':width_v,'iteration_mode':'ALI',
                  'average_beta_over_line_profile':False}
    for Ntot,ext_b,Tkin,coll_dens,colliders in param_iterator():
        level_pops = []
        params = {'Ntot':Ntot,'Tkin':Tkin,'ext_background':ext_b}
        #put different values in case more than one collider (not really necessary...)
        coll_densities = {collider:coll_dens*(i+1) for i,collider in
                          enumerate(colliders)}
        params['collider_densities'] = coll_densities
        for ng in (True,False):
            neb = nebula.Nebula(use_NG_acceleration=ng,**neb_kwargs)
            neb.set_cloud_parameters(**params)
            neb.solve_radiative_transfer()
            level_pops.append(neb.level_pop)
        np.allclose(*level_pops,atol=0,rtol=1e-3)

def test_compute_residual():
    min_tau = nebula.Nebula.min_tau_considered_for_convergence
    small_tau = np.array((0.7*min_tau,min_tau/2,min_tau/100,min_tau/500,min_tau/1.01))
    tau = np.array((1,10,2*min_tau,min_tau,min_tau/2))
    n_relevant_taus = (tau>min_tau).sum()
    Tex_residual = np.array((1,2,3,4,5))
    assert nebula.Nebula.compute_residual(Tex_residual=Tex_residual,tau_lines=small_tau,
                                          min_tau_considered_for_convergence=min_tau) == 0
    expected_residual = np.sum(Tex_residual[tau>min_tau])/n_relevant_taus
    assert nebula.Nebula.compute_residual(Tex_residual=Tex_residual,tau_lines=tau,
                                          min_tau_considered_for_convergence=min_tau)\
            == expected_residual
    
def test_radiative_transfer():
    Tkin = 150
    N_small = 1e11/constants.centi**2
    N_medium = 1e15/constants.centi**2
    N_large = 1e18/constants.centi**2
    collider_density_small = {'ortho-H2':1/constants.centi**3}
    collider_density_large = {'ortho-H2':1e11/constants.centi**3}
    LTE_background = lambda nu: helpers.B_nu(nu=nu,T=Tkin)
    for geo,lp,use_ng,average_beta in itertools.product(geometries,line_profiles,
                                                        use_ng_options,average_beta_options):
        neb = nebula.Nebula(
                      datafilepath=datafilepath,geometry=geo,
                      line_profile=lp,width_v=width_v,iteration_mode='ALI',
                      use_NG_acceleration=use_ng,average_beta_over_line_profile=average_beta)
        LTE_level_pop = neb.emitting_molecule.LTE_level_pop(T=Tkin)
        def check_LTE(neb):
            neb.solve_radiative_transfer()
            assert np.allclose(neb.level_pop,LTE_level_pop,atol=0,rtol=1e-2)
            assert np.allclose(neb.Tex,Tkin,atol=0,rtol=1e-2)
        for N in (N_small,N_medium,N_large):
            neb.set_cloud_parameters(ext_background=ext_background,Ntot=N,Tkin=Tkin,
                                     collider_densities=collider_density_large)
            check_LTE(neb)
            neb.set_cloud_parameters(ext_background=LTE_background,Ntot=N,Tkin=Tkin,
                                     collider_densities=collider_density_small)
            check_LTE(neb)

def test_flux_thin():
    Tkin = 45
    Ntot = 1e12/constants.centi**2
    collider_densities = {'ortho-H2':1e4/constants.centi**3}
    distance = 1*constants.parsec
    sphere_radius = 1*constants.au
    sphere_volume = 4/3*sphere_radius**3*np.pi
    sphere_Omega = sphere_radius**2*np.pi/distance**2
    Omega = sphere_Omega #use the same Omega for all geometries
    for lp,avg_beta in itertools.product(line_profiles,average_beta_options):
        for geometry in nebula.Nebula.geometries.keys():
            neb = nebula.Nebula(
                      datafilepath=datafilepath,geometry=geometry,
                      line_profile=lp,width_v=width_v,iteration_mode='ALI',
                      use_NG_acceleration=True,average_beta_over_line_profile=False)
            neb.set_cloud_parameters(ext_background=helpers.zero_background,Ntot=Ntot,
                                     Tkin=Tkin,collider_densities=collider_densities)
            neb.solve_radiative_transfer()
            neb.compute_line_fluxes(solid_angle=Omega)
            expected_fluxes = []
            expected_spectra = []
            for i,line in enumerate(neb.emitting_molecule.rad_transitions):
                n_up = line.up.number
                up_level_pop = neb.level_pop[n_up]
                if geometry == 'uniform sphere':
                    #in the case of the uniform sphere, we can do an elegant test
                    #using physics
                    number_density = Ntot/(2*sphere_radius)
                    total_mol = number_density*sphere_volume
                    flux = total_mol*up_level_pop*line.A21*line.Delta_E/(4*np.pi*distance**2)
                else:
                    #for slabs, could not come up with elegant test
                    flux_nu0 = helpers.B_nu(nu=line.nu0,T=neb.Tex[i]) * neb.tau_nu0[i]\
                               *Omega
                    if lp == 'Gaussian':
                        flux = np.sqrt(2*np.pi)*line.line_profile.sigma_nu*flux_nu0
                    elif lp == 'rectangular':
                        flux = flux_nu0*line.line_profile.width_nu
                expected_fluxes.append(flux)
                phi_nu = line.line_profile.dense_phi_nu_array
                nu = line.line_profile.dense_nu_array
                spectrum = phi_nu*flux/np.trapz(phi_nu,nu)
                expected_spectra.append(spectrum)
            expected_fluxes = np.array(expected_fluxes)
            assert np.allclose(neb.obs_line_fluxes,expected_fluxes,atol=0,rtol=3e-2)
            assert np.allclose(neb.tau_nu0,0,atol=1e-3,rtol=0)
            for i in range(neb.emitting_molecule.n_rad_transitions):
                assert np.allclose(expected_spectra[i],neb.obs_line_spectra[i],
                                   atol=0,rtol=3e-2)
                assert np.allclose(neb.tau_nu_lines[i],0,atol=1e-3,rtol=0)

def test_thick_LTE_flux():
    Tkin = 45
    Ntot = 1e19/constants.centi**2
    collider_densities = {'ortho-H2':1e10/constants.centi**3}
    distance = 1*constants.parsec
    Omega = 1*constants.au**2/distance**2
    for lp,avg_beta in itertools.product(line_profiles,average_beta_options):
        for geometry in nebula.Nebula.geometries.keys():
            neb = nebula.Nebula(
                      datafilepath=datafilepath,geometry=geometry,
                      line_profile=lp,width_v=width_v,iteration_mode='ALI',
                      use_NG_acceleration=True,average_beta_over_line_profile=False)
            neb.set_cloud_parameters(ext_background=helpers.zero_background,Ntot=Ntot,
                                     Tkin=Tkin,collider_densities=collider_densities)
            neb.set_cloud_parameters(ext_background=helpers.zero_background,Ntot=Ntot,
                                      Tkin=Tkin,collider_densities=collider_densities)
            neb.solve_radiative_transfer()
            neb.compute_line_fluxes(solid_angle=Omega)
            thick_lines = neb.tau_nu0 > 10
            assert thick_lines.sum() >= 10
            for i,line in enumerate(neb.emitting_molecule.rad_transitions):
                if not thick_lines[i]:
                    continue
                bb_flux_nu0 = helpers.B_nu(nu=line.nu0,T=Tkin)*Omega
                peak_flux = np.max(neb.obs_line_spectra[i])
                assert np.isclose(a=peak_flux,b=bb_flux_nu0,atol=0,rtol=3e-2)
                if lp == 'rectangular':
                    expected_total_flux = bb_flux_nu0*line.line_profile.width_nu
                    assert np.isclose(a=expected_total_flux,b=neb.obs_line_fluxes[i],
                                      atol=0,rtol=3e-2)
                else:
                    assert lp == 'Gaussian'

def test_print_results():
    for neb in generate_new_nebulae():
        neb.set_cloud_parameters(ext_background=ext_background,
                                  Ntot=1e14/constants.centi**2,Tkin=33.33,
                                  collider_densities={'ortho-H2':1e3/constants.centi**3})
        neb.solve_radiative_transfer()
        neb.print_results()