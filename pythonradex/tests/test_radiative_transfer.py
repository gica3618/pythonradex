# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:30:44 2017

@author: gianni
"""
import os
from pythonradex import radiative_transfer,helpers
from scipy import constants
import numpy as np
import itertools
import pytest
import numba as nb


here = os.path.dirname(os.path.abspath(__file__))
datafilepath = os.path.join(here,'LAMDA_files/co.dat')
N = 1e15/constants.centi**2
width_v = 2*constants.kilo
ext_background = helpers.generate_CMB_background()

line_profile_types = ('rectangular','Gaussian')
geometries = tuple(radiative_transfer.Cloud.geometries.keys())
iteration_modes = ('ALI','LI')
use_ng_options = (True,False)
average_beta_options = (True,False)

def cloud_params_iterator():
    return itertools.product(geometries,line_profile_types,iteration_modes,
                             use_ng_options,average_beta_options)

def generate_new_clouds():
    return [radiative_transfer.Cloud(
                          datafilepath=datafilepath,geometry=geo,
                          line_profile_type=lp,width_v=width_v,iteration_mode=mode,
                          use_NG_acceleration=use_ng,
                          average_beta_over_line_profile=average_beta)
            for geo,lp,mode,use_ng,average_beta in cloud_params_iterator()]

def test_warning_overlapping_lines():
    with pytest.warns(UserWarning):
        radiative_transfer.Cloud(
                              datafilepath=os.path.join(here,'LAMDA_files/hcl.dat'),
                              geometry='uniform sphere',
                              line_profile_type='rectangular',width_v=10*constants.kilo,
                              iteration_mode='ALI',use_NG_acceleration=True,
                              average_beta_over_line_profile=True)

def test_too_large_width_v():
    width_vs = np.array((1e4,1.2e4,1e5))*constants.kilo
    for width_v in width_vs:
        for geo,lp,mode,use_ng,average_beta in cloud_params_iterator():
            with pytest.raises(AssertionError):
                radiative_transfer.Cloud(
                                  datafilepath=datafilepath,geometry=geo,
                                  line_profile_type=lp,width_v=width_v,iteration_mode=mode,
                                  use_NG_acceleration=use_ng,
                                  average_beta_over_line_profile=average_beta)

def test_set_parameters():
    clouds = generate_new_clouds()
    collider_densities = {'para-H2':1}
    standard_params = {'ext_background':ext_background,'Tkin':20,
                        'collider_densities':collider_densities,'N':N}
    def verify_cloud_params(cloud,params):
        for p in ('N','collider_densities','N'):
            assert getattr(cloud,p) == params[p]
        I_ext_lines = np.array([params['ext_background'](line.nu0) for line in
                                cloud.emitting_molecule.rad_transitions])
        assert np.all(I_ext_lines==cloud.I_ext_lines)
    for cloud in clouds:
        for i in range(2):
            #two times the because the first time is special (initial setting of params)
            cloud.set_parameters(**standard_params)
            verify_cloud_params(cloud,standard_params)
        new_params = {'ext_background':lambda nu: ext_background(nu)/2,'N':4*N,
                      'Tkin':4*standard_params['Tkin']}
        for param_name,new_value in new_params.items():
            changed_params = standard_params.copy()
            changed_params[param_name] = new_value
            cloud.set_parameters(**changed_params)
            verify_cloud_params(cloud,changed_params)
        new_collider_densities = [standard_params['collider_densities'] | {'ortho-H2':200},
                                  {'para-H2':standard_params['collider_densities']['para-H2']*2},
                                  {'ortho-H2':300}]
        for new_coll_densities in new_collider_densities:
            changed_params = standard_params.copy()
            changed_params['collider_densities'] = new_coll_densities
            cloud.set_parameters(**changed_params)
            verify_cloud_params(cloud,changed_params)
            #test also change of colliders and Tkin at the same time:
            changed_params = standard_params.copy()
            changed_params['collider_densities'] = new_coll_densities
            changed_params['Tkin'] = 2*standard_params['Tkin']
            cloud.set_parameters(**changed_params)
            verify_cloud_params(cloud,changed_params)

N_values = np.logspace(12,16,4)/constants.centi**2
ext_backgrounds = [ext_background,lambda nu: 0,lambda nu: ext_background(nu)/2]
Tkins = np.linspace(20,200,4)
coll_density_values = np.logspace(2,7,4)/constants.centi**3
collider_cases = [['ortho-H2'],['para-H2'],['ortho-H2','para-H2']]
def param_iterator():
    return itertools.product(N_values,ext_backgrounds,Tkins,coll_density_values,
                             collider_cases)

@pytest.mark.filterwarnings("ignore:negative optical depth")
def test_set_parameters_with_physics():
    cloud_kwargs = {'datafilepath':datafilepath,'geometry':'uniform sphere',
                    'line_profile_type':'rectangular','width_v':width_v,'iteration_mode':'ALI',
                    'use_NG_acceleration':True,'average_beta_over_line_profile':False}
    cloud_to_modify = radiative_transfer.Cloud(**cloud_kwargs)
    def generate_new_parametrised_cloud(params):
        cloud = radiative_transfer.Cloud(**cloud_kwargs)
        cloud.set_parameters(**params)
        return cloud
    for N,ext_b,Tkin,coll_dens,colliders in param_iterator():
        params = {'N':N,'Tkin':Tkin,'ext_background':ext_b}
        #put different values in case more than one collider (not really necessary...)
        coll_densities = {collider:coll_dens*(i+1) for i,collider in
                          enumerate(colliders)}
        params['collider_densities'] = coll_densities
        cloud_to_modify.set_parameters(**params)
        reference_cloud = generate_new_parametrised_cloud(params=params)
        cloud_to_modify.solve_radiative_transfer()
        reference_cloud.solve_radiative_transfer()
        assert np.all(cloud_to_modify.level_pop==reference_cloud.level_pop)

def test_beta_alllines():
    T = 45
    N = 1e14/constants.centi**2
    for geo,avg_beta,lp in itertools.product(geometries,average_beta_options,line_profile_types):
        cloud = radiative_transfer.Cloud(
                            datafilepath=datafilepath,geometry=geo,
                            line_profile_type=lp,width_v=width_v,iteration_mode='ALI',
                            use_NG_acceleration=True,
                            average_beta_over_line_profile=avg_beta)
        cloud.set_parameters(ext_background=ext_background,N=N,Tkin=T,
                              collider_densities={'para-H2':1e4/constants.centi**3})
        level_pop = cloud.emitting_molecule.LTE_level_pop(T=T)
        expected_beta = []
        for line in cloud.emitting_molecule.rad_transitions:
            n_low = line.low.number
            n_up = line.up.number
            N1 = N*level_pop[n_low]
            N2 = N*level_pop[n_up]
            if avg_beta:
                nu = line.line_profile.coarse_nu_array
                phi_nu = line.line_profile.coarse_phi_nu_array
                tau_nu = line.tau_nu(N1=N1,N2=N2,nu=nu)
                beta = cloud.geometry.beta(tau_nu)
                beta = np.trapz(beta*phi_nu,nu)/np.trapz(phi_nu,nu)
                expected_beta.append(beta)
            else:
                tau_nu0 = line.tau_nu0(N1=N1,N2=N2)
                beta = cloud.geometry.beta(np.array((tau_nu0,)))
                expected_beta.append(beta[0])
        assert np.allclose(expected_beta,cloud.beta_alllines(level_populations=level_pop),
                            atol=0,rtol=1e-4)
    
def test_sourcefunction_alllines():
    T = 123
    for cloud in generate_new_clouds():
        level_pop = cloud.emitting_molecule.LTE_level_pop(T=T)
        expected_source_func = []
        for line in cloud.emitting_molecule.rad_transitions:
            n_low = line.low.number
            x1 = level_pop[n_low]
            n_up = line.up.number
            x2 = level_pop[n_up]
            s = line.A21*x2/(x1*line.B12-x2*line.B21)
            expected_source_func.append(s)
        cloud_source_func = cloud.source_function_alllines(level_populations=level_pop)
        assert np.all(expected_source_func==cloud_source_func)

@pytest.mark.filterwarnings("ignore:negative optical depth")
def test_ng_acceleration():
    cloud_kwargs = {'datafilepath':datafilepath,'geometry':'uniform sphere',
                    'line_profile_type':'rectangular','width_v':width_v,'iteration_mode':'ALI',
                    'average_beta_over_line_profile':False}
    for N,ext_b,Tkin,coll_dens,colliders in param_iterator():
        level_pops = []
        params = {'N':N,'Tkin':Tkin,'ext_background':ext_b}
        #put different values in case more than one collider (not really necessary...)
        coll_densities = {collider:coll_dens*(i+1) for i,collider in
                          enumerate(colliders)}
        params['collider_densities'] = coll_densities
        for ng in (True,False):
            cloud = radiative_transfer.Cloud(use_NG_acceleration=ng,**cloud_kwargs)
            cloud.set_parameters(**params)
            cloud.solve_radiative_transfer()
            level_pops.append(cloud.level_pop)
        np.allclose(*level_pops,atol=0,rtol=1e-3)

def test_compute_residual():
    min_tau = radiative_transfer.Cloud.min_tau_considered_for_convergence
    small_tau = np.array((0.7*min_tau,min_tau/2,min_tau/100,min_tau/500,min_tau/1.01))
    tau = np.array((1,10,2*min_tau,min_tau,min_tau/2))
    n_relevant_taus = (tau>min_tau).sum()
    Tex_residual = np.array((1,2,3,4,5))
    assert radiative_transfer.Cloud.compute_residual(
                                          Tex_residual=Tex_residual,tau_lines=small_tau,
                                          min_tau_considered_for_convergence=min_tau) == 0
    expected_residual = np.sum(Tex_residual[tau>min_tau])/n_relevant_taus
    assert radiative_transfer.Cloud.compute_residual(
                                          Tex_residual=Tex_residual,tau_lines=tau,
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
    for geo,lp,use_ng,average_beta in itertools.product(geometries,line_profile_types,
                                                        use_ng_options,average_beta_options):
        cloud = radiative_transfer.Cloud(
                      datafilepath=datafilepath,geometry=geo,
                      line_profile_type=lp,width_v=width_v,iteration_mode='ALI',
                      use_NG_acceleration=use_ng,average_beta_over_line_profile=average_beta)
        LTE_level_pop = cloud.emitting_molecule.LTE_level_pop(T=Tkin)
        def check_LTE(cloud):
            cloud.solve_radiative_transfer()
            assert np.allclose(cloud.level_pop,LTE_level_pop,atol=0,rtol=1e-2)
            assert np.allclose(cloud.Tex,Tkin,atol=0,rtol=1e-2)
        for N in (N_small,N_medium,N_large):
            cloud.set_parameters(ext_background=ext_background,N=N,Tkin=Tkin,
                                  collider_densities=collider_density_large)
            check_LTE(cloud)
            cloud.set_parameters(ext_background=LTE_background,N=N,Tkin=Tkin,
                                      collider_densities=collider_density_small)
            check_LTE(cloud)

@pytest.mark.filterwarnings("ignore:lines of input molecule are overlapping")
def test_fast_flux():
    test_widths_v = np.array((0.1,1,10,100,1000,9000))*constants.kilo
    solid_angle = 1
    Tex0 = 50
    tau_nu0_values = np.logspace(-3,3,10)
    test_transitions = nb.typed.List([2,4,5])
    for line_profile_type in ('rectangular','Gaussian'):
        for geometry,width_v in itertools.product(radiative_transfer.Cloud.geometries.keys(),
                                                  test_widths_v):
            cloud = radiative_transfer.Cloud(
                      datafilepath=datafilepath,geometry=geometry,
                      line_profile_type=line_profile_type,width_v=width_v,iteration_mode='ALI',
                      use_NG_acceleration=True,average_beta_over_line_profile=False)
            all_transitions = nb.typed.List(range(cloud.emitting_molecule.n_rad_transitions))
            nu0_lines = np.array([line.nu0 for line in cloud.emitting_molecule.rad_transitions])
            Tex = Tex0*np.ones(cloud.emitting_molecule.n_rad_transitions)
            for tau_nu0 in tau_nu0_values:
                tau_nu0_lines = np.ones_like(Tex)*tau_nu0
                kwargs = {'solid_angle':solid_angle,'tau_nu0_lines':tau_nu0_lines,
                          'Tex':Tex,'nu0_lines':nu0_lines,
                          'compute_flux_nu':cloud.geometry.compute_flux_nu,'width_v':width_v}
                if line_profile_type == 'rectangular':
                    all_fluxes = cloud.fast_fluxes_rectangular(
                                        **kwargs,transitions=all_transitions)
                    selected_fluxes = cloud.fast_fluxes_rectangular(
                                        **kwargs,transitions=test_transitions)
                elif line_profile_type == 'Gaussian':
                    all_fluxes = cloud.fast_fluxes_Gaussian(
                                          **kwargs,transitions=all_transitions)
                    selected_fluxes = cloud.fast_fluxes_Gaussian(
                                        **kwargs,transitions=test_transitions)
                else:
                    raise RuntimeError
                expected_fluxes = []
                for i,line in enumerate(cloud.emitting_molecule.rad_transitions):
                    nu0 = line.nu0
                    width_nu = cloud.emitting_molecule.width_v/constants.c*nu0
                    if line_profile_type == 'rectangular':
                        dense_nu = np.linspace(nu0-0.7*width_nu,nu0+0.7*width_nu,1000)
                    elif line_profile_type == 'Gaussian':
                        dense_nu = np.linspace(nu0-6*width_nu,nu0+6*width_nu,10000)
                    else:
                        raise RuntimeError
                    dense_phi_nu = line.line_profile.phi_nu(dense_nu)
                    expected_tau_nu = dense_phi_nu/dense_nu**2
                    nu0_index = np.argmin(np.abs(dense_nu-nu0))
                    norm = tau_nu0/expected_tau_nu[nu0_index]
                    expected_tau_nu *= norm
                    source_function = helpers.B_nu(T=Tex[i],nu=dense_nu)
                    expected_spec = cloud.geometry.compute_flux_nu(
                                      tau_nu=expected_tau_nu,source_function=source_function,
                                      solid_angle=solid_angle)
                    expected_flux = np.trapz(expected_spec,dense_nu)
                    expected_fluxes.append(expected_flux)
                atol = 0
                rtol = 1e-2
                assert np.allclose(all_fluxes, expected_fluxes,atol=atol,rtol=rtol)
                expected_selected_fluxes = [expected_fluxes[i] for i in test_transitions]
                assert np.allclose(selected_fluxes,expected_selected_fluxes,
                                    atol=atol,rtol=rtol)

def test_flux_thin():
    Tkin = 45
    N = 1e12/constants.centi**2
    collider_densities = {'ortho-H2':1e4/constants.centi**3}
    distance = 1*constants.parsec
    sphere_radius = 1*constants.au
    sphere_volume = 4/3*sphere_radius**3*np.pi
    sphere_Omega = sphere_radius**2*np.pi/distance**2
    Omega = sphere_Omega #use the same Omega for all geometries
    for lp,avg_beta in itertools.product(line_profile_types,average_beta_options):
        for geometry in radiative_transfer.Cloud.geometries.keys():
            cloud = radiative_transfer.Cloud(
                      datafilepath=datafilepath,geometry=geometry,
                      line_profile_type=lp,width_v=width_v,iteration_mode='ALI',
                      use_NG_acceleration=True,average_beta_over_line_profile=False)
            cloud.set_parameters(ext_background=helpers.zero_background,N=N,
                                  Tkin=Tkin,collider_densities=collider_densities)
            cloud.solve_radiative_transfer()
            fluxes = cloud.fluxes(solid_angle=Omega,transitions=None)
            expected_fluxes = []
            # expected_spectra = []
            for i,line in enumerate(cloud.emitting_molecule.rad_transitions):
                n_up = line.up.number
                up_level_pop = cloud.level_pop[n_up]
                if geometry == 'uniform sphere':
                    #in the case of the uniform sphere, we can do an elegant test
                    #using physics
                    number_density = N/(2*sphere_radius)
                    total_mol = number_density*sphere_volume
                    flux = total_mol*up_level_pop*line.A21*line.Delta_E/(4*np.pi*distance**2)
                else:
                    #for slabs, could not come up with elegant test
                    flux_nu0 = helpers.B_nu(nu=line.nu0,T=cloud.Tex[i]) * cloud.tau_nu0[i]\
                                *Omega
                    if lp == 'Gaussian':
                        flux = np.sqrt(2*np.pi)*line.line_profile.sigma_nu*flux_nu0
                    elif lp == 'rectangular':
                        flux = flux_nu0*line.line_profile.width_nu
                expected_fluxes.append(flux)
            expected_fluxes = np.array(expected_fluxes)
            assert np.allclose(fluxes,expected_fluxes,atol=0,rtol=3e-2)
            assert np.allclose(cloud.tau_nu0,0,atol=1e-3,rtol=0)

def test_thick_LTE_flux():
    Tkin = 45
    N = 1e19/constants.centi**2
    collider_densities = {'ortho-H2':1e10/constants.centi**3}
    distance = 1*constants.parsec
    Omega = 1*constants.au**2/distance**2
    for lp,avg_beta in itertools.product(line_profile_types,average_beta_options):
        for geometry in radiative_transfer.Cloud.geometries.keys():
            cloud = radiative_transfer.Cloud(
                      datafilepath=datafilepath,geometry=geometry,
                      line_profile_type=lp,width_v=width_v,iteration_mode='ALI',
                      use_NG_acceleration=True,average_beta_over_line_profile=False)
            cloud.set_parameters(ext_background=helpers.zero_background,N=N,
                                      Tkin=Tkin,collider_densities=collider_densities)
            cloud.solve_radiative_transfer()
            fluxes = cloud.fluxes(solid_angle=Omega,transitions=None)
            thick_lines = cloud.tau_nu0 > 10
            assert thick_lines.sum() >= 10
            for i,line in enumerate(cloud.emitting_molecule.rad_transitions):
                if not thick_lines[i]:
                    continue
                bb_flux_nu0 = helpers.B_nu(nu=line.nu0,T=Tkin)*Omega
                # peak_flux = np.max(cloud.obs_line_spectra[i])
                # assert np.isclose(a=peak_flux,b=bb_flux_nu0,atol=0,rtol=3e-2)
                if lp == 'rectangular':
                    expected_total_flux = bb_flux_nu0*line.line_profile.width_nu
                    assert np.isclose(a=expected_total_flux,b=fluxes[i],
                                      atol=0,rtol=3e-2)
                else:
                    assert lp == 'Gaussian'    

def test_tau_and_spectrum_single_lines():
    Tkin = 57
    solid_angle = 1
    collider_densities = {'ortho-H2':1e10/constants.centi**3}
    N_values = {'thin':1e12/constants.centi**2,
                'thick':1e19/constants.centi**2}
    trans_indices = (1,6)
    for lp,avg_beta in itertools.product(line_profile_types,average_beta_options):
        for geometry in radiative_transfer.Cloud.geometries.keys():
            cloud = radiative_transfer.Cloud(
                      datafilepath=datafilepath,geometry=geometry,
                      line_profile_type=lp,width_v=width_v,iteration_mode='ALI',
                      use_NG_acceleration=True,average_beta_over_line_profile=False)
            lines = [cloud.emitting_molecule.rad_transitions[i] for i in trans_indices]
            nu = np.array(())
            nu0_values = [line.nu0 for line in lines]
            for nu0 in nu0_values:
                width_nu = width_v/constants.c*nu0
                nu_values = np.linspace(nu0-2*width_nu,nu0+2*width_nu,100)
                nu = np.concatenate((nu,nu_values))
            for ID,N in N_values.items():
                cloud.set_parameters(ext_background=helpers.zero_background,N=N,
                                      Tkin=Tkin,collider_densities=collider_densities)
                cloud.solve_radiative_transfer()
                tau_nu = cloud.tau_nu(nu=nu)
                spectrum = cloud.spectrum(solid_angle=solid_angle,nu=nu)
                for line_index,line in zip(trans_indices,lines):
                    x1 = cloud.level_pop[line.low.number]
                    x2 = cloud.level_pop[line.up.number]
                    N1 = N*x1
                    N2 = N*x2
                    width_nu = width_v/constants.c*line.nu0
                    nu_selection = np.abs(nu-line.nu0)<5*width_nu
                    nu_line = nu[nu_selection]
                    expected_tau_nu = line.tau_nu(N1=N1,N2=N2,nu=nu_line)
                    expected_tau_nu0 = cloud.tau_nu0[line_index]
                    assert np.allclose(tau_nu[nu_selection],expected_tau_nu,atol=0,
                                        rtol=1e-3)
                    assert np.isclose(np.max(tau_nu[nu_selection]),expected_tau_nu0,
                                      atol=0,rtol=1e-2)
                    source_func = helpers.B_nu(T=cloud.Tex[line_index],nu=nu_line)
                    expected_spec = cloud.geometry.compute_flux_nu(
                                        tau_nu=expected_tau_nu,source_function=source_func,
                                        solid_angle=solid_angle)
                    assert np.allclose(spectrum[nu_selection],expected_spec,atol=0,
                                        rtol=1e-3)
                    if ID == 'thick':
                        bb_flux_nu0 = helpers.B_nu(nu=line.nu0,T=Tkin)*solid_angle
                        peak_flux = np.max(spectrum[nu_selection])
                        assert np.isclose(a=peak_flux,b=bb_flux_nu0,atol=0,rtol=3e-2)
                if ID == 'thin':
                    assert np.allclose(tau_nu,0,atol=1e-3,rtol=0)

@pytest.mark.filterwarnings("ignore:lines are overlapping")
@pytest.mark.filterwarnings("ignore:lines of input molecule are overlapping")
def test_tau_and_spectrum_overlapping_lines():
    datafilepath = os.path.join(here,'LAMDA_files/hcl.dat')
    solid_angle = 1
    Tkin = 102
    collider_densities = {'ortho-H2':1e10/constants.centi**3}
    N_values = {'thin':1e10/constants.centi**2,
                'thick':1e15/constants.centi**2}
    trans_indices = (0,1,2)
    width_v = 20*constants.kilo
    for lp,avg_beta in itertools.product(line_profile_types,average_beta_options):
        for geometry in radiative_transfer.Cloud.geometries.keys():
            cloud = radiative_transfer.Cloud(
                      datafilepath=datafilepath,geometry=geometry,
                      line_profile_type=lp,width_v=width_v,iteration_mode='ALI',
                      use_NG_acceleration=True,average_beta_over_line_profile=False)
            lines = [cloud.emitting_molecule.rad_transitions[i] for i in trans_indices]
            lines = sorted(lines,key=lambda l: l.nu0)
            assert len(lines) == 3
            #make sure lines are overlapping:
            for i in range(2):
                diff = lines[i+1].nu0-lines[i].nu0
                assert diff/lines[i].nu0*constants.c < width_v/2
            width_nu = width_v/constants.c*lines[2].nu0
            nu0s = [line.nu0 for line in lines]
            min_nu = np.min(nu0s) - 3*width_nu
            max_nu = np.max(nu0s) + 3*width_nu
            nu = np.linspace(min_nu,max_nu,400)
            for ID,N in N_values.items():
                cloud.set_parameters(ext_background=helpers.zero_background,N=N,
                                      Tkin=Tkin,collider_densities=collider_densities)
                cloud.solve_radiative_transfer()
                tau_nu = cloud.tau_nu(nu=nu)
                spectrum = cloud.spectrum(solid_angle=solid_angle,nu=nu)
                expected_tau_nu = np.zeros_like(nu)
                expected_spec = expected_tau_nu.copy()
                for line_index,line in zip(trans_indices,lines):
                    x1 = cloud.level_pop[line.low.number]
                    x2 = cloud.level_pop[line.up.number]
                    N1 = N*x1
                    N2 = N*x2
                    tau_nu_line = line.tau_nu(N1=N1,N2=N2,nu=nu)
                    source_func = helpers.B_nu(T=cloud.Tex[line_index],nu=nu)
                    spec_line = cloud.geometry.compute_flux_nu(
                                          tau_nu=tau_nu_line,source_function=source_func,
                                          solid_angle=solid_angle)
                    expected_tau_nu += tau_nu_line
                    expected_spec += spec_line
                assert np.allclose(tau_nu,expected_tau_nu,atol=1e-20,rtol=1e-3)
                assert np.allclose(spectrum,expected_spec,atol=0,rtol=1e-3)
            if ID == 'thin':
                assert np.allclose(tau_nu,0,atol=1e-3,rtol=0)

@pytest.mark.filterwarnings("ignore:lines of input molecule are overlapping")
@pytest.mark.filterwarnings("ignore:lines are overlapping")
def test_overlapping_lines():
    #first three transitions of HCl are separated by ~8 km/s and 6 km/s respectively
    def get_cloud(line_profile_type,width_v):
        return radiative_transfer.Cloud(
                  datafilepath=os.path.join(here,'LAMDA_files/hcl.dat'),
                  geometry='uniform sphere',line_profile_type=line_profile_type,
                  width_v=width_v,iteration_mode='ALI',
                  use_NG_acceleration=True,average_beta_over_line_profile=False)
    overlapping = [get_cloud(line_profile_type='rectangular',width_v=8.5*constants.kilo),
                   get_cloud(line_profile_type='rectangular',width_v=6.5*constants.kilo),
                   get_cloud(line_profile_type='Gaussian',width_v=4.5*constants.kilo),
                   get_cloud(line_profile_type='Gaussian',width_v=3.5*constants.kilo)]
    nonoverlapping = [get_cloud(line_profile_type='rectangular',width_v=5.5*constants.kilo),
                      get_cloud(line_profile_type='Gaussian',width_v=2.5*constants.kilo)
                      ]
    for ol in overlapping:
        lines = ol.emitting_molecule.rad_transitions
        lines = sorted(lines,key=lambda l: l.nu0)[:3]
        assert ol.lines_are_overlapping(lines)
    for nol in nonoverlapping:
        lines = nol.emitting_molecule.rad_transitions
        lines = sorted(lines,key=lambda l: l.nu0)[:3]
        assert not nol.lines_are_overlapping(lines)

def test_print_results():
    for cloud in generate_new_clouds():
        cloud.set_parameters(ext_background=ext_background,
                              N=1e14/constants.centi**2,Tkin=33.33,
                              collider_densities={'ortho-H2':1e3/constants.centi**3})
        cloud.solve_radiative_transfer()
        cloud.print_results()