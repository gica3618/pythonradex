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
average_options = (True,False)

def allowed_param_combination(geometry,line_profile_type):
    if geometry in ('LVG sphere','LVG slab') and line_profile_type=='Gaussian':
        return False
    else:
        return True

def cloud_params_iterator():
    return itertools.product(geometries,line_profile_types,iteration_modes,
                             use_ng_options,average_options)

def generate_new_clouds():
    clouds = []
    for geo,lp,mode,use_ng,average in cloud_params_iterator():
        if not allowed_param_combination(geometry=geo,line_profile_type=lp):
            continue
        cld = radiative_transfer.Cloud(
                              datafilepath=datafilepath,geometry=geo,
                              line_profile_type=lp,width_v=width_v,iteration_mode=mode,
                              use_NG_acceleration=use_ng,
                              average_over_line_profile=average)
        clouds.append(cld)
    return clouds

def test_warning_overlapping_lines():
    with pytest.warns(UserWarning):
        radiative_transfer.Cloud(
                              datafilepath=os.path.join(here,'LAMDA_files/hcl.dat'),
                              geometry='uniform sphere',
                              line_profile_type='rectangular',width_v=10*constants.kilo,
                              iteration_mode='ALI',use_NG_acceleration=True,
                              treat_overlapping_lines=False)

def test_overlapping_needs_averaging():
    with pytest.raises(ValueError):
        radiative_transfer.Cloud(
                              datafilepath=os.path.join(here,'LAMDA_files/hcl.dat'),
                              geometry='uniform sphere',
                              line_profile_type='rectangular',width_v=10*constants.kilo,
                              iteration_mode='ALI',use_NG_acceleration=True,
                              average_over_line_profile=False,treat_overlapping_lines=True)

def test_too_large_width_v():
    width_vs = np.array((1e4,1.2e4,1e5))*constants.kilo
    for width_v in width_vs:
        for geo,lp,mode,use_ng,average in cloud_params_iterator():
            with pytest.raises(AssertionError):
                radiative_transfer.Cloud(
                                  datafilepath=datafilepath,geometry=geo,
                                  line_profile_type=lp,width_v=width_v,iteration_mode=mode,
                                  use_NG_acceleration=use_ng,
                                  average_over_line_profile=average)

def test_LVG_rectangular():
    for geo in ('LVG sphere','LVG slab'):
        with pytest.raises(ValueError):
            radiative_transfer.Cloud(
                              datafilepath=datafilepath,geometry=geo,
                              line_profile_type='Gaussian',width_v=width_v,
                              iteration_mode='ALI',use_NG_acceleration=True,
                              average_over_line_profile=True)

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

class Test_ext_background_checks():

    @staticmethod
    def slow_background(nu):
        return helpers.generate_CMB_background(z=1)(nu)

    @staticmethod
    def fast_background(nu):
        return np.sin(nu) 

    def test_ext_background_variation_check(self):
        clouds = generate_new_clouds()
        for cloud in clouds:
            cloud.ext_background = self.fast_background
            assert not cloud.ext_background_is_slowly_varying()
            cloud.ext_background = self.slow_background
            assert cloud.ext_background_is_slowly_varying()
    
    def test_ext_background_check_at_initialisation(self):
        cloud = radiative_transfer.Cloud(
                              datafilepath=datafilepath,geometry='uniform sphere',
                              line_profile_type='Gaussian',width_v=1*constants.kilo,
                              iteration_mode='ALI',
                              average_over_line_profile=False)
        params = {'N':N,'Tkin':25,'collider_densities':{'para-H2':1}}
        with pytest.raises(ValueError):
            cloud.set_parameters(ext_background=self.fast_background,**params)
        cloud.set_parameters(ext_background=self.slow_background,**params)


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
                    'use_NG_acceleration':True,'average_over_line_profile':False}
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


class Test_rate_equation_quantities():

    T = 45

    def cloud_iterator(self):
        #note that it doesn't matter what I put for average_over_line_profile
        #here, because I use the averaged/non-averaged functions of the cloud
        #directly in the tests
        for geo,lp in itertools.product(geometries,line_profile_types):
            if not allowed_param_combination(geometry=geo,line_profile_type=lp):
                continue
            cloud = radiative_transfer.Cloud(
                                datafilepath=datafilepath,geometry=geo,
                                line_profile_type=lp,width_v=width_v,iteration_mode='ALI',
                                use_NG_acceleration=True,average_over_line_profile=False)
            cloud.set_parameters(ext_background=ext_background,N=N,Tkin=self.T,
                                 collider_densities={'para-H2':1e4/constants.centi**3})
            level_pop = cloud.emitting_molecule.LTE_level_pop(T=self.T)
            yield cloud,level_pop

    def line_iterator(self,cloud,level_pop):
        for line in cloud.emitting_molecule.rad_transitions:
            n_low = line.low.number
            n_up = line.up.number
            x1 = level_pop[n_low]
            x2 = level_pop[n_up]
            N1 = N*x1
            N2 = N*x2
            nu0 = line.nu0
            width_nu = line.line_profile.width_nu
            if cloud.line_profile_type == 'rectangular':
                nu = np.linspace(nu0-width_nu/2,nu0+width_nu/2,300)
            else:
                nu = np.linspace(nu0-width_nu*2,nu0+width_nu*2,400)
            tau_nu = line.tau_nu(N1=N1,N2=N2,nu=nu)
            beta_nu = cloud.geometry.beta(tau_nu)
            Iext_nu = cloud.ext_background(nu)
            phi_nu = line.line_profile.phi_nu(nu)
            S = line.A21*x2/(x1*line.B12-x2*line.B21)
            yield {'N1':N1,'N2':N2,'nu':nu,'tau_nu':tau_nu,'beta_nu':beta_nu,
                   'Iext_nu':Iext_nu,'phi_nu':phi_nu,'S':S,'line':line}

    def test_beta_wo_avg(self):
        for cloud,level_pop in self.cloud_iterator():
            expected_beta = []
            calculated_beta = cloud.beta_alllines_without_average(level_populations=level_pop)
            for line_data in self.line_iterator(cloud=cloud,level_pop=level_pop):
                N1 = line_data['N1']
                N2 = line_data['N2']
                tau_nu0 = line_data['line'].tau_nu0(N1=N1,N2=N2)
                beta = cloud.geometry.beta(np.array((tau_nu0,)))
                expected_beta.append(beta[0])
            assert np.allclose(expected_beta,calculated_beta,atol=0,rtol=1e-4)

    def test_beta_with_avg(self):
        for cloud,level_pop in self.cloud_iterator():
            expected_beta = []
            calculated_beta = cloud.beta_allines_averaged(level_populations=level_pop)
            for line_data in self.line_iterator(cloud=cloud,level_pop=level_pop):
                beta_nu = line_data['beta_nu']
                phi_nu = line_data['phi_nu']
                nu = line_data['nu']
                avg_beta = np.trapz(beta_nu*phi_nu,nu)/np.trapz(phi_nu,nu)
                expected_beta.append(avg_beta)
            assert np.allclose(expected_beta,calculated_beta,atol=0,rtol=1e-3)

    def test_source_func_wo_avg(self):
        for cloud,level_pop in self.cloud_iterator():
            expected_S = []
            calculated_S = cloud.source_function_alllines_without_average(
                                 level_populations=level_pop)
            for line_data in self.line_iterator(cloud=cloud,level_pop=level_pop):
                expected_S.append(line_data['S'])
            assert np.allclose(expected_S,calculated_S,atol=0,rtol=1e-3)
                
    def test_Jbar_wo_avg(self):
        for cloud,level_pop in self.cloud_iterator():
            calculated_beta = cloud.beta_alllines_without_average(
                                                  level_populations=level_pop)
            calculated_Jbar = cloud.Jbar_alllines_without_average( 
                                               level_populations=level_pop)
            calculated_S = cloud.source_function_alllines_without_average(
                                   level_populations=level_pop)
            expected_Jbar = calculated_beta*cloud.I_ext_lines\
                           + (1-calculated_beta)*calculated_S
            assert np.allclose(expected_Jbar,calculated_Jbar,atol=0,rtol=1e-3)
            
    def test_Jbar_with_avg(self):
        for cloud,level_pop in self.cloud_iterator():
            calculated_Jbar = cloud.Jbar_alllines_averaged(level_populations=level_pop)
            expected_Jbar = []
            for line_data in self.line_iterator(cloud=cloud,level_pop=level_pop):
                beta_nu = line_data['beta_nu']
                Iext_nu = line_data['Iext_nu']
                S = line_data['S']
                phi_nu = line_data['phi_nu']
                nu = line_data['nu']
                Jbar = np.trapz((beta_nu*Iext_nu+(1-beta_nu)*S)*phi_nu,nu)
                Jbar /= np.trapz(phi_nu,nu)
                expected_Jbar.append(Jbar)
            assert np.allclose(expected_Jbar,calculated_Jbar,atol=0,rtol=1e-2)

    def test_betaIext_averaged(self):
        for cloud,level_pop in self.cloud_iterator():
            calculated_avg = cloud.betaIext_alllines_averaged(level_populations=level_pop)
            expected_avg = []
            for line_data in self.line_iterator(cloud=cloud,level_pop=level_pop):
                beta_nu = line_data['beta_nu']
                Iext_nu = line_data['Iext_nu']
                phi_nu = line_data['phi_nu']
                nu = line_data['nu']
                beta_Iext_avg = np.trapz(beta_nu*Iext_nu*phi_nu,nu)
                beta_Iext_avg /= np.trapz(phi_nu,nu)
                expected_avg.append(beta_Iext_avg)
            assert np.allclose(expected_avg,calculated_avg,atol=0,rtol=1e-3)


@pytest.mark.filterwarnings("ignore:negative optical depth")
def test_ng_acceleration():
    cloud_kwargs = {'datafilepath':datafilepath,'geometry':'uniform sphere',
                    'line_profile_type':'rectangular','width_v':width_v,'iteration_mode':'ALI',
                    'average_over_line_profile':False}
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
    for geo,lp,use_ng,average in itertools.product(geometries,line_profile_types,
                                                        use_ng_options,average_options):
        if not allowed_param_combination(geometry=geo,line_profile_type=lp):
            continue
        cloud = radiative_transfer.Cloud(
                      datafilepath=datafilepath,geometry=geo,
                      line_profile_type=lp,width_v=width_v,iteration_mode='ALI',
                      use_NG_acceleration=use_ng,average_over_line_profile=average)
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


class TestFastFlux():

    test_widths_v = np.array((0.1,1,10,100,1000,9000))*constants.kilo
    solid_angle = 1
    Tex0 = 50
    test_transitions = nb.typed.List([2,4,5])

    @pytest.mark.filterwarnings("ignore:lines of input molecule are overlapping")
    def test_fast_flux(self):
        tau_nu0_values = np.logspace(-3,3,10)
        for line_profile_type in ('rectangular','Gaussian'):
            for geometry,width_v in itertools.product(radiative_transfer.Cloud.geometries.keys(),
                                                      self.test_widths_v):
                V_LVG_sphere = width_v/2
                if not allowed_param_combination(geometry=geometry,
                                                 line_profile_type=line_profile_type):
                    continue
                cloud = radiative_transfer.Cloud(
                          datafilepath=datafilepath,geometry=geometry,
                          line_profile_type=line_profile_type,width_v=width_v,iteration_mode='ALI',
                          use_NG_acceleration=True,average_over_line_profile=False)
                all_transitions = nb.typed.List(range(cloud.emitting_molecule.n_rad_transitions))
                nu0_lines = np.array([line.nu0 for line in cloud.emitting_molecule.rad_transitions])
                Tex = self.Tex0*np.ones(cloud.emitting_molecule.n_rad_transitions)
                for tau_nu0 in tau_nu0_values:
                    tau_nu0_lines = np.ones_like(Tex)*tau_nu0
                    general_kwargs = {'solid_angle':self.solid_angle,
                                      'tau_nu0_lines':tau_nu0_lines,
                                      'Tex':Tex,'nu0_lines':nu0_lines}
                    non_LVG_sphere_kwargs = {'compute_flux_nu':cloud.geometry.compute_flux_nu,
                                             'width_v':width_v}
                    gauss_kwargs = {'tau_peak_fraction':radiative_transfer.Cloud.tau_peak_fraction,
                                    'nu_per_FHWM':radiative_transfer.Cloud.nu_per_FHWM}
                    if geometry == 'LVG sphere':
                        all_fluxes = cloud.fast_fluxes_LVG_sphere(
                                         **general_kwargs,transitions=all_transitions,
                                         V_LVG_sphere=V_LVG_sphere)
                        selected_fluxes = cloud.fast_fluxes_LVG_sphere(
                                            **general_kwargs,transitions=self.test_transitions,
                                            V_LVG_sphere=V_LVG_sphere)
                    else:
                        if line_profile_type == 'rectangular':
                            all_fluxes = cloud.fast_fluxes_rectangular(
                                                **general_kwargs,**non_LVG_sphere_kwargs,
                                                transitions=all_transitions)
                            selected_fluxes = cloud.fast_fluxes_rectangular(
                                                **general_kwargs,**non_LVG_sphere_kwargs,
                                                transitions=self.test_transitions)
                        elif line_profile_type == 'Gaussian':
                            all_fluxes = cloud.fast_fluxes_Gaussian(
                                            **general_kwargs,**non_LVG_sphere_kwargs,
                                            **gauss_kwargs,transitions=all_transitions)
                            selected_fluxes = cloud.fast_fluxes_Gaussian(
                                                **general_kwargs,**non_LVG_sphere_kwargs,
                                                **gauss_kwargs,
                                                transitions=self.test_transitions)
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
                        if geometry == 'LVG sphere':
                            LVG_kwargs = {'nu':dense_nu,'nu0':nu0,'V':V_LVG_sphere}
                        else:
                            LVG_kwargs = {}
                        expected_spec = cloud.geometry.compute_flux_nu(
                                          tau_nu=expected_tau_nu,source_function=source_function,
                                          solid_angle=self.solid_angle,**LVG_kwargs)
                        expected_flux = np.trapz(expected_spec,dense_nu)
                        expected_fluxes.append(expected_flux)
                    atol = 0
                    rtol = 1e-2
                    assert np.allclose(all_fluxes, expected_fluxes,atol=atol,rtol=rtol)
                    expected_selected_fluxes = [expected_fluxes[i] for i in
                                                self.test_transitions]
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
    for lp,avg in itertools.product(line_profile_types,average_options):
        for geometry in radiative_transfer.Cloud.geometries.keys():
            if not allowed_param_combination(geometry=geometry,line_profile_type=lp):
                continue
            cloud = radiative_transfer.Cloud(
                      datafilepath=datafilepath,geometry=geometry,
                      line_profile_type=lp,width_v=width_v,iteration_mode='ALI',
                      use_NG_acceleration=True,average_over_line_profile=avg)
            cloud.set_parameters(ext_background=helpers.zero_background,N=N,
                                  Tkin=Tkin,collider_densities=collider_densities)
            cloud.solve_radiative_transfer()
            fluxes = cloud.fluxes(solid_angle=Omega,transitions=None)
            expected_fluxes = []
            for i,line in enumerate(cloud.emitting_molecule.rad_transitions):
                n_up = line.up.number
                up_level_pop = cloud.level_pop[n_up]
                if geometry in ('uniform sphere','LVG sphere'):
                    #in the case of spheres, we can do an elegant test
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
    for lp,avg in itertools.product(line_profile_types,average_options):
        for geometry in radiative_transfer.Cloud.geometries.keys():
            if not allowed_param_combination(geometry=geometry,line_profile_type=lp):
                continue
            cloud = radiative_transfer.Cloud(
                      datafilepath=datafilepath,geometry=geometry,
                      line_profile_type=lp,width_v=width_v,iteration_mode='ALI',
                      use_NG_acceleration=True,average_over_line_profile=avg)
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
                if lp == 'rectangular' and geometry != 'LVG sphere':
                    expected_total_flux = bb_flux_nu0*line.line_profile.width_nu
                    assert np.isclose(a=expected_total_flux,b=fluxes[i],
                                      atol=0,rtol=3e-2)
                else:
                    assert lp == 'Gaussian' or geometry == 'LVG sphere'  

def test_tau_and_spectrum_single_lines():
    Tkin = 57
    solid_angle = 1
    collider_densities = {'ortho-H2':1e10/constants.centi**3}
    N_values = {'thin':1e12/constants.centi**2,
                'thick':1e19/constants.centi**2}
    trans_indices = (1,6)
    for lp,avg in itertools.product(line_profile_types,average_options):
        for geometry in radiative_transfer.Cloud.geometries.keys():
            if not allowed_param_combination(geometry=geometry,line_profile_type=lp):
                continue
            cloud = radiative_transfer.Cloud(
                      datafilepath=datafilepath,geometry=geometry,
                      line_profile_type=lp,width_v=width_v,iteration_mode='ALI',
                      use_NG_acceleration=True,average_over_line_profile=avg)
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
                    if geometry == 'LVG sphere':
                        LVG_kwargs = {'nu':nu_line,'nu0':line.nu0,'V':width_v/2}
                    else:
                        LVG_kwargs = {}
                    expected_spec = cloud.geometry.compute_flux_nu(
                                        tau_nu=expected_tau_nu,source_function=source_func,
                                        solid_angle=solid_angle,**LVG_kwargs)
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
def test_tau_and_spectrum_overlapping_lines_without_overlap_treatment():
    datafilepath = os.path.join(here,'LAMDA_files/hcl.dat')
    solid_angle = 1
    Tkin = 102
    collider_densities = {'ortho-H2':1e10/constants.centi**3}
    N_values = {'thin':1e10/constants.centi**2,
                'thick':1e15/constants.centi**2}
    trans_indices = (0,1,2)
    width_v = 20*constants.kilo
    for lp,avg in itertools.product(line_profile_types,average_options):
        for geometry in radiative_transfer.Cloud.geometries.keys():
            if not allowed_param_combination(geometry=geometry,line_profile_type=lp):
                continue
            cloud = radiative_transfer.Cloud(
                      datafilepath=datafilepath,geometry=geometry,
                      line_profile_type=lp,width_v=width_v,iteration_mode='ALI',
                      use_NG_acceleration=True,average_over_line_profile=avg,
                      treat_overlapping_lines=False)
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
                    if geometry == 'LVG sphere':
                        LVG_kwargs = {'nu':nu,'nu0':line.nu0,'V':width_v/2}
                    else:
                        LVG_kwargs = {}
                    spec_line = cloud.geometry.compute_flux_nu(
                                          tau_nu=tau_nu_line,source_function=source_func,
                                          solid_angle=solid_angle,**LVG_kwargs)
                    expected_tau_nu += tau_nu_line
                    expected_spec += spec_line
                assert np.allclose(tau_nu,expected_tau_nu,atol=1e-20,rtol=1e-3)
                assert np.allclose(spectrum,expected_spec,atol=0,rtol=1e-3)
            if ID == 'thin':
                assert np.allclose(tau_nu,0,atol=1e-3,rtol=0)


class TestOverlappingLines():

    @staticmethod
    def get_cloud(line_profile_type,width_v,datafilename):
        return radiative_transfer.Cloud(
                  datafilepath=os.path.join(here,'LAMDA_files',datafilename),
                  geometry='uniform sphere',line_profile_type=line_profile_type,
                  width_v=width_v,iteration_mode='ALI',
                  use_NG_acceleration=True,average_over_line_profile=False,
                  verbose=True)

    def get_HCl_cloud(self,line_profile_type,width_v):
        return self.get_cloud(line_profile_type=line_profile_type,width_v=width_v,
                              datafilename='hcl.dat')

    def get_CO_cloud(self,line_profile_type,width_v):
        return self.get_cloud(line_profile_type=line_profile_type,width_v=width_v,
                              datafilename='co.dat')

    @pytest.mark.filterwarnings("ignore:lines of input molecule are overlapping")
    def test_overlapping_lines(self):
        #first three transitions of HCl are separated by ~8 km/s and 6 km/s respectively
        overlapping_3lines = [self.get_HCl_cloud(line_profile_type='rectangular',
                                                 width_v=16*constants.kilo),
                              self.get_HCl_cloud(line_profile_type='Gaussian',
                                                 width_v=10*constants.kilo)
                              ]
        for ol in overlapping_3lines:
            assert ol.overlapping_lines[0] == [1,2]
            assert ol.overlapping_lines[1] == [0,2]
            assert ol.overlapping_lines[2] == [0,1]
        overlapping_2lines = [self.get_HCl_cloud(line_profile_type='rectangular',
                                                 width_v=8.5*constants.kilo),
                              self.get_HCl_cloud(line_profile_type='Gaussian',
                                                 width_v=3.5*constants.kilo)
                              ]
        for ol in overlapping_2lines:
            assert ol.overlapping_lines[0] == [1,]
            assert ol.overlapping_lines[1] == [0,2]
            assert ol.overlapping_lines[2] == [1,]
        #transitions 4-11 are separated by ~11.2 km/s
        overlapping_8lines = [self.get_HCl_cloud(line_profile_type='rectangular',
                                                 width_v=11.5*constants.kilo),
                              self.get_HCl_cloud(line_profile_type='Gaussian',
                                                 width_v=4*constants.kilo)
                              ]
        for ol in overlapping_8lines:
            for i in range(3,11):
                assert ol.overlapping_lines[i] == [index for index in range(3,11)
                                                   if index!=i]
        for line_profile_type in line_profile_types:
            CO_cloud = self.get_CO_cloud(line_profile_type=line_profile_type,
                                         width_v=1*constants.kilo)
            for overlap_lines in CO_cloud.overlapping_lines:
                assert overlap_lines == []
            HCl_cloud = self.get_HCl_cloud(line_profile_type=line_profile_type,
                                           width_v=0.01*constants.kilo)
            assert HCl_cloud.overlapping_lines[0] == []
            assert HCl_cloud.overlapping_lines[11] == []

    @pytest.mark.filterwarnings("ignore:lines of input molecule are overlapping")
    def test_any_overlapping(self):
        #raise NotImplementedError('add some more tests here')
        for line_profile_type in line_profile_types:
            HCl_cloud = self.get_HCl_cloud(line_profile_type=line_profile_type,
                                           width_v=10*constants.kilo)
            assert HCl_cloud.any_line_has_overlap(line_indices=[0,1,2,3,4])
            assert HCl_cloud.any_line_has_overlap(line_indices=[0,])
            assert HCl_cloud.any_line_has_overlap(
                          line_indices=list(range(len(HCl_cloud.emitting_molecule.rad_transitions))))
            HCl_cloud = self.get_HCl_cloud(line_profile_type=line_profile_type,
                                           width_v=1*constants.kilo)
            assert not HCl_cloud.any_line_has_overlap(line_indices=[0,1,2])
            CO_cloud = self.get_CO_cloud(line_profile_type=line_profile_type,
                                         width_v=1*constants.kilo)
            assert not CO_cloud.any_line_has_overlap(
                       line_indices=list(range(len(CO_cloud.emitting_molecule.rad_transitions))))


class TestModelGrid():
    
    cloud = radiative_transfer.Cloud(
                          datafilepath=datafilepath,geometry='uniform sphere',
                          line_profile_type='rectangular',width_v=1*constants.kilo,
                          iteration_mode='ALI',use_NG_acceleration=True,
                          average_over_line_profile=False)
    ext_backgrounds = {'CMB':helpers.generate_CMB_background(z=2),
                       'zero':helpers.zero_background}
    N_values = np.array((1e14,1e16))/constants.centi**2
    Tkin_values = [20,200,231]
    collider_densities_values = {'ortho-H2':np.array((1e3,1e4))/constants.centi**3,
                                 'para-H2':np.array((1e4,1e7))/constants.centi**3}
    grid_kwargs = {'ext_backgrounds':ext_backgrounds,'N_values':N_values,
                   'Tkin_values':Tkin_values,
                   'collider_densities_values':collider_densities_values}
    requested_output = ['level_pop','Tex','tau_nu0','fluxes','tau_nu','spectrum']
    transitions = [3,4]
    solid_angle = 1
    nu = np.linspace(115.27,115.28,10)*constants.giga
    
    def test_wrong_requested_output(self):
        wrong_requested_output = ['Tex','levelpop']
        with pytest.raises(AssertionError):
            for model in self.cloud.model_grid(**self.grid_kwargs,
                                               requested_output=wrong_requested_output):
                pass

    def test_insufficient_input(self):
        #test that errors are thrown if solid_angle or nu are not provided
        for request in ('fluxes','spectrum'):
            with pytest.raises(AssertionError): 
                for model in self.cloud.model_grid(
                                **self.grid_kwargs,requested_output=[request,],
                                nu=self.nu):
                    pass
        for request in ('tau_nu','spectrum'):
            with pytest.raises(AssertionError): 
                for model in self.cloud.model_grid(
                                **self.grid_kwargs,requested_output=[request,],
                                solid_angle=self.solid_angle):
                    pass

    @pytest.mark.filterwarnings("ignore:negative optical depth")
    def test_grid(self):
        iterator = self.cloud.model_grid(
                        **self.grid_kwargs,requested_output=self.requested_output,
                        solid_angle=self.solid_angle,transitions=self.transitions,
                        nu=self.nu)
        models = [m for m in iterator]
        n_check_models = 0
        for backgroundID,ext_background in self.ext_backgrounds.items():
            for N in self.N_values:
                for Tkin in self.Tkin_values:
                    for n_ortho,n_para in itertools.product(self.collider_densities_values['ortho-H2'],
                                                            self.collider_densities_values['para-H2']):
                        collider_densities = {'ortho-H2':n_ortho,'para-H2':n_para}
                        self.cloud.set_parameters(
                              ext_background=ext_background,N=N,Tkin=Tkin,
                              collider_densities=collider_densities)
                        self.cloud.solve_radiative_transfer()
                        fluxes = self.cloud.fluxes(solid_angle=self.solid_angle,
                                                   transitions=self.transitions)
                        tau_nu = self.cloud.tau_nu(nu=self.nu)
                        spectrum = self.cloud.spectrum(
                                            solid_angle=self.solid_angle,nu=self.nu)
                        #'level_pop','Tex','tau_nu0','fluxes','tau_nu', and 'spectrum'
                        matching_models = [m for m in models if
                                           m['ext_background']==backgroundID
                                           and m['N']==N and m['Tkin']==Tkin
                                           and m['collider_densities']==collider_densities]
                        assert len(matching_models) == 1
                        matching_model = matching_models[0]
                        assert np.all(matching_model['level_pop'] == self.cloud.level_pop)
                        assert np.all(matching_model['Tex']
                                      == self.cloud.Tex[self.transitions])
                        assert np.all(matching_model['tau_nu0']
                                      == self.cloud.tau_nu0[self.transitions])
                        assert np.all(matching_model['fluxes'] == fluxes)
                        assert np.all(matching_model['tau_nu'] == tau_nu)
                        assert np.all(matching_model['spectrum']==spectrum)
                        n_check_models += 1
        assert n_check_models == len(models)


def test_print_results():
    for cloud in generate_new_clouds():
        cloud.set_parameters(ext_background=ext_background,
                              N=1e14/constants.centi**2,Tkin=33.33,
                              collider_densities={'ortho-H2':1e3/constants.centi**3})
        cloud.solve_radiative_transfer()
        cloud.print_results()