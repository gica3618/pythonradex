# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:30:44 2017

@author: gianni
"""
import os
from pythonradex import radiative_transfer,helpers,escape_probability
from scipy import constants
import numpy as np
import itertools
import pytest
import numbers
import copy


here = os.path.dirname(os.path.abspath(__file__))
datafolder = os.path.join(here,'LAMDA_files')
datafilepath = {'CO':os.path.join(datafolder,'co.dat'),
                'HCl':os.path.join(datafolder,'hcl.dat'),
                'C+':os.path.join(datafolder,'c+.dat')}
cmb = helpers.generate_CMB_background()

line_profile_types = ('rectangular','Gaussian')
geometries = tuple(radiative_transfer.Source.geometries.keys())
use_ng_options = (True,False)
treat_overlap_options = (False,True)

def get_general_test_source(specie,width_v):
    return radiative_transfer.Source(
                          datafilepath=datafilepath[specie],geometry='static sphere',
                          line_profile_type='Gaussian',width_v=width_v)

def allowed_geo_param_combination(geometry,line_profile_type,treat_line_overlap):
    if 'LVG' in geometry and line_profile_type=='Gaussian':
        return False
    if 'LVG' in geometry and treat_line_overlap:
        return False
    return True

def iterate_all_source_params():
    return itertools.product(geometries,line_profile_types,use_ng_options,
                             treat_overlap_options)

def iterate_allowed_source_params():
    for geo,lp,use_ng,treat_overlap in iterate_all_source_params():
        if allowed_geo_param_combination(geometry=geo,line_profile_type=lp,
                                         treat_line_overlap=treat_overlap):
            yield geo,lp,use_ng,treat_overlap

def general_source_iterator(specie,width_v):
    for geo,lp,use_ng,treat_overlap in iterate_allowed_source_params():
        src = radiative_transfer.Source(
                              datafilepath=datafilepath[specie],geometry=geo,
                              line_profile_type=lp,width_v=width_v,
                              use_Ng_acceleration=use_ng,
                              treat_line_overlap=treat_overlap)
        yield src


class TestInitialisation():

    def test_too_large_width_v(self):
        width_vs = np.array((1e4,1.2e4,1e5))*constants.kilo
        for width_v in width_vs:
            for geo,lp,use_ng,treat_overlap in iterate_allowed_source_params():
                with pytest.raises(AssertionError):
                    radiative_transfer.Source(
                                      datafilepath=datafilepath['CO'],geometry=geo,
                                      line_profile_type=lp,width_v=width_v,
                                      use_Ng_acceleration=use_ng,
                                      treat_line_overlap=treat_overlap)

    def test_LVG_cannot_treat_overlap(self):
        for geo,lp,use_ng,average in iterate_allowed_source_params():
            if 'LVG' in geo:
                with pytest.raises(ValueError):
                    radiative_transfer.Source(
                                          datafilepath=datafilepath['HCl'],
                                          geometry=geo,line_profile_type=lp,
                                          width_v=10*constants.kilo,
                                          use_Ng_acceleration=use_ng,
                                          treat_line_overlap=True)    

    def test_warning_overlapping_lines(self):
        for geo,lp,use_ng,average in iterate_allowed_source_params():
            with pytest.warns(UserWarning):
                radiative_transfer.Source(
                                      datafilepath=datafilepath['HCl'],
                                      geometry=geo,line_profile_type=lp,
                                      width_v=10*constants.kilo,
                                      use_Ng_acceleration=use_ng,
                                      treat_line_overlap=False)
    
    def test_LVG_needs_rectangular(self):
        for geo in ('LVG sphere','LVG slab'):
            with pytest.raises(ValueError):
                radiative_transfer.Source(
                                  datafilepath=datafilepath['CO'],geometry=geo,
                                  line_profile_type='Gaussian',width_v=1*constants.kilo,
                                  use_Ng_acceleration=True)


class TestUpdateParameters():

    standard_params = {'ext_background':cmb,
                       'Tkin':20,'collider_densities':{'para-H2':1},
                       'N':1e15/constants.centi**2,'T_dust':0,'tau_dust':0}    
    non_func_params = ['Tkin','collider_densities','N']
    func_params = ['ext_background','T_dust','tau_dust']
    all_params = non_func_params+func_params

    def copy_cloud_parameters(self,source):
        params = {}
        for param in self.all_params:
            params[param] = copy.deepcopy(getattr(source.rate_equations,param))
        return params

    def test_copy_cloud_parameters(self):
        def T_dust_old(nu):
            return 2*nu
        def T_dust_new(nu):
            return 3*nu
        source = radiative_transfer.Source(
                          datafilepath=datafilepath['CO'],geometry='static sphere',
                          line_profile_type='Gaussian',width_v=1*constants.kilo,
                          use_Ng_acceleration=True)
        source.update_parameters(N=1e14,Tkin=20,collider_densities={'para-H2':1},
                                ext_background=0,T_dust=T_dust_old,tau_dust=1)
        old_params = self.copy_cloud_parameters(source)
        source.update_parameters(T_dust=T_dust_new)
        test_nu = 100*constants.giga
        assert old_params['T_dust'](test_nu) == 2*test_nu
        assert source.rate_equations.T_dust(test_nu) == 3*test_nu

    def verify_cloud_params(self,source,new_params,original_params):
        #original_params=None is used for the very first setting of the params when
        #previous params do not yet exist
        #original_params are only used if any of the new params is None
        nu0s = source.emitting_molecule.nu0
        for pname in self.all_params:
            if original_params is None:
                assert new_params[pname] is not None
            else:
                assert pname in original_params
            if pname not in new_params:
                new_params[pname] = None
        for pname,p in new_params.items():
            if p is None:
                assert original_params is not None
                expected_p = original_params[pname]
            else:
                expected_p = p
            if pname in self.non_func_params:
                assert getattr(source.rate_equations,pname) == expected_p
            elif pname in self.func_params:
                if isinstance(expected_p,numbers.Number):
                    expected = np.ones_like(nu0s)*expected_p
                else:
                    expected =  expected_p(nu0s)
                values_nu0 = getattr(source.rate_equations,pname)(nu0s)
                assert np.all(expected==values_nu0)
            else:
                raise ValueError

    def test_should_be_updated(self):
        def func1(x):
            return x**2
        def func2(x):
            return x*9
        func3 = lambda x: x-4
        for old in (0,1.2,func1,func3):
            #if I give None, it should not be updated
            assert not radiative_transfer.Source.should_be_updated(
                                               new_func_or_number=None,old=old)
        #case 1: new value is a number
        assert radiative_transfer.Source.should_be_updated(new_func_or_number=1.1,old=func1)
        assert radiative_transfer.Source.should_be_updated(new_func_or_number=1,old=2)
        for num in (1,1.23):
            assert not radiative_transfer.Source.should_be_updated(
                                               new_func_or_number=num,old=num)
        #case 2: new value is a func
        for func in (func1,func2,func3):
            #whenever I give a func, I want to update (even when giving the same func
            #as the old one)
            assert radiative_transfer.Source.should_be_updated(
                                                  new_func_or_number=func,old=func1)
            assert radiative_transfer.Source.should_be_updated(
                                            new_func_or_number=func,old=2.1)

    def test_initial_param_setting(self):
        for source in general_source_iterator(specie='CO',width_v=1*constants.kilo):
            source.update_parameters(**self.standard_params)
            self.verify_cloud_params(source=source,new_params=self.standard_params,
                                     original_params=None)

    def test_None_not_allowed_initial_setting(self):
        for p_name in self.standard_params.keys():
            invalid_initial_params = self.standard_params.copy()
            invalid_initial_params[p_name] = None
            source = get_general_test_source(specie='CO',width_v=1*constants.kilo)
            with pytest.raises(AssertionError):
                source.update_parameters(**invalid_initial_params)

    def test_update_N(self):
        for source in general_source_iterator(specie='CO',width_v=1*constants.kilo):
            source.update_parameters(**self.standard_params)
            changed_values = [None,self.standard_params['N'],
                              self.standard_params['N']/2.12]
            for value in changed_values:
                changed_params = self.standard_params.copy()
                changed_params['N'] = value
                original_params = self.copy_cloud_parameters(source)
                source.update_parameters(**changed_params)
                self.verify_cloud_params(source=source,new_params=changed_params,
                                         original_params=original_params)

    def test_update_Tkin_colldens(self):
        for source in general_source_iterator(specie='CO',width_v=1*constants.kilo):
            source.update_parameters(**self.standard_params)
            changed_Tkin = [None,self.standard_params['Tkin'],12.3]
            changed_coll_dens = [None,self.standard_params['collider_densities'],
                                 {'para-H2':23},{'para-H2':31,'ortho-H2':34},
                                 {'ortho-H2':2389}]
            for Tkin,coll_dens in itertools.product(changed_Tkin,changed_coll_dens):
                changed_params = self.standard_params.copy()
                changed_params['Tkin'] = Tkin
                changed_params['collider_densities'] = coll_dens
                original_params = self.copy_cloud_parameters(source)
                source.update_parameters(**changed_params)
                self.verify_cloud_params(source=source,new_params=changed_params,
                                         original_params=original_params)

    def test_update_ext_background(self):
        for source in general_source_iterator(specie='CO',width_v=1*constants.kilo):
            source.update_parameters(**self.standard_params)
            changed_ext_bgs = [None,self.standard_params['ext_background'],
                               lambda nu: nu*2.234,helpers.generate_CMB_background(z=2)]
            for ext_bg in changed_ext_bgs:
                changed_params = self.standard_params.copy()
                changed_params['ext_background'] = ext_bg
                original_params = self.copy_cloud_parameters(source)
                source.update_parameters(**changed_params)
                self.verify_cloud_params(source=source,new_params=changed_params,
                                         original_params=original_params)

    def test_update_dust(self):
        for source in general_source_iterator(specie='CO',width_v=1*constants.kilo):
            if 'LVG' in source.geometry_name:
                continue
            def some_func(nu):
                return nu/2
            source.update_parameters(**self.standard_params)
            changed_Tdust = [None,self.standard_params['T_dust'],some_func,
                             lambda nu: nu/(100*constants.giga)*123]
            changed_tau_dust = [None,self.standard_params['tau_dust'],some_func,
                                lambda nu: nu/(100*constants.giga)*0.258]
            for T_dust,tau_dust in itertools.product(changed_Tdust,changed_tau_dust):
                changed_params = self.standard_params.copy()
                changed_params['T_dust'] = T_dust
                changed_params['tau_dust'] = tau_dust
                original_params = self.copy_cloud_parameters(source)
                source.update_parameters(**changed_params)
                self.verify_cloud_params(source=source,new_params=changed_params,
                                         original_params=original_params)

    def test_update_several_parameters(self):
        for source in general_source_iterator(specie='CO',width_v=1*constants.kilo):
            source.update_parameters(**self.standard_params)
            self.verify_cloud_params(source=source,new_params=self.standard_params,
                                     original_params=None)
            new_params = {'ext_background':lambda nu: cmb(nu)/2,
                          'N':4*self.standard_params['N'],
                          'Tkin':4*self.standard_params['Tkin']}
            if 'LVG' in source.geometry_name:
                T_dust = None
                tau_dust = None
            else:
                T_dust = lambda nu: nu/(100*constants.giga)*174
                tau_dust = lambda nu: nu/(100*constants.giga)*0.89
            new_params['T_dust'] = T_dust
            new_params['tau_dust'] = tau_dust
            old_params = self.copy_cloud_parameters(source=source)
            source.update_parameters(**new_params)
            self.verify_cloud_params(source=source,new_params=new_params,
                                     original_params=old_params)

    def test_None_parameters(self):
        for source in general_source_iterator(specie='CO',width_v=1*constants.kilo):
            old_params = {'ext_background':cmb,
                          'Tkin':201,'collider_densities':{'para-H2':1},
                          'N':2e12,'T_dust':lambda nu: np.ones_like(nu)*145,
                          'tau_dust':lambda nu: np.ones_like(nu)*5}
            if 'LVG' in source.geometry_name:
                old_params['T_dust'] = old_params['tau_dust'] = 0
            source.update_parameters(**old_params)
            for param in old_params:
                new_params = old_params.copy()
                new_params[param] = None
                source.update_parameters(**new_params)
                self.verify_cloud_params(source=source,new_params=new_params,
                                         original_params=old_params)
            #test also if everything is None
            current_params = self.copy_cloud_parameters(source)
            new_params = {p:None for p in old_params}
            source.update_parameters(**new_params)
            self.verify_cloud_params(source=source,new_params=new_params,
                                     original_params=current_params)

    def test_invalid_collider_densities(self):
        test_source = get_general_test_source(specie='CO',width_v=1*constants.kilo)
        invalid_coll_densities = {'e':1/constants.centi**3} #CO has not electron data
        with pytest.raises(ValueError):
            test_source.update_parameters(ext_background=cmb,
                                         N=1e12/constants.centi*2,Tkin=20,
                                         collider_densities=invalid_coll_densities)

    def test_dust_not_allowed_for_LVG(self):
        tau_dust = lambda nu: np.ones_like(nu)*1
        T_dust = lambda nu: np.ones_like(nu)*100
        for geometry in radiative_transfer.Source.geometries.keys():
            if 'LVG' in geometry:
                source = radiative_transfer.Source(
                                      datafilepath=datafilepath['CO'],geometry=geometry,
                                      line_profile_type='rectangular',
                                      width_v=1*constants.kilo)
                with pytest.raises(ValueError):
                    source.update_parameters(ext_background=cmb,
                                            N=1e12/constants.centi**2,Tkin=20,
                                            collider_densities={'para-H2':1/constants.centi**3},
                                            tau_dust=tau_dust,T_dust=T_dust)

    @pytest.mark.filterwarnings("ignore:negative optical depth")
    def test_update_parameters_with_physics(self):
        N_values = [1e14/constants.centi**2,]
        ext_backgrounds = [cmb,lambda nu: 0,lambda nu: cmb(nu)/2]
        Tkins = [120,]
        coll_density_values = [1e5/constants.centi**3,]
        collider_cases = [['ortho-H2'],['para-H2'],['ortho-H2','para-H2']]
        T_dust_cases = [0,123,lambda nu: np.ones_like(nu)*250]
        tau_dust_cases = [0,1.1,lambda nu: np.ones_like(nu)*4]
        def param_iterator():
            return itertools.product(N_values,ext_backgrounds,Tkins,coll_density_values,
                                     collider_cases,T_dust_cases,tau_dust_cases)
        cloud_kwargs = {'datafilepath':datafilepath['CO'],'geometry':'static sphere',
                        'line_profile_type':'rectangular','width_v':1*constants.kilo,
                        'use_Ng_acceleration':True,'treat_line_overlap':False}
        cloud_to_modify = radiative_transfer.Source(**cloud_kwargs)
        def generate_cloud_and_set_params(params):
            source = radiative_transfer.Source(**cloud_kwargs)
            source.update_parameters(**params)
            return source
        for N,ext_b,Tkin,coll_dens,colliders,T_dust,tau_dust in param_iterator():
            params = {'N':N,'Tkin':Tkin,'ext_background':ext_b,'T_dust':T_dust,
                      'tau_dust':tau_dust}
            #put different values in case more than one collider (not really necessary...)
            coll_densities = {collider:coll_dens*(i+1) for i,collider in
                              enumerate(colliders)}
            params['collider_densities'] = coll_densities
            cloud_to_modify.update_parameters(**params)
            reference_cloud = generate_cloud_and_set_params(params=params)
            cloud_to_modify.solve_radiative_transfer()
            reference_cloud.solve_radiative_transfer()
            assert np.all(cloud_to_modify.level_pop==reference_cloud.level_pop)


@pytest.mark.filterwarnings("ignore:negative optical depth")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_ng_acceleration():
    N_values = 10**np.array((12,14,16))/constants.centi**2
    Tkin_values = np.array((40,400))
    coll_dens_values = np.array((1e3,1e6))/constants.centi**3
    ext_backgrounds = [0,cmb]
    for geo,lp,treat_overlap in itertools.product(geometries,line_profile_types,
                                                  [True,False]):
        if not allowed_geo_param_combination(geometry=geo,line_profile_type=lp,
                                             treat_line_overlap=treat_overlap):
            continue
        cloud_kwargs = {'datafilepath':datafilepath['CO'],'geometry':geo,
                        'line_profile_type':lp,'width_v':1*constants.kilo,
                        'treat_line_overlap':treat_overlap}
        for N,Tkin,coll_dens,ext_background in\
               itertools.product(N_values,Tkin_values,coll_dens_values,ext_backgrounds):
            params = {'N':N,'Tkin':Tkin,'ext_background':ext_background,
                      'collider_densities':{'para-H2':coll_dens},'T_dust':0,
                      'tau_dust':0}
            level_pops = []
            for ng in (True,False):
                source = radiative_transfer.Source(use_Ng_acceleration=ng,**cloud_kwargs)
                source.update_parameters(**params)
                source.solve_radiative_transfer()
                level_pops.append(source.level_pop)
            for lp1,lp2 in itertools.pairwise(level_pops):
                np.allclose(lp1,lp2,atol=0,rtol=1e-3)


def test_compute_residual():
    min_tau = radiative_transfer.Source.min_tau_considered_for_convergence
    small_tau = np.array((0.7*min_tau,min_tau/2,min_tau/100,min_tau/500,min_tau/1.01))
    tau = np.array((1,10,2*min_tau,min_tau,min_tau/2))
    n_relevant_taus = (tau>min_tau).sum()
    Tex_residual = np.array((1,2,3,4,5))
    assert radiative_transfer.Source.compute_residual(
                                          Tex_residual=Tex_residual,tau=small_tau,
                                          min_tau=min_tau) == 0
    expected_residual = np.sum(Tex_residual[tau>min_tau])/n_relevant_taus
    assert radiative_transfer.Source.compute_residual(
                                          Tex_residual=Tex_residual,tau=tau,
                                          min_tau=min_tau)\
            == expected_residual

def test_upper_lower_level_pop():
    source = get_general_test_source(specie="CO", width_v=1.3*constants.kilo)
    source.update_parameters(N=1e15*constants.centi**-2,Tkin=123,
                             collider_densities={"ortho-H2":1e6*constants.centi**-3},
                             ext_background=cmb,T_dust=0,tau_dust=0)
    source.solve_radiative_transfer()
    for t,trans in enumerate(source.emitting_molecule.rad_transitions):
        assert source.upper_level_population[t] == source.level_pop[trans.up.index]
        assert source.lower_level_population[t] == source.level_pop[trans.low.index]

def test_intensity_transformation():
    nu = np.linspace(100,101,10)*constants.giga
    T = 123.234
    specific_intensity_Planck = 2*constants.h*nu**3/constants.c**2\
                       *(np.exp(constants.h*nu/(constants.k*T))-1)**-1
    specific_intensity_RJ = 2*nu**2*constants.k*T/constants.c**2
    with pytest.raises(ValueError):
        radiative_transfer.Source.transform_specific_intensity(
                   specific_intensity=specific_intensity_Planck,
                   nu=nu,solid_angle=None,output_type="flux density")
    with pytest.raises(ValueError):
        radiative_transfer.Source.transform_specific_intensity(
                   specific_intensity=specific_intensity_Planck,nu=nu,
                   solid_angle=None,output_type="test123")
    T_RJ = radiative_transfer.Source.transform_specific_intensity(
               specific_intensity=specific_intensity_RJ,nu=nu,solid_angle=None,
               output_type="Rayleigh-Jeans")
    T_Planck= radiative_transfer.Source.transform_specific_intensity(
               specific_intensity=specific_intensity_Planck,nu=nu,solid_angle=None,
               output_type="Planck")
    for brightness_T in (T_RJ,T_Planck):
        assert np.allclose(brightness_T,T,atol=0,rtol=1e-8)

def test_solid_angle_warnings():
    source = get_general_test_source(specie="CO", width_v=1.3*constants.kilo)
    source.update_parameters(N=1e15*constants.centi**-2,Tkin=123,
                             collider_densities={"ortho-H2":1e6*constants.centi**-3},
                             ext_background=cmb,T_dust=0,tau_dust=0)
    source.solve_radiative_transfer()
    with pytest.warns(UserWarning):
        source.frequency_integrated_emission(
                        output_type="intensity",transitions=None,solid_angle=1.23)
    for output_type in ("specific intensity","Rayleigh-Jeans","Planck"):
        with pytest.warns(UserWarning):
            trans = source.emitting_molecule.rad_transitions[1]
            width_v = trans.line_profile.width_v
            v = np.linspace(-2*width_v,2*width_v,20)
            nu = trans.nu0*(1-v/constants.c)
            source.spectrum(nu=nu, output_type=output_type,solid_angle=0.23)
        with pytest.warns(UserWarning):
            source.emission_at_line_center(output_type=output_type,transitions=None,
                                           solid_angle=0.45)

def CO_HCl_model_iterator():
    use_Ng_acceleration = True
    treat_line_overlap = False
    for specie in ("CO","HCl"):
        for T_dust,tau_dust in zip((0,12),(0,0.5)):
            for geo,lp in itertools.product(geometries,line_profile_types):
                if not allowed_geo_param_combination(
                           geometry=geo,line_profile_type=lp,
                           treat_line_overlap=treat_line_overlap):
                    continue
                source = radiative_transfer.Source(
                                      datafilepath=datafilepath[specie],geometry=geo,
                                      line_profile_type=lp,width_v=20*constants.kilo,
                                      use_Ng_acceleration=use_Ng_acceleration,
                                      treat_line_overlap=treat_line_overlap)
                if T_dust != 0 and "LVG" in source.geometry_name:
                    continue
                if specie == "HCl":
                    assert source.emitting_molecule.has_overlapping_lines
                source.update_parameters(N=1e15/constants.centi**2,Tkin=123,
                                         collider_densities={"para-H2":1e5/constants.centi**3},
                                         ext_background=cmb,T_dust=T_dust,
                                         tau_dust=tau_dust)
                source.solve_radiative_transfer()
                yield source

spec_output_types = ("specific intensity","flux density","Rayleigh-Jeans","Planck")

@pytest.mark.filterwarnings("ignore:some lines are overlapping")
@pytest.mark.filterwarnings("ignore:negative optical depth")
@pytest.mark.filterwarnings("ignore:LVG sphere geometry")
def test_emission_at_line_center():
    for source in CO_HCl_model_iterator():
        for transitions in (None,[0,2],1):
            if transitions is None:
                nu0s = source.emitting_molecule.nu0
            else:
                nu0s =  np.array([source.emitting_molecule.rad_transitions[i].nu0
                                  for i in np.atleast_1d(transitions)])
            for output_type in spec_output_types:
                solid_angle = 0.23 if output_type=="flux density" else None
                emission = source.emission_at_line_center(
                               output_type=output_type,transitions=transitions,
                               solid_angle=solid_angle)
                expected_emission = source.spectrum(
                                      nu=nu0s,output_type=output_type,
                                      solid_angle=solid_angle)
                assert np.allclose(emission,expected_emission,atol=0,rtol=1e-3)

def test_transition_index_transformation():
    source = get_general_test_source(specie="CO", width_v=1.23*constants.kilo)
    None_indices = source.transform_transition_indices(indices=None)
    assert np.all(None_indices==np.arange(source.emitting_molecule.n_rad_transitions))
    indices_0d = source.transform_transition_indices(indices=2)
    assert indices_0d.ndim == 1
    assert len(indices_0d) == 1
    assert indices_0d[0] == 2
    indices_1d = source.transform_transition_indices(indices=[2,3,5])
    assert np.all(indices_1d==[2,3,5])
    with pytest.raises(ValueError):
        source.transform_transition_indices(indices=np.zeros((3,4)))

def test_single_integer_and_None_and_invalid_transition_indices():
    source = get_general_test_source(specie="CO", width_v=1.23*constants.kilo)
    source.update_parameters(N=1e15*constants.centi**-2,Tkin=123,
                             collider_densities={"ortho-H2":1e6*constants.centi**-3},
                             ext_background=cmb,T_dust=0,tau_dust=0)
    source.solve_radiative_transfer()
    all_indices = np.arange(source.emitting_molecule.n_rad_transitions)
    invalid_indices = np.ones((2,3,4))
    for output_type in ("intensity","flux"):
        solid_angle = None if output_type!="flux" else 0.54
        f1 = source.frequency_integrated_emission(
              output_type=output_type,transitions=1,solid_angle=solid_angle)
        f2 = source.frequency_integrated_emission(
              output_type=output_type,transitions=[1,],solid_angle=solid_angle)
        assert f1 == f2
        f1_None = source.frequency_integrated_emission(
              output_type=output_type,transitions=None,solid_angle=solid_angle)
        f2_None = source.frequency_integrated_emission(
              output_type=output_type,transitions=all_indices,solid_angle=solid_angle)
        assert np.all(f1_None==f2_None)
        with pytest.raises(ValueError):
            source.frequency_integrated_emission(
                  output_type=output_type,transitions=invalid_indices,solid_angle=solid_angle)
    for output_type in ("specific intensity","flux density","Rayleigh-Jeans",
                        "Planck"):
        solid_angle = None if output_type!="flux density" else 0.54
        e1 = source.emission_at_line_center(
              output_type=output_type,transitions=1,solid_angle=solid_angle)
        e2 = source.emission_at_line_center(
              output_type=output_type,transitions=[1,],solid_angle=solid_angle)
        assert e1 == e2
        e1_None = source.emission_at_line_center(
              output_type=output_type,transitions=None,solid_angle=solid_angle)
        e2_None = source.emission_at_line_center(
              output_type=output_type,transitions=all_indices,solid_angle=solid_angle)
        assert np.all(e1_None==e2_None)
        with pytest.raises(ValueError):
            source.emission_at_line_center(
              output_type=output_type,transitions=invalid_indices,solid_angle=solid_angle)

@pytest.mark.filterwarnings("ignore:some lines are overlapping")
@pytest.mark.filterwarnings("ignore:negative optical depth")
@pytest.mark.filterwarnings("ignore:LVG sphere geometry")
def test_spectrum():
    for source in CO_HCl_model_iterator():
        transition = source.emitting_molecule.rad_transitions[2]
        v = np.linspace(-40,40,50)
        nu0 = transition.nu0
        nu = nu0*(1-v/constants.c)
        specific_intensity = source.spectrum(
                                  nu=nu, output_type="specific intensity")
        for output_type in spec_output_types:
            solid_angle = 1 if output_type=="flux density" else None
            spec = source.spectrum(nu=nu,output_type=output_type,
                                   solid_angle=solid_angle)
            if output_type == "specific intensity":
                pass
            elif output_type == "Rayleigh-Jeans":
                spec = spec*2*nu**2*constants.k/constants.c**2
            elif output_type == "Planck":
                spec = 2*constants.h*nu**3/constants.c**2\
                                 *(np.exp(constants.h*nu/(constants.k*spec))-1)**-1
            elif output_type == "flux density":
                spec = spec/solid_angle
            assert np.allclose(spec,specific_intensity,atol=0,rtol=1e-6)


class TestSpectrumPhysics():

    N_thick = 1e20*constants.centi**-2
    N_regular = 1e15*constants.centi**-2
    N_thin = 1e12*constants.centi**-2
    Tkin = 123
    collider_densities = {"ortho-H2":1e5*constants.centi**-3,
                          "para-H2":1e5*constants.centi**-3}
    specie = "CO"
    width_v = 1*constants.kilo
    T_dust = 45
    tau_dust_thick = 50
    general_params = {"N":1e14*constants.centi**-2,"Tkin":101,
                      "collider_densities":{"ortho-H2":1e5*constants.centi**-3},
                      "ext_background":cmb,"T_dust":0,"tau_dust":0}
    general_params_dust = {"N":1e14*constants.centi**-2,"Tkin":101,
                           "collider_densities":{"ortho-H2":1e5*constants.centi**-3},
                           "ext_background":cmb,
                           "T_dust":lambda nu: np.ones_like(nu)*78,
                           "tau_dust":lambda nu: np.ones_like(nu)*1.2}

    def test_thick(self):
        for source in general_source_iterator(specie=self.specie,width_v=self.width_v):
            #thick lines:
            source.update_parameters(
                  N=self.N_thick,Tkin=self.Tkin,
                  collider_densities=self.collider_densities,
                  ext_background=cmb,T_dust=0,tau_dust=0)
            source.solve_radiative_transfer()
            thick = source.tau_nu0_individual_transitions > 20
            assert np.any(thick)
            T_Planck = source.emission_at_line_center(output_type="Planck")
            assert np.allclose(T_Planck[thick],source.Tex[thick],atol=0,rtol=1e-2)
            nu = source.emitting_molecule.nu0[thick]
            T_Planck_spec = source.spectrum(nu=nu,output_type="Planck")
            assert np.allclose(T_Planck_spec,source.Tex[thick],atol=0,rtol=1e-2)
            #thick dust:
            if "LVG" not in source.geometry_name:
                source.update_parameters(
                      N=self.N_thin,Tkin=self.Tkin,
                      collider_densities=self.collider_densities,
                      ext_background=cmb,T_dust=self.T_dust,tau_dust=self.tau_dust_thick)
                source.solve_radiative_transfer()
                T_Planck = source.emission_at_line_center(output_type="Planck")
                assert np.allclose(T_Planck,self.T_dust,atol=0,rtol=1e-3)
                T_Planck_spec = source.spectrum(nu=source.emitting_molecule.nu0,
                                                output_type="Planck")
                assert np.allclose(T_Planck_spec,self.T_dust,atol=0,rtol=1e-3)

    def test_RJ_Planck_consistency(self):
        for source in general_source_iterator(specie=self.specie,width_v=self.width_v):
            for T_dust,tau_dust in zip((0,45),(0,1.2)):
                if "LVG" in source.geometry_name and T_dust > 0:
                    continue
                source.update_parameters(
                      N=self.N_thick,Tkin=self.Tkin,
                      collider_densities=self.collider_densities,
                      ext_background=cmb,T_dust=T_dust,tau_dust=tau_dust)
                source.solve_radiative_transfer()
                T_Planck = source.emission_at_line_center(output_type="Planck")
                T_Planck_spec = source.spectrum(nu=source.emitting_molecule.nu0,
                                                output_type="Planck")
                T_RJ = source.emission_at_line_center(output_type="Rayleigh-Jeans")
                T_RJ_spec = source.spectrum(nu=source.emitting_molecule.nu0,
                                            output_type="Rayleigh-Jeans")
                for Planck,RJ in zip((T_Planck,T_Planck_spec),(T_RJ,T_RJ_spec)):
                    intensity_from_T_Planck = helpers.B_nu(T=Planck,nu=source.emitting_molecule.nu0)
                    T_RJ_from_intensity = helpers.RJ_brightness_temperature(
                                     specific_intensity=intensity_from_T_Planck,
                                     nu=source.emitting_molecule.nu0)
                    assert np.allclose(RJ,T_RJ_from_intensity,atol=0,rtol=1e-5)

    def test_explicitly(self):
        #calculate brightness temperature explicitly from Tex and tau_nu for a simple
        #case (no overlap)
        for source in general_source_iterator(specie=self.specie,width_v=self.width_v):
            for T_dust,tau_dust in zip((0,45,lambda nu: np.ones_like(nu)*87),
                                       (0,1.2,lambda nu:np.ones_like(nu)*0.36)):
                if "LVG" in source.geometry_name and T_dust != 0:
                    continue
                source.update_parameters(
                      N=self.N_regular,Tkin=self.Tkin,
                      collider_densities=self.collider_densities,
                      ext_background=cmb,T_dust=T_dust,tau_dust=tau_dust)
                source.solve_radiative_transfer()
                T_Planck = source.emission_at_line_center(output_type="Planck")
                T_Planck_spec = source.spectrum(nu=source.emitting_molecule.nu0,
                                                output_type="Planck")
                T_RJ = source.emission_at_line_center(output_type="Rayleigh-Jeans")
                T_RJ_spec = source.spectrum(nu=source.emitting_molecule.nu0,
                                                output_type="Rayleigh-Jeans")
                nu = source.emitting_molecule.nu0
                tau_d = source.rate_equations.tau_dust(nu)
                T_d = source.rate_equations.T_dust(nu)
                tau_tot = tau_d+source.tau_nu0_individual_transitions
                S = tau_d*helpers.B_nu(nu=nu,T=T_d)\
                     +source.tau_nu0_individual_transitions*helpers.B_nu(nu=nu,T=source.Tex)
                S /= tau_tot
                intensity_kwargs = {"tau_nu":tau_tot,"source_function":S}
                if source.geometry_name == "LVG sphere":
                    specific_intensity = escape_probability.specific_intensity_nu0_lvg_sphere(
                                      **intensity_kwargs)
                else:
                    specific_intensity = source.geometry.specific_intensity(**intensity_kwargs)
                T_Planck_expected = helpers.Planck_brightness_temperature(
                                     specific_intensity=specific_intensity,nu=nu)
                T_RJ_expected = helpers.RJ_brightness_temperature(
                                            specific_intensity=specific_intensity, nu=nu)
                for P in (T_Planck,T_Planck_spec):
                    assert np.allclose(P,T_Planck_expected,atol=0,rtol=1e-4)
                for RJ in (T_RJ,T_RJ_spec):
                    assert np.allclose(RJ,T_RJ_expected,atol=0,rtol=1e-4)


# class TestModelGrid():

#     ext_backgrounds = {'CMB':helpers.generate_CMB_background(z=2),
#                        'zero':0}
#     N_values = np.array((1e14,1e16))/constants.centi**2
#     Tkin_values = [20,200,231]
#     collider_densities_values = {'ortho-H2':np.array((1e3,1e4))/constants.centi**3,
#                                  'para-H2':np.array((1e4,1e7))/constants.centi**3}
#     grid_kwargs_no_dust = {'ext_backgrounds':ext_backgrounds,'N_values':N_values,
#                            'Tkin_values':Tkin_values,
#                            'collider_densities_values':collider_densities_values}
#     grid_kwargs_with_dust = grid_kwargs_no_dust.copy()
#     grid_kwargs_with_dust['T_dust'] = lambda nu: np.ones_like(nu)*111
#     #make dust optically thin to allow computation of individual line fluxes
#     grid_kwargs_with_dust['tau_dust'] = lambda nu: np.ones_like(nu)*0.05

#     @staticmethod
#     def generate_source():
#         source = radiative_transfer.Source(
#                               datafilepath=datafilepath['CO'],geometry='static sphere',
#                               line_profile_type='rectangular',width_v=1.4*constants.kilo,
#                               use_Ng_acceleration=True,treat_line_overlap=False)
#         return source

#     def test_invalid_coll_densities(self):
#         invalid_coll_densities = [{'ortho-H2':[1,2],'para-H2':[1,]},
#                                   {'ortho-H2':[1,],'para-H2':[1,2]},
#                                   {'ortho-H2':[1,2],'para-H2':[1,3,4,5,5]},]
#         source = self.generate_source()
#         for coll_densities in invalid_coll_densities:
#             grid_kwargs = self.grid_kwargs_no_dust.copy()
#             grid_kwargs['collider_densities_values'] = coll_densities
#             with pytest.raises(AssertionError):
#                 grid = source.efficient_parameter_iterator(**grid_kwargs)
#                 list(grid)

#     def test_grid(self):
#         source = self.generate_source()
#         nu = source.emitting_molecule.nu0
#         for grid_kwargs in (self.grid_kwargs_no_dust,self.grid_kwargs_with_dust):
#             for params in source.efficient_parameter_iterator(**grid_kwargs):
#                 if "T_dust" in grid_kwargs:
#                     expected_T_dust = grid_kwargs["T_dust"](nu)
#                 else:
#                     expected_T_dust = np.zeros_like(nu)
#                 assert np.all(source.rate_equations.T_dust(nu)==
#                               expected_T_dust)
#                 if "tau_dust" in grid_kwargs:
#                     expected_tau_dust = grid_kwargs["tau_dust"](nu)
#                 else:
#                     expected_tau_dust = np.zeros_like(nu)
#                 assert np.all(source.rate_equations.tau_dust(nu)==
#                               expected_tau_dust)
#                 ext_background_name = params["ext_background"]
#                 if ext_background_name == "zero":
#                     expected_background = np.zeros_like(nu)
#                 else:
#                     expected_background = grid_kwargs["ext_backgrounds"][ext_background_name](nu)
#                 assert np.all(source.rate_equations.ext_background(nu)==expected_background)
#                 assert source.rate_equations.N == params["N"]
#                 assert source.rate_equations.Tkin == params["Tkin"]
#                 assert source.rate_equations.collider_densities\
#                                                == params["collider_densities"]

#     @pytest.mark.filterwarnings("ignore:negative optical depth")
#     @pytest.mark.filterwarnings("ignore:invalid value encountered")
#     def test_grid_explicitly(self):
#         source = self.generate_source()
#         v = np.linspace(-source.emitting_molecule.width_v,source.emitting_molecule.width_v,10)
#         nu = source.emitting_molecule.nu0[1]*(1-v/constants.c)
#         for grid_kwargs in (self.grid_kwargs_no_dust,self.grid_kwargs_with_dust):
#             for params in source.efficient_parameter_iterator(**grid_kwargs):
#                 source.solve_radiative_transfer()
#                 check_source = self.generate_source()
#                 T_dust = grid_kwargs["T_dust"] if "T_dust" in grid_kwargs else 0
#                 tau_dust = grid_kwargs["tau_dust"] if "tau_dust" in grid_kwargs\
#                                else 0
#                 check_source.update_parameters(
#                           N=params["N"],Tkin=params["Tkin"],
#                           collider_densities=params["collider_densities"],
#                           ext_background=grid_kwargs["ext_backgrounds"][params["ext_background"]],
#                           T_dust=T_dust,tau_dust=tau_dust)
#                 check_source.solve_radiative_transfer()
#                 assert np.all(source.Tex==check_source.Tex)
#                 assert np.all(source.level_pop==check_source.level_pop)
#                 assert np.all(source.tau_nu0_individual_transitions
#                               ==check_source.tau_nu0_individual_transitions)
#                 assert np.all(source.frequency_integrated_emission(output_type="intensity")
#                               ==check_source.frequency_integrated_emission(output_type="intensity"))
#                 assert np.all(source.spectrum(nu=nu,output_type="Planck")==
#                               check_source.spectrum(nu=nu,output_type="Planck"))
#                 assert np.all(source.emission_at_line_center(output_type="Rayleigh-Jeans")
#                               ==check_source.emission_at_line_center(output_type="Rayleigh-Jeans"))


def test_print_results():
    for source in general_source_iterator(specie='CO',width_v=1*constants.kilo):
        source.update_parameters(ext_background=helpers.generate_CMB_background(),
                                N=1e14/constants.centi**2,Tkin=33.33,
                                collider_densities={'ortho-H2':1e3/constants.centi**3},
                                T_dust=0,tau_dust=0)
        source.solve_radiative_transfer()
        source.print_results()

def test_line_profile_averaging():
    #for a molecule with no overlapping lines, nu0 and averaging should give 
    #same result if line profile is rectangular
    level_pops = []
    for treat_line_overlap in (True,False):
        src = radiative_transfer.Source(
                          datafilepath=datafilepath['CO'],geometry='static sphere',
                          line_profile_type='rectangular',width_v=1*constants.kilo,
                          use_Ng_acceleration=True,
                          treat_line_overlap=treat_line_overlap)
        src.update_parameters(ext_background=helpers.generate_CMB_background(),
                              N=1e14/constants.centi**2,Tkin=33.33,
                              collider_densities={'ortho-H2':1e3/constants.centi**3},
                              T_dust=0,tau_dust=0)
        src.solve_radiative_transfer()
        level_pops.append(src.level_pop)
    assert np.allclose(*level_pops,atol=0,rtol=1e-3)
        

class TestPhysics():

    zero_coll_dens = collider_densities={'para-H2':0,'ortho-H2':0}

    def cloud_iterator(self):
        for specie,width_v in zip(('CO','HCl'),(1*constants.kilo,20*constants.kilo)):
            for source in general_source_iterator(specie=specie,width_v=width_v):
                if specie == 'HCl':
                    assert source.emitting_molecule.has_overlapping_lines
                yield source

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_LTE_from_coll(self):
        collider_densities={'para-H2':1e11/constants.centi**3,
                            'ortho-H2':1e11/constants.centi**3}
        Tkin = 50
        for source in self.cloud_iterator():
            source.update_parameters(N=1e10/constants.centi**2,Tkin=Tkin,
                                    collider_densities=collider_densities,
                                    ext_background=0,T_dust=0,
                                    tau_dust=0)
            source.solve_radiative_transfer()
            assert np.all(source.tau_nu0_individual_transitions<0.1)
            Boltzmann_level_population = source.emitting_molecule.Boltzmann_level_population(T=Tkin)
            assert np.allclose(source.level_pop,Boltzmann_level_population,atol=1e-5,rtol=1e-2)

    @pytest.mark.filterwarnings("ignore:negative optical depth")
    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_level_populations_no_excitation(self):
        for source in self.cloud_iterator():
            source.update_parameters(N=1e10/constants.centi**2,Tkin=111,
                                    collider_densities=self.zero_coll_dens,
                                    ext_background=0,T_dust=0,
                                    tau_dust=0)
            source.solve_radiative_transfer()
            expected_level_pop = np.zeros(len(source.emitting_molecule.levels))
            expected_level_pop[0] = 1
            assert np.all(source.level_pop==expected_level_pop)

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    def test_LTE_from_ext_background(self):
        Tbg = 45
        def ext_background(nu):
            return helpers.B_nu(nu=nu,T=Tbg)
        for source in self.cloud_iterator():
            source.update_parameters(N=1/constants.centi**2,Tkin=123,
                                    collider_densities=self.zero_coll_dens,
                                    ext_background=ext_background,T_dust=0,
                                    tau_dust=0)
            source.solve_radiative_transfer()
            Boltzmann_level_population = source.emitting_molecule.Boltzmann_level_population(T=Tbg)
            assert np.allclose(source.level_pop,Boltzmann_level_population,atol=1e-5,rtol=1e-2)

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    def test_LTE_from_dust(self):
        T_dust_value = 123
        def T_dust(nu):
            return np.ones_like(nu)*T_dust_value
        def tau_dust(nu):
            #for some reason I do not really understand, the dust needs to be
            #extremely optically thick to make LTE
            return np.ones_like(nu)*1500
        for source in self.cloud_iterator():
            if 'LVG' in source.geometry_name:
                continue
            source.update_parameters(N=1e-4/constants.centi**2,Tkin=48,
                                    collider_densities=self.zero_coll_dens,
                                    ext_background=0,T_dust=T_dust,
                                    tau_dust=tau_dust)
            source.solve_radiative_transfer()
            Boltzmann_level_population\
                   = source.emitting_molecule.Boltzmann_level_population(T=T_dust_value)
            assert np.allclose(source.level_pop,Boltzmann_level_population,
                               atol=1e-5,rtol=1e-2)


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_single_transition_molecule():
    #test C+ which only has a single transition
    #just a single transition is an edge case, so let's see if the radiative transfer
    #can be solved
    solid_angle = 1
    transitions = [0,]
    N = 1e-4/constants.centi**2
    Tkin = 48
    collider_densities = {'H':10/constants.centi**3,'e':123/constants.centi**3}
    ext_background_values = [0,cmb]
    T_dust_values = [0,50,lambda nu: np.ones_like(nu)*451]
    tau_dust_values = [0,0.01,12,lambda nu: np.ones_like(nu)*2.3]
    width_v = 1*constants.kilo
    for source in general_source_iterator(specie='C+',width_v=width_v):
        for ext_background in ext_background_values:
            for T_dust,tau_dust in zip(T_dust_values,tau_dust_values):
                if 'LVG' in source.geometry_name and T_dust != 0:
                    continue
                source.update_parameters(
                        N=N,Tkin=Tkin,collider_densities=collider_densities,
                        ext_background=ext_background,T_dust=T_dust,
                        tau_dust=tau_dust)
                source.solve_radiative_transfer()
                nu0 = source.emitting_molecule.nu0[0]
                if source.rate_equations.tau_dust(nu0) < 0.1:
                    source.frequency_integrated_emission(
                             output_type="intensity",transitions=transitions)
                v = np.linspace(-2*width_v,2*width_v,10)
                nu = nu0*(1-v/constants.c)
                source.tau_nu(nu=nu)
                source.spectrum(output_type="flux density",solid_angle=solid_angle,nu=nu)
                source.print_results()


class TestFrequencyIntegratedEmission():

    @staticmethod
    def generate_and_solve_source(geometry,N,line_profile_type):
        source = radiative_transfer.Source(
                              datafilepath=datafilepath["CO"],geometry=geometry,
                              line_profile_type=line_profile_type,width_v=1*constants.kilo)
        source.update_parameters(N=N,Tkin=123,
                                 collider_densities={"ortho-H2":1e6*constants.centi**-3},
                                 ext_background=helpers.generate_CMB_background(),
                                 T_dust=0,tau_dust=0)
        source.solve_radiative_transfer()
        return source

    def test_wrong_input(self):
        source = self.generate_and_solve_source(
                                      geometry="static sphere",
                                      N=1e12/constants.centi**-2,
                                      line_profile_type="Gaussian")
        with pytest.raises(ValueError):
            source.frequency_integrated_emission(
                        output_type="flux")
        with pytest.raises(ValueError):
            source.frequency_integrated_emission(
                            output_type="flux",solid_angle=None)
        with pytest.raises(ValueError):
            source.frequency_integrated_emission(
                            output_type="asdf")

    def test_with_physics_thin(self):
        N = 1e12/constants.centi**2
        for line_profile in ("Gaussian","rectangular"):
            source = self.generate_and_solve_source(
                                          geometry="static sphere",N=N,
                                          line_profile_type=line_profile)
            distance = 1*constants.parsec
            sphere_radius = 1*constants.au
            sphere_volume = 4/3*sphere_radius**3*np.pi
            sphere_Omega = sphere_radius**2*np.pi/distance**2
            number_density = N/(2*sphere_radius)
            total_mol = number_density*sphere_volume
            expected_flux = []
            for i,line in enumerate(source.emitting_molecule.rad_transitions):
                up_level_pop = source.level_pop[line.up.index]
                f = total_mol*up_level_pop*line.A21*line.Delta_E\
                       /(4*np.pi*distance**2)
                expected_flux.append(f)
            fluxes = source.frequency_integrated_emission(
                              output_type="flux",solid_angle=sphere_Omega)
            assert np.allclose(fluxes,expected_flux,atol=0,rtol=1e-2)
        
    def test_with_physics_thick(self):
        N = 1e19*constants.centi**-2
        source = self.generate_and_solve_source(
                                      geometry="static slab",N=N,
                                      line_profile_type="rectangular")
        thick = source.tau_nu0_individual_transitions > 10
        assert np.any(thick)
        for i,line in enumerate(source.emitting_molecule.rad_transitions):
            if not thick[i]:
                continue
            expected_specific_intensity = helpers.B_nu(nu=line.nu0,T=source.Tex[i])
            expected_intensity = expected_specific_intensity*line.line_profile.width_nu
            intensity = source.frequency_integrated_emission(
                             output_type="intensity",transitions=[i,])
            assert np.isclose(intensity,expected_intensity,atol=0,rtol=1e-3)