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

def iterate_all_cloud_params():
    return itertools.product(geometries,line_profile_types,use_ng_options,
                             treat_overlap_options)

def iterate_allowed_cloud_params():
    for geo,lp,use_ng,treat_overlap in iterate_all_cloud_params():
        if allowed_geo_param_combination(geometry=geo,line_profile_type=lp,
                                         treat_line_overlap=treat_overlap):
            yield geo,lp,use_ng,treat_overlap

def general_cloud_iterator(specie,width_v):
    for geo,lp,use_ng,treat_overlap in iterate_allowed_cloud_params():
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
            for geo,lp,use_ng,treat_overlap in iterate_allowed_cloud_params():
                with pytest.raises(AssertionError):
                    radiative_transfer.Source(
                                      datafilepath=datafilepath['CO'],geometry=geo,
                                      line_profile_type=lp,width_v=width_v,
                                      use_Ng_acceleration=use_ng,
                                      treat_line_overlap=treat_overlap)

    def test_LVG_cannot_treat_overlap(self):
        for geo,lp,use_ng,average in iterate_allowed_cloud_params():
            if 'LVG' in geo:
                with pytest.raises(ValueError):
                    radiative_transfer.Source(
                                          datafilepath=datafilepath['HCl'],
                                          geometry=geo,line_profile_type=lp,
                                          width_v=10*constants.kilo,
                                          use_Ng_acceleration=use_ng,
                                          treat_line_overlap=True)    

    def test_warning_overlapping_lines(self):
        for geo,lp,use_ng,average in iterate_allowed_cloud_params():
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
        for source in general_cloud_iterator(specie='CO',width_v=1*constants.kilo):
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
        for source in general_cloud_iterator(specie='CO',width_v=1*constants.kilo):
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
        for source in general_cloud_iterator(specie='CO',width_v=1*constants.kilo):
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
        for source in general_cloud_iterator(specie='CO',width_v=1*constants.kilo):
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
        for source in general_cloud_iterator(specie='CO',width_v=1*constants.kilo):
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
        for source in general_cloud_iterator(specie='CO',width_v=1*constants.kilo):
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
        for source in general_cloud_iterator(specie='CO',width_v=1*constants.kilo):
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


class TestModelGrid():

    source = radiative_transfer.Source(
                          datafilepath=datafilepath['CO'],geometry='static sphere',
                          line_profile_type='rectangular',width_v=1.4*constants.kilo,
                          use_Ng_acceleration=True,treat_line_overlap=False)
    ext_backgrounds = {'CMB':helpers.generate_CMB_background(z=2),
                       'zero':0}
    N_values = np.array((1e14,1e16))/constants.centi**2
    Tkin_values = [20,200,231]
    collider_densities_values = {'ortho-H2':np.array((1e3,1e4))/constants.centi**3,
                                 'para-H2':np.array((1e4,1e7))/constants.centi**3}
    grid_kwargs_no_dust = {'ext_backgrounds':ext_backgrounds,'N_values':N_values,
                           'Tkin_values':Tkin_values,
                           'collider_densities_values':collider_densities_values}
    grid_kwargs_with_dust = grid_kwargs_no_dust.copy()
    grid_kwargs_with_dust['T_dust'] = lambda nu: np.ones_like(nu)*111
    #make dust optically thin to allow computation of individual line fluxes
    grid_kwargs_with_dust['tau_dust'] = lambda nu: np.ones_like(nu)*0.05
    requested_output = ['level_pop','Tex','tau_nu0_individual_transitions',
                        'fluxes_of_individual_transitions','tau_nu','spectrum']
    transitions = [3,4]
    solid_angle = 1
    nu = np.linspace(115.27,115.28,10)*constants.giga
    
    def test_wrong_requested_output(self):
        wrong_requested_output = ['Tex','levelpop']
        with pytest.raises(AssertionError):
            for model in self.source.model_grid(**self.grid_kwargs_no_dust,
                                               requested_output=wrong_requested_output):
                pass

    def test_insufficient_input(self):
        #test that errors are thrown if solid_angle or nu are not provided
        for request in ('fluxes_of_individual_transitions','spectrum'):
            with pytest.raises(AssertionError): 
                for model in self.source.model_grid(
                                **self.grid_kwargs_no_dust,requested_output=[request,],
                                nu=self.nu):
                    pass
        for request in ('tau_nu','spectrum'):
            with pytest.raises(AssertionError):
                for model in self.source.model_grid(
                                **self.grid_kwargs_no_dust,requested_output=[request,],
                                solid_angle=self.solid_angle):
                    pass

    def tests_invalid_coll_densities(self):
        invalid_coll_densities = [{'ortho-H2':[1,2],'para-H2':[1,]},
                                  {'ortho-H2':[1,],'para-H2':[1,2]},
                                  {'ortho-H2':[1,2],'para-H2':[1,3,4,5,5]},]
        for coll_densities in invalid_coll_densities:
            grid_kwargs = self.grid_kwargs_no_dust.copy()
            grid_kwargs['collider_densities_values'] = coll_densities
            with pytest.raises(AssertionError):
                grid = self.source.model_grid(
                                **grid_kwargs,requested_output=self.requested_output,
                                solid_angle=self.solid_angle)
                list(grid)

    @pytest.mark.filterwarnings("ignore:negative optical depth")
    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_grid(self):
        for grid_kwargs in (self.grid_kwargs_no_dust,self.grid_kwargs_with_dust):
            iterator = self.source.model_grid(
                            **grid_kwargs,requested_output=self.requested_output,
                            solid_angle=self.solid_angle,transitions=self.transitions,
                            nu=self.nu)
            models = [m for m in iterator]
            n_check_models = 0
            for backgroundID,ext_background in self.ext_backgrounds.items():
                for N in self.N_values:
                    for Tkin in self.Tkin_values:
                        for n_ortho,n_para in zip(self.collider_densities_values['ortho-H2'],
                                                  self.collider_densities_values['para-H2']):
                            collider_densities = {'ortho-H2':n_ortho,'para-H2':n_para}
                            update_kwargs = {'ext_background':ext_background,'N':N,
                                             'Tkin':Tkin,
                                             'collider_densities':collider_densities}
                            if 'T_dust' in grid_kwargs:
                                update_kwargs['T_dust'] = grid_kwargs['T_dust']
                                update_kwargs['tau_dust'] = grid_kwargs['tau_dust']
                            self.source.update_parameters(**update_kwargs)
                            self.source.solve_radiative_transfer()
                            fluxes = self.source.fluxes_of_individual_transitions(
                                                 solid_angle=self.solid_angle,
                                                 transitions=self.transitions)
                            tau_nu = self.source.tau_nu(nu=self.nu)
                            spectrum = self.source.spectrum(
                                                solid_angle=self.solid_angle,nu=self.nu)
                            matching_models = [m for m in models if
                                               m['ext_background']==backgroundID
                                               and m['N']==N and m['Tkin']==Tkin
                                               and m['collider_densities']==collider_densities]
                            assert len(matching_models) == 1
                            matching_model = matching_models[0]
                            assert np.all(matching_model['level_pop'] == self.source.level_pop)
                            assert np.all(matching_model['Tex']
                                          == self.source.Tex[self.transitions])
                            assert np.all(matching_model['tau_nu0_individual_transitions']
                                          == self.source.tau_nu0_individual_transitions[self.transitions])
                            assert np.all(matching_model['fluxes_of_individual_transitions']
                                          == fluxes)
                            assert np.all(matching_model['tau_nu'] == tau_nu)
                            assert np.all(matching_model['spectrum']==spectrum)
                            n_check_models += 1
            assert n_check_models == len(models)

    def test_error_handling(self):
        requested_output = ['level_pop','Tex','tau_nu0_individual_transitions']
        kwargs = {'ext_backgrounds':{'zero':0},'N_values':[np.nan,],'Tkin_values':[120,],
                  'collider_densities_values':{coll:[200,] for  coll in
                                               ('para-H2','ortho-H2')}}
        grid_with_failing_models = self.source.model_grid(
                                       **kwargs,requested_output=requested_output)
        for model in grid_with_failing_models:
            for rout in requested_output:
                assert model[rout] is None

    def test_brightness_temp(self):
        raise NotImplementedError("need to implemente brightness temp support")


def test_print_results():
    for source in general_cloud_iterator(specie='CO',width_v=1*constants.kilo):
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
            for source in general_cloud_iterator(specie=specie,width_v=width_v):
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
            LTE_level_pop = source.emitting_molecule.LTE_level_pop(T=Tkin)
            assert np.allclose(source.level_pop,LTE_level_pop,atol=1e-5,rtol=1e-2)

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
            LTE_level_pop = source.emitting_molecule.LTE_level_pop(T=Tbg)
            assert np.allclose(source.level_pop,LTE_level_pop,atol=1e-5,rtol=1e-2)

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
            LTE_level_pop = source.emitting_molecule.LTE_level_pop(T=T_dust_value)
            assert np.allclose(source.level_pop,LTE_level_pop,atol=1e-5,rtol=1e-2)


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
    for source in general_cloud_iterator(specie='C+',width_v=width_v):
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
                    source.fluxes_of_individual_transitions(
                             solid_angle=solid_angle,transitions=transitions)
                v = np.linspace(-2*width_v,2*width_v,10)
                nu = nu0*(1-v/constants.c)
                source.tau_nu(nu=nu)
                source.spectrum(solid_angle=solid_angle,nu=nu)
                source.print_results()


class TestBrightnessTemp():

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

    def test_invalid_temp_type(self):
        source = get_general_test_source(specie=self.specie, width_v=self.width_v)
        source.update_parameters(**self.general_params)
        source.solve_radiative_transfer()
        for temp_type in ("Planckd","Rayleigh-Jeanss",""):
            with pytest.raises(ValueError):
                source.brightness_temperature_nu0(temperature_type=temp_type)

    def test_None_transitions(self):
        source = get_general_test_source(specie=self.specie, width_v=self.width_v)
        source.update_parameters(**self.general_params)
        source.solve_radiative_transfer()
        for temp_type in ("Planck","Rayleigh-Jeans"):
            T_None = source.brightness_temperature_nu0(temperature_type=temp_type,
                                                       transitions=None)
            T_default = source.brightness_temperature_nu0(temperature_type=temp_type,
                                                       transitions=None)
            T_explicit = source.brightness_temperature_nu0(
                        temperature_type=temp_type,
                        transitions=list(range(source.emitting_molecule.n_rad_transitions)))
            assert np.all(T_None==T_explicit)
            assert np.all(T_default==T_explicit)

    def test_brightness_temp_nu0_thick(self):
        for source in general_cloud_iterator(specie=self.specie,width_v=self.width_v):
            #thick lines:
            source.update_parameters(
                  N=self.N_thick,Tkin=self.Tkin,
                  collider_densities=self.collider_densities,
                  ext_background=cmb,T_dust=0,tau_dust=0)
            source.solve_radiative_transfer()
            thick = source.tau_nu0_individual_transitions > 20
            assert np.any(thick)
            T_Planck = source.brightness_temperature_nu0(temperature_type="Planck")
            assert np.allclose(T_Planck[thick],source.Tex[thick],atol=0,rtol=1e-2)
            #thick dust:
            if "LVG" not in source.geometry_name:
                source.update_parameters(
                      N=self.N_thin,Tkin=self.Tkin,
                      collider_densities=self.collider_densities,
                      ext_background=cmb,T_dust=self.T_dust,tau_dust=self.tau_dust_thick)
                source.solve_radiative_transfer()
                T_Planck = source.brightness_temperature_nu0(temperature_type="Planck")
                assert np.allclose(T_Planck,self.T_dust,atol=0,rtol=1e-3)

    def test_RJ_Planck_consistency(self):
        for source in general_cloud_iterator(specie=self.specie,width_v=self.width_v):
            for T_dust,tau_dust in zip((0,45),(0,1.2)):
                if "LVG" in source.geometry_name and T_dust > 0:
                    continue
                source.update_parameters(
                      N=self.N_thick,Tkin=self.Tkin,
                      collider_densities=self.collider_densities,
                      ext_background=cmb,T_dust=T_dust,tau_dust=tau_dust)
                source.solve_radiative_transfer()
                T_Planck = source.brightness_temperature_nu0(temperature_type="Planck")
                intensity_from_T_Planck = helpers.B_nu(T=T_Planck,nu=source.emitting_molecule.nu0)
                T_RJ = source.brightness_temperature_nu0(temperature_type="Rayleigh-Jeans")
                T_RJ_from_intensity = helpers.RJ_brightness_temperature(
                                     intensity=intensity_from_T_Planck,
                                     nu=source.emitting_molecule.nu0)
                assert np.allclose(T_RJ,T_RJ_from_intensity,atol=0,rtol=1e-5)

    def test_explicitly(self):
        #calculate brightness temperature explicitly from Tex and tau_nu for a simple
        #case (no overlap)
        for source in general_cloud_iterator(specie=self.specie,width_v=self.width_v):
            for T_dust,tau_dust in zip((0,45,lambda nu: np.ones_like(nu)*87),
                                       (0,1.2,lambda nu:np.ones_like(nu)*0.36)):
                if "LVG" in source.geometry_name and T_dust > 0:
                    continue
                source.update_parameters(
                      N=self.N_regular,Tkin=self.Tkin,
                      collider_densities=self.collider_densities,
                      ext_background=cmb,T_dust=T_dust,tau_dust=tau_dust)
                source.solve_radiative_transfer()
                T_Planck = source.brightness_temperature_nu0(temperature_type="Planck")
                T_RJ = source.brightness_temperature_nu0(temperature_type="Rayleigh-Jeans")
                nu = source.emitting_molecule.nu0
                tau_tot = tau_dust+source.tau_nu0_individual_transitions
                S = tau_dust*helpers.B_nu(nu=nu,T=T_dust)\
                     +source.tau_nu0_individual_transitions*helpers.B_nu(nu=nu,T=source.Tex)
                S /= tau_tot
                Omega = 0.8
                flux_kwargs = {"tau_nu":tau_tot,"source_function":S,
                               "solid_angle":Omega}
                if source.geometry_name == "LVG sphere":
                    intensity = escape_probability.compute_flux_nu0_lvg_sphere(
                                      **flux_kwargs)/Omega
                else:
                    intensity = source.geometry.compute_flux_nu(**flux_kwargs)/Omega
                T_Planck_expected = helpers.Planck_brightness_temperature(
                                     intensity=intensity,nu=nu)
                T_RJ_expected = helpers.RJ_brightness_temperature(
                                            intensity=intensity, nu=nu)
                assert np.allclose(T_Planck,T_Planck_expected,atol=0,rtol=1e-4)
                assert np.allclose(T_RJ,T_RJ_expected,atol=0,rtol=1e-4)

    @pytest.mark.filterwarnings(r"ignore:LVG sphere geometry")
    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    def test_brightness_temp_nu0_spec_consistency(self):
        #here I should also test overlapping lines
        for specie in ("CO","HCl"):
            for source in general_cloud_iterator(specie=specie,width_v=10*constants.kilo):
                if specie == "HCl":
                    assert source.emitting_molecule.has_overlapping_lines
                for params in (self.general_params,self.general_params_dust):
                    if "LVG" in source.geometry_name and params["T_dust"] > 0:
                        continue
                    source.update_parameters(**params)
                    source.solve_radiative_transfer()
                    for temp_type in ("Planck","Rayleigh-Jeans"):
                        nu = source.emitting_molecule.nu0
                        spec_T = source.brightness_temperature_spectrum(
                                                    nu=nu,temperature_type=temp_type)
                        nu0_T = source.brightness_temperature_nu0(
                                         temperature_type=temp_type)
                        assert np.allclose(spec_T,nu0_T,atol=0,rtol=1e-4)

    def test_brightness_temp_spectrum(self):
        raise NotImplementedError