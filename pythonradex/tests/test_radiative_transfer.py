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


here = os.path.dirname(os.path.abspath(__file__))
datafilepath = {'CO':os.path.join(here,'LAMDA_files/co.dat'),
                'HCl':os.path.join(here,'LAMDA_files/hcl.dat')}

line_profile_types = ('rectangular','Gaussian')
geometries = tuple(radiative_transfer.Cloud.geometries.keys())
iteration_modes = ('ALI','LI')
use_ng_options = (True,False)
average_options = (True,False)

def get_general_test_cloud(specie):
    return radiative_transfer.Cloud(
                          datafilepath=datafilepath[specie],geometry='uniform sphere',
                          line_profile_type='Gaussian',width_v=1*constants.kilo,
                          iteration_mode='ALI')

def allowed_geo_lp_combination(geometry,line_profile_type):
    if geometry in ('LVG sphere','LVG slab','LVG sphere RADEX')\
                                    and line_profile_type=='Gaussian':
        return False
    else:
        return True

def iterate_all_cloud_params():
    return itertools.product(geometries,line_profile_types,iteration_modes,
                             use_ng_options,average_options)

def iterate_allowed_cloud_params():
    for geo,lp,mode,use_ng,average in iterate_all_cloud_params():
        if allowed_geo_lp_combination(geometry=geo,line_profile_type=lp):
            yield geo,lp,mode,use_ng,average

def general_cloud_iterator(specie,width_v):
    for geo,lp,mode,use_ng,average in iterate_allowed_cloud_params():
        cld = radiative_transfer.Cloud(
                              datafilepath=datafilepath[specie],geometry=geo,
                              line_profile_type=lp,width_v=width_v,iteration_mode=mode,
                              use_NG_acceleration=use_ng,
                              average_over_line_profile=average)
        yield cld


class TestInitialisation():

    def test_too_large_width_v(self):
        width_vs = np.array((1e4,1.2e4,1e5))*constants.kilo
        for width_v in width_vs:
            for geo,lp,mode,use_ng,average in iterate_allowed_cloud_params():
                with pytest.raises(AssertionError):
                    radiative_transfer.Cloud(
                                      datafilepath=datafilepath['CO'],geometry=geo,
                                      line_profile_type=lp,width_v=width_v,
                                      iteration_mode=mode,
                                      use_NG_acceleration=use_ng,
                                      average_over_line_profile=average)    

    def test_overlapping_needs_averaging(self):
        for geo,lp,mode,use_ng,average in iterate_allowed_cloud_params():
            with pytest.raises(ValueError):
                radiative_transfer.Cloud(
                                      datafilepath=datafilepath['HCl'],
                                      geometry=geo,line_profile_type=lp,
                                      width_v=10*constants.kilo,
                                      iteration_mode=mode,use_NG_acceleration=use_ng,
                                      average_over_line_profile=False,
                                      treat_line_overlap=True)    

    def test_LVG_cannot_treat_overlap(self):
        for geo,lp,mode,use_ng,average in iterate_allowed_cloud_params():
            if 'LVG' in geo:
                with pytest.raises(ValueError):
                    radiative_transfer.Cloud(
                                          datafilepath=datafilepath['HCl'],
                                          geometry=geo,line_profile_type=lp,
                                          width_v=10*constants.kilo,
                                          iteration_mode=mode,use_NG_acceleration=use_ng,
                                          average_over_line_profile=True,
                                          treat_line_overlap=True)    

    def test_warning_overlapping_lines(self):
        for geo,lp,mode,use_ng,average in iterate_allowed_cloud_params():
            with pytest.warns(UserWarning):
                radiative_transfer.Cloud(
                                      datafilepath=datafilepath['HCl'],
                                      geometry=geo,line_profile_type=lp,
                                      width_v=10*constants.kilo,
                                      iteration_mode=mode,use_NG_acceleration=use_ng,
                                      treat_line_overlap=False)
    
    def test_LVG_rectangular(self):
        for geo in ('LVG sphere','LVG slab'):
            with pytest.raises(ValueError):
                radiative_transfer.Cloud(
                                  datafilepath=datafilepath['CO'],geometry=geo,
                                  line_profile_type='Gaussian',width_v=1*constants.kilo,
                                  iteration_mode='ALI',use_NG_acceleration=True,
                                  average_over_line_profile=True)


class TestSetParameters():

    @staticmethod
    def verify_cloud_params(cloud,params):
        for p in ('N','collider_densities','Tkin'):
            assert getattr(cloud,p) == params[p]
        nu0s = np.array([line.nu0 for line in cloud.emitting_molecule.rad_transitions])
        I_ext_nu0 = np.array([params['ext_background'](nu0) for nu0 in nu0s])
        assert np.all(I_ext_nu0==cloud.I_ext_nu0)
        for dust_param in ('T_dust','tau_dust'):
            if params[dust_param] == None:
                assert np.all(getattr(cloud,dust_param)(nu0s)==0)
            else:
                T_dust_nu0 = [params[dust_param](nu0) for nu0 in nu0s]
                assert np.all(T_dust_nu0==getattr(cloud,dust_param)(nu0s))

    def test_set_parameters(self):
        collider_densities = {'para-H2':1}
        ext_background = helpers.generate_CMB_background()
        N = 1e15/constants.centi**2
        standard_params = {'ext_background':ext_background,
                           'Tkin':20,'collider_densities':collider_densities,
                           'N':N,'T_dust':None,'tau_dust':None}
        for cloud in general_cloud_iterator(specie='CO',width_v=1*constants.kilo):
            for i in range(2):
                #two times the because the first time is special (initial setting of params)
                cloud.set_parameters(**standard_params)
                self.verify_cloud_params(cloud,standard_params)
            new_params = {'ext_background':lambda nu: ext_background(nu)/2,'N':4*N,
                          'Tkin':4*standard_params['Tkin']}
            for param_name,new_value in new_params.items():
                changed_params = standard_params.copy()
                changed_params[param_name] = new_value
                cloud.set_parameters(**changed_params)
                self.verify_cloud_params(cloud,changed_params)
            #T_dust and tau_dust have to be changed at the same time
            #(only on of them None is not allowed)
            if 'LVG' not in cloud.geometry_name:
                changed_params = standard_params.copy()
                changed_params['T_dust'] = lambda nu: np.ones_like(nu)*145
                changed_params['tau_dust'] = lambda nu: np.ones_like(nu)*0.5
                cloud.set_parameters(**changed_params)
                self.verify_cloud_params(cloud,changed_params)
            new_collider_densities\
                   = [standard_params['collider_densities'] | {'ortho-H2':200},
                      {'para-H2':standard_params['collider_densities']['para-H2']*2},
                      {'ortho-H2':300}]
            for new_coll_densities in new_collider_densities:
                changed_params = standard_params.copy()
                changed_params['collider_densities'] = new_coll_densities
                cloud.set_parameters(**changed_params)
                self.verify_cloud_params(cloud,changed_params)
                #test also change of colliders and Tkin at the same time:
                changed_params = standard_params.copy()
                changed_params['collider_densities'] = new_coll_densities
                changed_params['Tkin'] = 2*standard_params['Tkin']
                cloud.set_parameters(**changed_params)
                self.verify_cloud_params(cloud,changed_params)

    def test_invalid_collider_densities(self):
        test_cloud = get_general_test_cloud(specie='CO')
        invalid_coll_densities = {'e':1/constants.centi**3} #CO has not electron data
        with pytest.raises(ValueError):
            test_cloud.set_parameters(ext_background=helpers.zero_background,
                                      N=1e12/constants.centi*2,Tkin=20,
                                      collider_densities=invalid_coll_densities)

    def test_invalid_dust_params(self):
        test_cloud = get_general_test_cloud(specie='CO')
        def identity(x): return x
        invalid_dust_params = [{'T_dust':None,'tau_dust':identity},
                               {'T_dust':identity,'tau_dust':None}]
        for invalid_params in invalid_dust_params:
            with pytest.raises(ValueError):
                test_cloud.set_parameters(ext_background=helpers.zero_background,
                                          N=1e13/constants.centi*2,Tkin=20,
                                          collider_densities={'para-H2':1/constants.centi**3},
                                          **invalid_params)

    def test_dust_not_allowed_for_LVG(self):
        tau_dust = lambda nu: np.ones_like(nu)*1
        T_dust = lambda nu: np.ones_like(nu)*100
        for geometry in radiative_transfer.Cloud.geometries.keys():
            if 'LVG' in geometry:
                cloud = radiative_transfer.Cloud(
                                      datafilepath=datafilepath['CO'],geometry=geometry,
                                      line_profile_type='rectangular',
                                      width_v=1*constants.kilo,
                                      iteration_mode='ALI')
                with pytest.raises(ValueError):
                    cloud.set_parameters(ext_background=helpers.generate_CMB_background(),
                                         N=1e12/constants.centi**2,Tkin=20,
                                         collider_densities={'para-H2':1/constants.centi**3},
                                         tau_dust=tau_dust,T_dust=T_dust)

    @staticmethod
    def slow_ext_background(nu):
        return helpers.generate_CMB_background(z=1)(nu)

    @staticmethod
    def slow_T_dust(nu):
        return 100*(nu/(230*constants.giga))**-0.5

    @staticmethod
    def slow_tau_dust(nu):
        return 1*(nu/(230*constants.giga))

    @staticmethod
    def fast(nu):
        return np.sin(nu) 

    def test_slow_variation_check(self):
        for cloud in general_cloud_iterator(specie='CO',width_v=1*constants.kilo):
            assert not cloud.is_slowly_varying_over_linewidth(self.fast)
            for slow in (self.slow_T_dust,self.slow_ext_background,
                         self.slow_tau_dust):
                assert cloud.is_slowly_varying_over_linewidth(slow)
    
    def test_slow_variation_check_at_initialisation(self):
        cloud = radiative_transfer.Cloud(
                              datafilepath=datafilepath['CO'],geometry='uniform sphere',
                              line_profile_type='Gaussian',width_v=1*constants.kilo,
                              iteration_mode='ALI',average_over_line_profile=False)
        std_params = {'N':1e13/constants.centi**2,'Tkin':25,
                      'collider_densities':{'para-H2':1},
                      'ext_background':self.slow_ext_background,
                      'T_dust':self.slow_T_dust,'tau_dust':self.slow_tau_dust}
        #this should not throw an error:
        cloud.set_parameters(**std_params)
        for attr_name in ('ext_background','T_dust','tau_dust'):
            params = std_params.copy()
            params[attr_name] = self.fast
            with pytest.warns(UserWarning):
                cloud.set_parameters(**params)

    @pytest.mark.filterwarnings("ignore:negative optical depth")
    def test_set_parameters_with_physics(self):
        N_values = np.logspace(12,16,4)/constants.centi**2
        ext_background = helpers.generate_CMB_background()
        ext_backgrounds = [ext_background,lambda nu: 0,lambda nu: ext_background(nu)/2]
        Tkins = np.linspace(20,200,4)
        coll_density_values = np.logspace(2,7,4)/constants.centi**3
        collider_cases = [['ortho-H2'],['para-H2'],['ortho-H2','para-H2']]
        T_dust_cases = [lambda nu: np.zeros_like(nu),lambda nu: np.ones_like(nu)*250]
        tau_dust_cases = [lambda nu: np.zeros_like(nu),lambda nu: np.ones_like(nu)*4]
        def param_iterator():
            return itertools.product(N_values,ext_backgrounds,Tkins,coll_density_values,
                                     collider_cases,T_dust_cases,tau_dust_cases)
        cloud_kwargs = {'datafilepath':datafilepath['CO'],'geometry':'uniform sphere',
                        'line_profile_type':'rectangular','width_v':1*constants.kilo,
                        'iteration_mode':'ALI','use_NG_acceleration':True,
                        'average_over_line_profile':False}
        cloud_to_modify = radiative_transfer.Cloud(**cloud_kwargs)
        def generate_cloud_and_set_params(params):
            cloud = radiative_transfer.Cloud(**cloud_kwargs)
            cloud.set_parameters(**params)
            return cloud
        for N,ext_b,Tkin,coll_dens,colliders,T_dust,tau_dust in param_iterator():
            params = {'N':N,'Tkin':Tkin,'ext_background':ext_b,'T_dust':T_dust,
                      'tau_dust':tau_dust}
            #put different values in case more than one collider (not really necessary...)
            coll_densities = {collider:coll_dens*(i+1) for i,collider in
                              enumerate(colliders)}
            params['collider_densities'] = coll_densities
            cloud_to_modify.set_parameters(**params)
            reference_cloud = generate_cloud_and_set_params(params=params)
            cloud_to_modify.solve_radiative_transfer()
            reference_cloud.solve_radiative_transfer()
            assert np.all(cloud_to_modify.level_pop==reference_cloud.level_pop)


class Test_rate_equation_quantities():
    #everything evaluated at nu0; no treatment of overlapping lines, but dust is ok

    def get_S(self,line,x1,x2):
        if x1==x2==0:
            return 0
        else:
            return line.A21*x2/(x1*line.B12-x2*line.B21)

    def expected_line_data_nu0(self,cloud,level_pop):
        lines = cloud.emitting_molecule.rad_transitions
        n_low = np.array([line.low.number for line in lines])
        n_up = np.array([line.up.number for line in lines])
        x1 = np.array([level_pop[n] for n in n_low])
        x2 = np.array([level_pop[n] for n in n_up])
        N1 = cloud.N*x1
        N2 = cloud.N*x2
        nu0 = cloud.emitting_molecule.nu0
        tau_nu0_line = np.array([line.tau_nu(N1=Nlow,N2=Nup,nu=nu) for Nlow,Nup,nu,line
                                 in zip(N1,N2,nu0,lines)])
        tau_nu0_line_and_dust = tau_nu0_line + cloud.tau_dust(nu0)
        beta_nu0_line_and_dust = cloud.geometry.beta(tau_nu0_line_and_dust)
        Iext_nu0 = cloud.ext_background(nu0)
        S_nu0_line = []
        for line,xlow,xup in zip(lines,x1,x2):
            S_nu0_line.append(self.get_S(line=line,x1=xlow,x2=xup))
        S_nu0_line = np.array(S_nu0_line)
        S_nu0_line_and_dust = (tau_nu0_line*S_nu0_line + cloud.tau_dust(nu0)*cloud.S_dust(nu0))
        S_nu0_line_and_dust = np.where(tau_nu0_line_and_dust!=0,
                                       S_nu0_line_and_dust/tau_nu0_line_and_dust,0)
        return {'tau_nu0_line':tau_nu0_line,'tau_nu0_line_and_dust':tau_nu0_line_and_dust,
               'beta_nu0_line_and_dust':beta_nu0_line_and_dust,
               'Iext_nu0':Iext_nu0,'S_nu0_line_and_dust':S_nu0_line_and_dust}

    def expected_line_data_avg(self,cloud,level_pop):
        lines = cloud.emitting_molecule.rad_transitions
        n_low = np.array([line.low.number for line in lines])
        n_up = np.array([line.up.number for line in lines])
        x1 = np.array([level_pop[n] for n in n_low])
        x2 = np.array([level_pop[n] for n in n_up])
        N1 = cloud.N*x1
        N2 = cloud.N*x2
        nu0 = cloud.emitting_molecule.nu0
        nu_arrays = []
        tau_line_nu = []
        tau_line_dust_nu = []
        tau_line_dust_overlap_nu = []
        beta_line_dust_nu = []
        beta_line_dust_overlap_nu = []
        Iext_nu = []
        phi_nu = []
        S_line_dust_nu = []
        S_line_dust_overlap_nu = []
        K_dust_nu = []
        K_dust_overlap_nu = []
        for i,line in enumerate(cloud.emitting_molecule.rad_transitions):
            width_nu = line.line_profile.width_nu
            if cloud.emitting_molecule.line_profile_type == 'rectangular':
                nu = np.linspace(nu0[i]-width_nu,nu0[i]+width_nu,600)
            else:
                nu = np.linspace(nu0[i]-width_nu*2.5,nu0[i]+width_nu*2.5,700)
            nu_arrays.append(nu)
            tau_line = line.tau_nu(N1=N1[i],N2=N2[i],nu=nu)
            tau_line_nu.append(tau_line)
            tau_dust = cloud.tau_dust(nu)
            tau_line_dust = tau_line + tau_dust
            tau_line_dust_nu.append(tau_line_dust)
            tau_line_dust_overlap = tau_line_dust.copy()
            for j in cloud.emitting_molecule.overlapping_lines[i]:
                overlapping_line = cloud.emitting_molecule.rad_transitions[j]
                tau_line_dust_overlap += overlapping_line.tau_nu(N1=N1[j],N2=N2[j],nu=nu)
            tau_line_dust_overlap_nu.append(tau_line_dust_overlap)
            beta_line_dust_nu.append(cloud.geometry.beta(tau_line_dust))
            beta_line_dust_overlap_nu.append(cloud.geometry.beta(tau_line_dust_overlap))
            Iext_nu.append(cloud.ext_background(nu))
            phi_nu.append(line.line_profile.phi_nu(nu))
            S_line = self.get_S(line=line,x1=x1[i],x2=x2[i])
            S_dust = cloud.S_dust(nu)
            S_line_dust = np.where(tau_line_dust!=0,
                                   (S_line*tau_line+S_dust*tau_dust)/tau_line_dust,
                                   0)
            S_line_dust_nu.append(S_line_dust)
            S_line_dust_overlap = S_line*tau_line+S_dust*tau_dust
            K_dust = np.where(tau_line_dust!=0,S_dust*tau_dust/tau_line_dust,0)
            K_dust_nu.append(K_dust)
            K_dust_overlap = S_dust*tau_dust
            for j in cloud.emitting_molecule.overlapping_lines[i]:
                overlapping_line = cloud.emitting_molecule.rad_transitions[j]
                S_ol = self.get_S(line=overlapping_line,x1=x1[j],x2=x2[j])
                tau_ol = overlapping_line.tau_nu(N1=N1[j],N2=N2[j],nu=nu)
                S_line_dust_overlap += tau_ol*S_ol
                K_dust_overlap += tau_ol*S_ol
            S_line_dust_overlap = np.where(tau_line_dust_overlap!=0,
                                           S_line_dust_overlap/tau_line_dust_overlap,0)
            S_line_dust_overlap_nu.append(S_line_dust_overlap)
            K_dust_overlap = np.where(tau_line_dust_overlap!=0,
                                      K_dust_overlap/tau_line_dust_overlap,0)
            K_dust_overlap_nu.append(K_dust_overlap)
        return {'nu':nu_arrays,'Iext_nu':Iext_nu,'beta_line_dust_nu':beta_line_dust_nu,
                'beta_line_dust_overlap_nu':beta_line_dust_overlap_nu,
                'S_line_dust_nu':S_line_dust_nu,
                'S_line_dust_overlap_nu':S_line_dust_overlap_nu,
                'tau_line_nu':tau_line_nu,'tau_line_dust_nu':tau_line_dust_nu,
                'tau_line_dust_overlap_nu':tau_line_dust_overlap_nu,
                'K_dust_nu':K_dust_nu,'K_dust_overlap_nu':K_dust_overlap_nu}

    def cloud_iterator(self,average_over_line_profile,treat_overlap_options):
        T = 45
        width_v = 30*constants.kilo
        T_dust_cases = [None,lambda nu: (nu/(100*constants.giga))**0.5*100]
        tau_dust_cases = [None,lambda nu: (nu/(100*constants.giga))**-0.3]
        for geo,lp,specie,T_dust,tau_dust in\
                       itertools.product(geometries,line_profile_types,('CO','HCl'),
                                         T_dust_cases,tau_dust_cases):
            if not allowed_geo_lp_combination(geometry=geo,line_profile_type=lp):
                continue
            if 'LVG' in geo and not (T_dust is None and tau_dust is None):
                continue
            if T_dust is None and tau_dust is not None:
                continue
            if T_dust is not None and tau_dust is None:
                continue
            for ovl in treat_overlap_options:
                if ovl and 'LVG' in geo:
                    continue
                cloud = radiative_transfer.Cloud(
                                    datafilepath=datafilepath[specie],geometry=geo,
                                    line_profile_type=lp,width_v=width_v,iteration_mode='ALI',
                                    use_NG_acceleration=True,
                                    average_over_line_profile=average_over_line_profile,
                                    treat_line_overlap=ovl)
                cloud.set_parameters(ext_background=helpers.generate_CMB_background(),
                                     N=1e12/constants.centi**2,Tkin=T,
                                     collider_densities={'para-H2':1e4/constants.centi**3},
                                     T_dust=T_dust,tau_dust=tau_dust)
                LTE_level_pop = cloud.emitting_molecule.LTE_level_pop(T)
                zero_level_pop = np.zeros_like(LTE_level_pop)
                zero_level_pop[0] = 1
                for level_pop in (LTE_level_pop,zero_level_pop):
                    yield cloud,level_pop

    def cloud_iterator_nu0(self):
        for cloud,level_pop in self.cloud_iterator(average_over_line_profile=False,
                                                   treat_overlap_options=[False,]):
            yield cloud,level_pop,self.expected_line_data_nu0(cloud=cloud,level_pop=level_pop)

    def cloud_iterator_averaging(self):
        for cloud,level_pop in self.cloud_iterator(average_over_line_profile=True,
                                                   treat_overlap_options=[True,False]):
            yield cloud,level_pop,self.expected_line_data_avg(cloud=cloud,level_pop=level_pop)

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_tau_nu0(self):
        for cloud,level_pop,expected_line_data in self.cloud_iterator_nu0():
            calculated_tau_nu0 = cloud.tau_nu0_onlyline(level_population=level_pop)
            assert np.all(expected_line_data['tau_nu0_line']==calculated_tau_nu0)
            calculated_tau_nu0_with_dust = cloud.tau_nu0_including_dust(
                                               level_population=level_pop)
            assert np.all(expected_line_data['tau_nu0_line_and_dust']
                          ==calculated_tau_nu0_with_dust)

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_beta_nu0(self):
        for cloud,level_pop,expected_line_data in self.cloud_iterator_nu0():
            calculated_beta = cloud.beta_nu0(level_population=level_pop)
            expected_beta = expected_line_data['beta_nu0_line_and_dust']
            assert np.all(expected_beta==calculated_beta)

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_S_nu0(self):
        for cloud,level_pop,expected_line_data in self.cloud_iterator_nu0():
            calculated_S = cloud.S_nu0(level_population=level_pop)
            assert np.allclose(expected_line_data['S_nu0_line_and_dust'],
                               calculated_S,atol=0,rtol=1e-6)

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_Jbar_nu0(self):
        for cloud,level_pop,expected_line_data in self.cloud_iterator_nu0():
            calculated_J = cloud.Jbar_nu0(level_population=level_pop)
            beta = expected_line_data['beta_nu0_line_and_dust']
            Iext = expected_line_data['Iext_nu0']
            S = expected_line_data['S_nu0_line_and_dust']
            assert np.allclose(calculated_J,beta*Iext+(1-beta)*S,atol=0,rtol=1e-6)

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_A21_factor_nu0(self):
        for cloud,level_pop,expected_line_data in self.cloud_iterator_nu0():
            calc_factor = cloud.A21_factor_nu0(level_population=level_pop)
            beta = expected_line_data['beta_nu0_line_and_dust']
            tau_line = expected_line_data['tau_nu0_line']
            tau_tot = expected_line_data['tau_nu0_line_and_dust']
            A21 = np.where(tau_tot!=0,1-(1-beta)*tau_line/tau_tot,1)
            assert np.all(A21==calc_factor)

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_B21_factor_nu0(self):
        for cloud,level_pop,expected_line_data in self.cloud_iterator_nu0():
            calc_factor = cloud.B21_factor_nu0(level_population=level_pop)
            beta = expected_line_data['beta_nu0_line_and_dust']
            Iext = expected_line_data['Iext_nu0']
            S_dust = cloud.S_dust(cloud.emitting_molecule.nu0)
            tau_dust = cloud.tau_dust_nu0
            tau_tot = expected_line_data['tau_nu0_line_and_dust']
            K = np.where(tau_tot!=0,tau_dust*S_dust/tau_tot,0)
            B21 = beta*Iext+(1-beta)*K
            assert np.all(B21==calc_factor)

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_A21_factor_averaged(self):
        for cloud,level_pop,expected_line_data in self.cloud_iterator_averaging():
            A21_factor = []
            for i,line in enumerate(cloud.emitting_molecule.rad_transitions):
                if cloud.treat_line_overlap:
                    beta_nu = expected_line_data['beta_line_dust_overlap_nu'][i]
                    tau_tot_nu = expected_line_data['tau_line_dust_overlap_nu'][i]
                else:
                    beta_nu = expected_line_data['beta_line_dust_nu'][i]
                    tau_tot_nu = expected_line_data['tau_line_dust_nu'][i]
                nu_i = expected_line_data['nu'][i]
                tau_line_nu = expected_line_data['tau_line_nu'][i]
                def A21_factor_func(nu):
                    tau_line = np.interp(nu,nu_i,tau_line_nu)
                    tau_tot = np.interp(nu,nu_i,tau_tot_nu)
                    beta = np.interp(nu,nu_i,beta_nu)
                    return np.where(tau_tot!=0,1-(1-beta)*tau_line/tau_tot,1)
                A21_factor.append(line.line_profile.average_over_phi_nu(A21_factor_func))
            calculated_A21_factor = cloud.A21_factor_averaged(level_population=level_pop)
            assert np.allclose(A21_factor,calculated_A21_factor,atol=0,rtol=1e-2)

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_Jbar_averaged(self):
        for cloud,level_pop,expected_line_data in self.cloud_iterator_averaging():
            Jbar = []
            for i,line in enumerate(cloud.emitting_molecule.rad_transitions):
                if cloud.treat_line_overlap:
                    beta = expected_line_data['beta_line_dust_overlap_nu'][i]
                    S = expected_line_data['S_line_dust_overlap_nu'][i]
                else:
                    beta = expected_line_data['beta_line_dust_nu'][i]
                    S = expected_line_data['S_line_dust_nu'][i]
                Iext = expected_line_data['Iext_nu'][i]
                nu_i = expected_line_data['nu'][i]
                def Jbar_func(nu):
                    return np.interp(nu,nu_i,beta*Iext+(1-beta)*S)
                Jbar.append(line.line_profile.average_over_phi_nu(Jbar_func))
            calculated_Jbar = cloud.Jbar_averaged(level_population=level_pop)
            assert np.allclose(Jbar,calculated_Jbar,atol=0,rtol=2e-2)

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_B21_factor_averaged(self):
        for cloud,level_pop,expected_line_data in self.cloud_iterator_averaging():
            B21_factor = []
            for i,line in enumerate(cloud.emitting_molecule.rad_transitions):
                if cloud.treat_line_overlap:
                    beta_nu = expected_line_data['beta_line_dust_overlap_nu'][i]
                    K_nu = expected_line_data['K_dust_overlap_nu'][i]
                else:
                    beta_nu = expected_line_data['beta_line_dust_nu'][i]
                    K_nu = expected_line_data['K_dust_nu'][i]
                Iext = expected_line_data['Iext_nu'][i]
                nu_i = expected_line_data['nu'][i]
                def B21_factor_func(nu):
                    return np.interp(nu,nu_i,beta_nu*Iext+(1-beta_nu)*K_nu)
                B21_factor.append(line.line_profile.average_over_phi_nu(B21_factor_func))
            calculated_B21_factor = cloud.B21_factor_averaged(level_population=level_pop)
            assert np.allclose(B21_factor,calculated_B21_factor,atol=0,rtol=2e-2)


@pytest.mark.filterwarnings("ignore:negative optical depth")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_ng_acceleration_and_iterationmode():
    N_values = 10**np.array((12,14,16))/constants.centi**2
    Tkin_values = np.array((40,400))
    coll_dens_values = np.array((1e3,1e6))/constants.centi**3
    ext_backgrounds = [helpers.zero_background,helpers.generate_CMB_background()]
    for geo,lp,average in itertools.product(geometries,line_profile_types,[True,False]):
        if not allowed_geo_lp_combination(geometry=geo,line_profile_type=lp):
            continue
        cloud_kwargs = {'datafilepath':datafilepath['CO'],'geometry':geo,
                        'line_profile_type':lp,'width_v':1*constants.kilo,
                        'average_over_line_profile':average}
        for N,Tkin,coll_dens,ext_background in\
               itertools.product(N_values,Tkin_values,coll_dens_values,ext_backgrounds):
            params = {'N':N,'Tkin':Tkin,'ext_background':ext_background,
                      'collider_densities':{'para-H2':coll_dens}}
            level_pops = []
            for ng,iter_mode in itertools.product((True,False),iteration_modes):
                cloud = radiative_transfer.Cloud(use_NG_acceleration=ng,
                                                 iteration_mode=iter_mode,**cloud_kwargs)
                cloud.set_parameters(**params)
                cloud.solve_radiative_transfer()
                level_pops.append(cloud.level_pop)
            for lp1,lp2 in itertools.pairwise(level_pops):
                np.allclose(lp1,lp2,atol=0,rtol=1e-3)

def test_compute_residual():
    min_tau = radiative_transfer.Cloud.min_tau_considered_for_convergence
    small_tau = np.array((0.7*min_tau,min_tau/2,min_tau/100,min_tau/500,min_tau/1.01))
    tau = np.array((1,10,2*min_tau,min_tau,min_tau/2))
    n_relevant_taus = (tau>min_tau).sum()
    Tex_residual = np.array((1,2,3,4,5))
    assert radiative_transfer.Cloud.compute_residual(
                                          Tex_residual=Tex_residual,tau=small_tau,
                                          min_tau_considered_for_convergence=min_tau) == 0
    expected_residual = np.sum(Tex_residual[tau>min_tau])/n_relevant_taus
    assert radiative_transfer.Cloud.compute_residual(
                                          Tex_residual=Tex_residual,tau=tau,
                                          min_tau_considered_for_convergence=min_tau)\
            == expected_residual

def test_LTE_radiative_transfer():
    Tkin = 150
    N_small = 1e11/constants.centi**2
    N_medium = 1e15/constants.centi**2
    N_large = 1e18/constants.centi**2
    collider_density_small = {'ortho-H2':1/constants.centi**3}
    collider_density_large = {'ortho-H2':1e11/constants.centi**3}
    LTE_background = lambda nu: helpers.B_nu(nu=nu,T=Tkin)
    ext_background = helpers.generate_CMB_background()
    for geo,lp,use_ng,average in itertools.product(geometries,line_profile_types,
                                                   use_ng_options,average_options):
        if not allowed_geo_lp_combination(geometry=geo,line_profile_type=lp):
            continue
        cloud = radiative_transfer.Cloud(
                      datafilepath=datafilepath['CO'],geometry=geo,
                      line_profile_type=lp,width_v=3*constants.kilo,iteration_mode='ALI',
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

def test_LI_vs_ALI():
    raise NotImplementedError


class TestModelGrid():
    
    cloud = radiative_transfer.Cloud(
                          datafilepath=datafilepath['CO'],geometry='uniform sphere',
                          line_profile_type='rectangular',width_v=1.4*constants.kilo,
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
    requested_output = ['level_pop','Tex','tau_nu0_individual_transitions',
                        'fluxes_of_individual_transitions','tau_nu','spectrum']
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
        for request in ('fluxes_of_individual_transitions','spectrum'):
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
    @pytest.mark.filterwarnings("ignore:invalid value encountered")
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
                    for n_ortho,n_para in\
                                itertools.product(self.collider_densities_values['ortho-H2'],
                                                  self.collider_densities_values['para-H2']):
                        collider_densities = {'ortho-H2':n_ortho,'para-H2':n_para}
                        self.cloud.set_parameters(
                              ext_background=ext_background,N=N,Tkin=Tkin,
                              collider_densities=collider_densities)
                        self.cloud.solve_radiative_transfer()
                        fluxes = self.cloud.fluxes_of_individual_transitions(
                                             solid_angle=self.solid_angle,
                                             transitions=self.transitions)
                        tau_nu = self.cloud.tau_nu(nu=self.nu)
                        spectrum = self.cloud.spectrum(
                                            solid_angle=self.solid_angle,nu=self.nu)
                        matching_models = [m for m in models if
                                           m['ext_background']==backgroundID
                                           and m['N']==N and m['Tkin']==Tkin
                                           and m['collider_densities']==collider_densities]
                        assert len(matching_models) == 1
                        matching_model = matching_models[0]
                        assert np.all(matching_model['level_pop'] == self.cloud.level_pop)
                        assert np.all(matching_model['Tex']
                                      == self.cloud.Tex[self.transitions])
                        assert np.all(matching_model['tau_nu0_individual_transitions']
                                      == self.cloud.tau_nu0_individual_transitions[self.transitions])
                        assert np.all(matching_model['fluxes_of_individual_transitions']
                                      == fluxes)
                        assert np.all(matching_model['tau_nu'] == tau_nu)
                        assert np.all(matching_model['spectrum']==spectrum)
                        n_check_models += 1
        assert n_check_models == len(models)


def test_print_results():
    for cloud in general_cloud_iterator(specie='CO',width_v=1*constants.kilo):
        cloud.set_parameters(ext_background=helpers.generate_CMB_background(),
                              N=1e14/constants.centi**2,Tkin=33.33,
                              collider_densities={'ortho-H2':1e3/constants.centi**3})
        cloud.solve_radiative_transfer()
        cloud.print_results()