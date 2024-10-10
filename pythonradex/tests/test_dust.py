#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:11:07 2024

@author: gianni
"""


from pythonradex import radiative_transfer,helpers
from scipy import constants
import os
import numpy as np
import itertools
import pytest


tau_dust_values = {'thick':10,'thin':1e-4}
default_T_dust_value = 100
Tkin = 30
default_T_dust = lambda nu: np.ones_like(nu)*default_T_dust_value
geometries = ('uniform sphere','uniform slab')
line_profile_types = ('rectangular','Gaussian')
iteration_modes = ('LI','ALI')

#initially I put the different functions into a dict, but that did not work; each
#function just returned the same value; so it's safer to generate on the fly each time
def generate_tau_dust_func(dust_case):
    def tau_dust(nu):
        return np.ones_like(nu)*tau_dust_values[dust_case]
    return tau_dust

def generate_cloud(datafilename,geometry,line_profile_type,
                   width_v,iteration_mode,treat_line_overlap,N,
                   collider_densities,average_over_line_profile,tau_dust,
                   T_dust):
    here = os.path.dirname(os.path.abspath(__file__))
    datafilepath = os.path.join(here,f'LAMDA_files/{datafilename}')
    cld = radiative_transfer.Cloud(
                          datafilepath=datafilepath,geometry=geometry,
                          line_profile_type=line_profile_type,
                          width_v=width_v,iteration_mode=iteration_mode,
                          use_NG_acceleration=True,
                          average_over_line_profile=average_over_line_profile,
                          treat_line_overlap=treat_line_overlap)
    cld.set_parameters(ext_background=helpers.zero_background,N=N,Tkin=Tkin,
                       collider_densities=collider_densities,T_dust=T_dust,
                       tau_dust=tau_dust)
    cld.solve_radiative_transfer()
    return cld


class TestDust():

    width_v = {'co':1*constants.kilo,'cn':1000*constants.kilo}
    treat_line_overlap = {'co':False,'cn':True}
    datafilenames = {'co':'co.dat','cn':'cn.dat'}

    def cloud_iterator(self,N,collider_densities,tau_dust,T_dust,molecule_name):
        for geo,lp,iter_mode,lp_average in\
              itertools.product(geometries,line_profile_types,iteration_modes,
                                (True,False)):
                  treat_line_overlap = self.treat_line_overlap[molecule_name]
                  if treat_line_overlap and not lp_average:
                      continue
                  yield generate_cloud(datafilename=self.datafilenames[molecule_name],
                                       geometry=geo,line_profile_type=lp,
                                       width_v=self.width_v[molecule_name],
                                       iteration_mode=iter_mode,
                                       treat_line_overlap=treat_line_overlap,
                                       N=N,collider_densities=collider_densities,
                                       average_over_line_profile=lp_average,
                                       tau_dust=tau_dust,T_dust=T_dust)

    def test_thin_dust_thick_gas(self):
        #expect that dust does not have any effect
        gas_params = {'co':{'N':1e16/constants.centi**2,
                            'collider_densities':{'ortho-H2':1e5/constants.centi**3}},
                      'cn':{'N':1e15/constants.centi**2,
                            'collider_densities':{'e':1e3/constants.centi**3}}}
        tau_dust = generate_tau_dust_func('thin')
        for mol_name,params in gas_params.items():
            cloud_iterator_with_dust = self.cloud_iterator(
                                                **params,tau_dust=tau_dust,
                                                T_dust=default_T_dust,
                                                molecule_name=mol_name)
            cloud_iterator_wo_dust = self.cloud_iterator(
                                                **params,tau_dust=None,
                                                T_dust=None,molecule_name=mol_name)
            for dust_cloud,no_dust_cloud in zip(cloud_iterator_with_dust,
                                                cloud_iterator_wo_dust):
                assert np.allclose(dust_cloud.level_pop,no_dust_cloud.level_pop,
                                   atol=1e-4,rtol=1e-2)

    @pytest.mark.filterwarnings("ignore:negative optical depth")
    def test_thick_dust_thin_gas(self):
        #expect LTE at T_dust
        #for this test to pass I need to use a relatively generous atol
        T_dust_func = default_T_dust
        T_dust_value = default_T_dust_value
        tau_dust = generate_tau_dust_func('thick')
        gas_params = {'co':{'N':1e12/constants.centi**2,
                            'collider_densities':{'ortho-H2':1e1/constants.centi**3}},
                      'cn':{'N':1e11/constants.centi**2,
                            'collider_densities':{'e':1e-1/constants.centi**3}}}
        for mol_name,params in gas_params.items():
            cld_iter = self.cloud_iterator(**params,tau_dust=tau_dust,
                                           T_dust=T_dust_func,molecule_name=mol_name)
            for cloud in cld_iter:
                assert cloud.Tkin != T_dust_value,\
                            'if Tkin=Tdust, cannot say if LTE is caused by gas or dust'
                expected_level_pop = cloud.emitting_molecule.LTE_level_pop(
                                                              T=T_dust_value)
                assert np.allclose(cloud.level_pop,expected_level_pop,atol=5e-2)

    def test_zero_tau_dust(self):
        #calculation with tau_dust=0 should give the same result as when tau_dust
        #is left at the default value (i.e. None)
        gas_params = {'co':{'N':1e15/constants.centi**2,
                            'collider_densities':{'ortho-H2':1e4/constants.centi**3}},
                      'cn':{'N':1e14/constants.centi**2,
                            'collider_densities':{'e':1e2/constants.centi**3}}}
        def zero_tau_dust(nu):
            return np.zeros_like(nu)
        for mol_name,params in gas_params.items():
            cld_iter_zero_dust = self.cloud_iterator(**params,tau_dust=zero_tau_dust,
                                                     T_dust=default_T_dust,
                                                     molecule_name=mol_name)
            cld_iter_no_dust = self.cloud_iterator(**params,tau_dust=None,
                                                   T_dust=None,molecule_name=mol_name)
            for cld_0dust,cld_nodust in zip(cld_iter_zero_dust,cld_iter_no_dust):
                assert np.all(cld_0dust.level_pop==cld_nodust.level_pop)