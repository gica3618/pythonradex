# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:09:51 2017

@author: gianni
"""

from scipy import constants
from pythonradex import radiative_transfer,helpers
import os
import itertools
import pytest
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
datafolder = os.path.join(here,'LAMDA_files')
N_values = {'co':np.array((1e12,1e15,1e18))/constants.centi**2,
               'hcl':np.array((1e10,1e12,1e14))/constants.centi**2,
               'ocs@xpol':np.array((1e10,1e13,1e15))/constants.centi**2,
               'c':np.array((1e12,1e15,1e20))/constants.centi**2}
collider_densities = {'co':{'ortho-H2':1e9/constants.centi**3},
                      'hcl':{'ortho-H2':1e11/constants.centi**3},
                      'ocs@xpol':{'H2':1e9/constants.centi**3},
                      'c':{'e':1e6/constants.centi**3}}
#for HCL, some levels have relativel low LTE population (~1e-3), but pythonradex gives 0
#so I have to restrict to only the most highest populated levels for HCL
min_level_pop = {'co':0,'hcl':1e-2,'ocs@xpol':0,'c':0}
Tkin = 101
width_v = 2*constants.kilo
ext_background = helpers.generate_CMB_background()

filenames = ['co.dat','hcl.dat','ocs@xpol.dat','c.dat']
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

#HCl has overlapping lines, let's filter the warning
@pytest.mark.filterwarnings("ignore:lines of input molecule are overlapping")
def test_LTE():
    max_taus = []
    for filename,geo,lp,iter_mode,ng,avg in itertools.product(
                        filenames,geometries,line_profile_types,iteration_modes,
                        use_ng_options,average_options):
        if not allowed_param_combination(geometry=geo,line_profile_type=lp):
            continue
        specie = filename.split('.')[0]
        datafilepath = os.path.join(datafolder,filename)
        cloud = radiative_transfer.Cloud(
                            datafilepath=datafilepath,geometry=geo,
                            line_profile_type=lp,width_v=width_v,iteration_mode=iter_mode,
                            use_NG_acceleration=ng,average_over_line_profile=avg)
        cloud_params = {'Tkin':Tkin,'ext_background':ext_background,
                        'collider_densities':collider_densities[specie]}
        for N in N_values[specie]:
            cloud_params['N'] = N
            cloud.set_parameters(**cloud_params)
            cloud.solve_radiative_transfer()
            LTE_level_pop = cloud.emitting_molecule.LTE_level_pop(T=Tkin)
            selection = LTE_level_pop > min_level_pop[specie]*np.max(LTE_level_pop)
            assert np.allclose(cloud.level_pop[selection],LTE_level_pop[selection],
                               atol=1e-6,rtol=1e-2)
            max_taus.append(np.max(cloud.tau_nu0))
    print(f'max tau: {np.max(max_taus)}')