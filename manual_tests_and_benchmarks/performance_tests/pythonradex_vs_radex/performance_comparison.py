#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:50:22 2025

@author: gianni
"""

#this script is essentially a simplified version of radex_vs_pythonradex_performance.py
#IMPORTANT: if you want to delete the numba compilation (__pycache__), you need to
#run this in an environment where pythonradex is installed in editable mode, e.g.
#the virtual environment virt_env_for_testing

import sys
sys.path.append('../..')
import general
from pythonradex import radiative_transfer,helpers
from scipy import constants
import os
import numpy as np
import time
import itertools
import mini_radex_wrapper
import shutil

ext_background = helpers.generate_CMB_background(z=0)


#can only use LVG slab and uniform sphere, since these are the only two
#where RADEX and pythonradex use same escape probability
geometry = 'uniform sphere'
# geometry = 'LVG slab'

n_grid_elements = 20

grids = {"CO":{'data_filename':'co.dat','colliders':['para-H2','ortho-H2'],
               "log_N_limits" :(13+4,18+4),"Tmin":20,'Tmax':250},
         #narrow range where RADEX does not throw warnings:
         "CO narrow range":{'data_filename':'co.dat','colliders':['para-H2','ortho-H2'],
                            "log_N_limits" :(16+4,16.3+4),"Tmin":50,'Tmax':51},
         "HCO+":{'data_filename':'hco+.dat','colliders':['H2',],
                 "log_N_limits" :(10+4,14+4),"Tmin":20,'Tmax':250},
         "SO":{'data_filename':'so@lique.dat','colliders':['H2',],
               "log_N_limits" :(10+4,12+4),"Tmin":60,'Tmax':250},
         "C":{'data_filename':'c.dat','colliders':['para-H2','ortho-H2'],
              "log_N_limits" :(12+4,18+4),"Tmin":60,'Tmax':250}
         }

#actually, RADEX assumes rectangular, but then converts it Gaussian for the line flux
line_profile_type = 'rectangular'
width_v = 1*constants.kilo

radex_input_file = 'radex_test_preformance.inp'
radex_output_file = 'radex_test_preformance.out'

def remove_pythonradex_cache():
    src_folder = '../../../src/pythonradex'
    assert os.path.exists(src_folder)
    cache_folder = os.path.join(src_folder,'__pycache__')
    if os.path.exists(cache_folder):
        print(f'removing python cache ({cache_folder})')
        shutil.rmtree(cache_folder)

pythonradex_times = {}
radex_times = {}

for ID,grid in grids.items():
    print(f'now running following grid: {ID}')
    # #ATTENTION: if no H2 is given, RADEX just puts 1e5 cm-3 by default! WTF!
    #RADEX does other strange things with the H2 density:
    #see line 112 in io.f, line 168 in io.f, and line 225 in readdata.f
    for collider in grid['colliders']:
        assert 'H2' in collider

    datafilepath = os.path.join(general.lamda_data_folder,grid['data_filename'])
    N_values = np.logspace(*grid['log_N_limits'],n_grid_elements)
    coll_density_values = np.logspace(3,5,n_grid_elements)/constants.centi**3
    Tkin_values = np.linspace(grid['Tmin'],grid['Tmax'],n_grid_elements)

    remove_pythonradex_cache()
    print('running pythonradex')
    start = time.time()
    cloud = radiative_transfer.Cloud(
                datafilepath=datafilepath,geometry=geometry,
                line_profile_type=line_profile_type,width_v=width_v,
                use_Ng_acceleration=True,treat_line_overlap=False,
                warn_negative_tau=False)
    #IMPORTANT: put N in innermost loop to improve performance
    for coll_dens,Tkin,N in itertools.product(coll_density_values,Tkin_values,N_values):
        collider_densities = {collider:coll_dens for collider in grid['colliders']}
        cloud.update_parameters(ext_background=ext_background,Tkin=Tkin,
                                collider_densities=collider_densities,N=N,T_dust=0,
                                tau_dust=0)
        cloud.solve_radiative_transfer()
    end = time.time()
    pythonradex_time = end-start
    print(f"pythonradex time: {pythonradex_time:.3g}")
    pythonradex_times[ID] = pythonradex_time

    print('Running RADEX')
    start = time.time()
    for N,coll_dens,Tkin in itertools.product(N_values,coll_density_values,
                                              Tkin_values):
        collider_densities = {collider:coll_dens for collider in grid['colliders']}
        # start_wrapper = time.perf_counter()
        wrapper_times = mini_radex_wrapper.run_radex(
                        datafilename=grid['data_filename'],geometry=geometry,
                        collider_densities=collider_densities,Tkin=Tkin,N=N,
                        width_v=width_v,input_filepath=radex_input_file,
                        output_filepath=radex_output_file)
        # end_wrapper = time.perf_counter()
        # print(wrapper_times)
        # print(f"wrapper time: {end_wrapper-start_wrapper}")
    end = time.time()
    RADEX_time = end-start
    print(f"RADEX time: {RADEX_time:.3g}")
    radex_times[ID] = RADEX_time
for ID in radex_times.keys():
    print(f'time ratio pythonradex/RADEX for grid {ID}:'
          +f' {radex_times[ID]/pythonradex_times[ID]:.3g}')