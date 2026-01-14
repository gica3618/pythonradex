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


#can only use LVG slab and static sphere, since these are the only two
#where RADEX and pythonradex use same escape probability
geometries = ['static sphere', 'LVG slab']

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

for grid_name,grid in grids.items():
    print(f'now running following grid: {grid_name}')
    # #ATTENTION: if no H2 is given, RADEX just puts 1e5 cm-3 by default! WTF!
    #RADEX does other strange things with the H2 density:
    #see line 112 in io.f, line 168 in io.f, and line 225 in readdata.f
    for collider in grid['colliders']:
        assert 'H2' in collider

    datafilepath = os.path.join(general.lamda_data_folder,grid['data_filename'])
    N_values = np.logspace(*grid['log_N_limits'],n_grid_elements)
    coll_density_values = np.logspace(3,5,n_grid_elements)/constants.centi**3
    Tkin_values = np.linspace(grid['Tmin'],grid['Tmax'],n_grid_elements)
    
    pythonradex_times[grid_name] = {}
    for geometry in geometries:
        #actually, RADEX assumes rectangular, but then converts it to Gaussian
        #for the line flux; let's just test both...
        for line_profile_type in ("Gaussian","rectangular"):
            if "LVG" in geometry and line_profile_type == "Gaussian":
                continue
            remove_pythonradex_cache()
            print(f'running pythonradex ({geometry}, {line_profile_type})')
            start = time.time()
            source = radiative_transfer.Source(
                        datafilepath=datafilepath,geometry=geometry,
                        line_profile_type=line_profile_type,width_v=width_v,
                        use_Ng_acceleration=True,treat_line_overlap=False,
                        warn_negative_tau=False)
            #IMPORTANT: put N in innermost loop to improve performance
            for coll_dens,Tkin,N in itertools.product(coll_density_values,Tkin_values,N_values):
                collider_densities = {collider:coll_dens for collider in grid['colliders']}
                source.update_parameters(ext_background=ext_background,Tkin=Tkin,
                                        collider_densities=collider_densities,N=N,T_dust=0,
                                        tau_dust=0)
                source.solve_radiative_transfer()
                source.frequency_integrated_emission(
                           output_type="flux",solid_angle=1,transitions=None)
            end = time.time()
            pythonradex_time = end-start
            print(f"time: {pythonradex_time:.3g}")
            pythonradex_times[grid_name][f"{geometry} {line_profile_type}"]\
                                                        = pythonradex_time

    radex_times[grid_name] = {}
    for geometry in geometries:
        print(f'Running RADEX {geometry}')
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
        radex_times[grid_name][geometry] = RADEX_time

print("\n")
average_times = {}
for grid_name in grids.keys():
    print(grid_name)
    time_collections = {"pythonradex":pythonradex_times[grid_name],
                        "RADEX":radex_times[grid_name]}
    average_times[grid_name] = {code:np.mean(list(tc.values())) for code,tc in
                                time_collections.items()}
    for code,tc in time_collections.items():
        print(code)
        for ID,t in tc.items():
            print(f"{ID}: {t:.3g}")
    print("\n")

print("average times:")
for grid_name,avg_times in average_times.items():
    print(grid_name)
    for code,avg_time in avg_times.items():
        print(f"{code}: {avg_time:.3g}")
    print(f"ratio: {avg_times['RADEX']/avg_times['pythonradex']:.3g}")