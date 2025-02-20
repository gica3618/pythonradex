#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:04:47 2023

@author: gianni
"""

import sys
sys.path.append('..')
import general
from pythonradex import radiative_transfer,helpers
from scipy import constants
import os
import numpy as np
import time
import shutil
import matplotlib.pyplot as plt
import itertools


radex_collider_keys = {'H2':'H2','para-H2':'p-H2','ortho-H2':'o-H2','e':'e',
                       'He':'He'}
ext_background = helpers.generate_CMB_background(z=0)
n_elements = [5,7,10,15,20] #for Tkin, collider and N
#most optimistic case is if only N is varied because then rate equations don't
#need to be re-calculated every time
#however, seems like it doesn't really change anything...
vary_only_N = False


geometry = 'uniform sphere'
radex_executable = '../../tests/Radex/bin/radex_static_sphere'
# geometry = 'LVG slab'
# radex_executable = '../../tests/Radex/bin/radex_LVG_slab'

# data_filename = 'co.dat'
# colliders = ['para-H2','ortho-H2']
# log_N_limits = 13+4,18+4
# Tmin,Tmax = 20,250
#narrow range where RADEX does not throw warnings:
# log_N_limits = 16+4,16.3+4
# Tmin,Tmax = 50,51

# data_filename = 'hco+.dat'
# colliders = ['H2',]
# log_N_limits = 10+4,14+4
# Tmin,Tmax = 20,250

# data_filename = 'so@lique.dat'
# colliders = ['H2',]
# log_N_limits = 10+4,12+4
# Tmin,Tmax = 60,250

data_filename = 'c.dat'
colliders = ['para-H2','ortho-H2']
log_N_limits = 12+4,18+4
Tmin,Tmax = 60,250


# #ATTENTION: if no H2 is given, RADEX just puts 1e5 cm-3 by default! WTF!
#RADEX does other strange things with the H2 density:
#see line 112 in io.f, line 168 in io.f, and line 225 in readdata.f
for collider in colliders:
    assert 'H2' in collider

line_profile_type = 'rectangular' #actually, RADEX assumes rectangular, but than converts it Gaussian for the line flux
width_v = 1*constants.kilo
use_Ng_acceleration = True
treat_line_overlap = False
remove_cache = True


datafilepath = os.path.join(general.lamda_data_folder,data_filename)
radex_input_file = 'radex_test_preformance.inp'

pythonradex_times = np.empty(len(n_elements))
pythonradex_grid_times = np.empty_like(pythonradex_times)
RADEX_times = np.empty_like(pythonradex_times)

def remove_pythonradex_cache():
    cache_folder = '../../src/pythonradex/__pycache__'
    if os.path.exists(cache_folder):
        print(f'removing python cache ({cache_folder})')
        shutil.rmtree(cache_folder)

for i,n in enumerate(n_elements):
    print(f'n elements: {n}')
    if vary_only_N:
        N_values = np.logspace(log_N_limits[0],log_N_limits[1],n**3)
        coll_density_values = [1e4/constants.centi**3,]
        Tkin_values = [(Tmin+Tmax)/2,]
    else:
        N_values = np.logspace(log_N_limits[0],log_N_limits[1],n)
        coll_density_values = np.logspace(3,5,n)/constants.centi**3
        Tkin_values = np.linspace(Tmin,Tmax,n)

    cloud_kwargs = {'datafilepath':datafilepath,'geometry':geometry,
                    'line_profile_type':line_profile_type,'width_v':width_v,
                    'use_Ng_acceleration':use_Ng_acceleration,
                    'treat_line_overlap':treat_line_overlap,
                    'warn_negative_tau':False}

    print('running pythonradex')
    if remove_cache:
        remove_pythonradex_cache()
    start = time.time()
    cloud = radiative_transfer.Cloud(**cloud_kwargs)
    #IMPORTANT: here I put N in the outer loop on purpose to have the worst case
    #if I put N in the innermost loop, performance will be better because
    #rate equations don't need to be re-computed for every iteration
    for N,coll_dens,Tkin in itertools.product(N_values,coll_density_values,
                                              Tkin_values):
        collider_densities = {collider:coll_dens for collider in colliders}
        cloud.update_parameters(ext_background=ext_background,Tkin=Tkin,
                                collider_densities=collider_densities,N=N,T_dust=0,
                                tau_dust=0)
        cloud.solve_radiative_transfer()
    end = time.time()
    pythonradex_times[i] = end-start

    print('running pythonradex in grid mode')
    if remove_cache:
        remove_pythonradex_cache()
    start = time.time()
    cloud = radiative_transfer.Cloud(**cloud_kwargs)
    requested_output=['level_pop','Tex','tau_nu0_individual_transitions',
                      'fluxes_of_individual_transitions']
    collider_densities_values={collider:coll_density_values for collider in
                               colliders}
    grid = cloud.model_grid(ext_backgrounds={'extbg':ext_background},N_values=N_values,
                            Tkin_values=Tkin_values,
                            collider_densities_values=collider_densities_values,
                            requested_output=requested_output,solid_angle=1)
    for model in grid:
        pass
    end = time.time()
    pythonradex_grid_times[i] = end-start

    print('Running RADEX')
    start = time.time()
    for N,coll_dens,Tkin in itertools.product(N_values,coll_density_values,
                                              Tkin_values):
        #print(N,coll_dens,Tkin)
        collider_densities = {collider:coll_dens for collider in colliders}
        with open(radex_input_file,mode='w') as f:
            f.write(data_filename+'\n')
            f.write('radex_test_performance.out\n')
            f.write('0 0\n')
            f.write(f'{Tkin}\n')
            f.write(f'{len(collider_densities)}\n')
            for collider,density in collider_densities.items():
                f.write(radex_collider_keys[collider]+'\n')
                f.write(f'{density/constants.centi**-3}\n')
            f.write('2.73\n')
            f.write(f'{N/constants.centi**-2}\n')
            f.write(f'{width_v/constants.kilo}\n')
            f.write('0\n')
        os.system(f'{radex_executable} < {radex_input_file} > /dev/null')
        #os.system(f'radex < {radex_input_file}')
    end = time.time()
    RADEX_times[i] = end-start
    print(f'time ratio pythonradex/RADEX: {pythonradex_times[i]/RADEX_times[i]:.3g}')
    print(f'time ratio pythonradex grid/RADEX: {pythonradex_grid_times[i]/RADEX_times[i]:.3g}')

fig,ax = plt.subplots()
ax.plot(n_elements,pythonradex_times/RADEX_times,label='normal')
ax.plot(n_elements,pythonradex_grid_times/RADEX_times,label='grid')
secax = ax.secondary_xaxis('top', functions=(lambda x: x**3, lambda x: x**(1/3)))
secax.set_xlabel('total number of calculations')
plt.xlabel('n_elements')
plt.ylabel('pythonradex time / RADEX time')
ax.legend(loc='best')