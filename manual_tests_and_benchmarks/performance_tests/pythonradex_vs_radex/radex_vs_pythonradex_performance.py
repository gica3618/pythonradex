#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:04:47 2023

@author: gianni
"""

import sys
sys.path.append('../..')
import general
from pythonradex import radiative_transfer,helpers
from scipy import constants
import os
import numpy as np
import time
import shutil
import matplotlib.pyplot as plt
import itertools
import mini_radex_wrapper

#in this script, radex and pythonradex are directly compared


ext_background = helpers.generate_CMB_background(z=0)

#n_elements = [5,7,10,15,20] #for Tkin, collider and N
n_elements = [5,7,10]

#most optimistic case is if only N is varied because then rate equations don't
#need to be re-calculated every time
#however, seems like it doesn't really change anything...
vary_only_N = False

#can only use LVG slab and static sphere, since these are the only two
#where RADEX and pythonradex use same escape probability
geometry = 'static sphere'
# geometry = 'LVG slab'

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


#ATTENTION: if no H2 is given, RADEX just puts 1e5 cm-3 by default! WTF!
#RADEX does other strange things with the H2 density:
#see line 112 in io.f, line 168 in io.f, and line 225 in readdata.f
for collider in colliders:
    assert 'H2' in collider

#actually, RADEX assumes rectangular, but than converts it Gaussian for optical
#depth and the line flux
line_profile_type = 'rectangular' 
width_v = 1*constants.kilo
use_Ng_acceleration = True
treat_line_overlap = False
remove_cache = True


datafilepath = os.path.join(general.lamda_data_folder,data_filename)

radex_input_file = 'radex_test_preformance.inp'
radex_output_file = 'radex_test_preformance.out'

pythonradex_times = np.empty(len(n_elements))
pythonradex_grid_times = np.empty_like(pythonradex_times)
RADEX_times = np.empty_like(pythonradex_times)

def remove_pythonradex_cache():
    src_folder = '../../../src/pythonradex'
    assert os.path.exists(src_folder)
    cache_folder = os.path.join(src_folder,'__pycache__')
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
    source = radiative_transfer.Source(**cloud_kwargs)
    #IMPORTANT: here I put N in the outer loop on purpose to have the worst case
    #if I put N in the innermost loop, performance will be better because
    #rate equations don't need to be re-computed for every iteration
    for N,coll_dens,Tkin in itertools.product(N_values,coll_density_values,
                                              Tkin_values):
        collider_densities = {collider:coll_dens for collider in colliders}
        source.update_parameters(ext_background=ext_background,Tkin=Tkin,
                                collider_densities=collider_densities,N=N,T_dust=0,
                                tau_dust=0)
        source.solve_radiative_transfer()
        source.frequency_integrated_emission(
                     output_type="flux",solid_angle=1,transitions=None)
    end = time.time()
    pythonradex_times[i] = end-start

    print('running pythonradex in grid mode')
    if remove_cache:
        remove_pythonradex_cache()
    start = time.time()
    source = radiative_transfer.Source(**cloud_kwargs)
    collider_densities_values={collider:coll_density_values for collider in
                               colliders}
    iterator = source.efficient_parameter_iterator(
                 ext_backgrounds={'extbg':ext_background},
                 N_values=N_values,Tkin_values=Tkin_values,
                 collider_densities_values=collider_densities_values,T_dust=0,
                 tau_dust=0)
    for param_values in iterator:
        source.solve_radiative_transfer()
        source.level_pop
        source.Tex
        source.tau_nu0_individual_transitions
        source.frequency_integrated_emission(
                 output_type="flux",solid_angle=0.25)
    end = time.time()
    pythonradex_grid_times[i] = end-start

    print('Running RADEX')
    start = time.time()
    for N,coll_dens,Tkin in itertools.product(N_values,coll_density_values,
                                              Tkin_values):
        collider_densities = {collider:coll_dens for collider in colliders}
        # start_wrapper = time.perf_counter()
        radex_times = mini_radex_wrapper.run_radex(
                         datafilename=data_filename, geometry=geometry,
                         collider_densities=collider_densities, Tkin=Tkin, N=N,
                         width_v=width_v, input_filepath=radex_input_file,
                         output_filepath=radex_output_file)
        # end_wrapper = time.perf_counter()
        # print(radex_times)
        # print(f"wrapper time: {end_wrapper-start_wrapper:.3g}")
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