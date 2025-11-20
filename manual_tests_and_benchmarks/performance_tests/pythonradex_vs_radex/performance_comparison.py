#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:50:22 2025

@author: gianni
"""

#this script is essentially a simplified version of radex_vs_pythonradex_performance.py

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


ext_background = helpers.generate_CMB_background(z=0)


#can only use LVG slab and uniform sphere, since these are the only two
#where RADEX and pythonradex use same escape probability
geometry = 'uniform sphere'
# geometry = 'LVG slab'

n_grid_elements = 15

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

#actually, RADEX assumes rectangular, but then converts it Gaussian for the line flux
line_profile_type = 'rectangular'
width_v = 1*constants.kilo

datafilepath = os.path.join(general.lamda_data_folder,data_filename)

radex_input_file = 'radex_test_preformance.inp'
radex_output_file = 'radex_test_preformance.out'

N_values = np.logspace(*log_N_limits,n_grid_elements)
coll_density_values = np.logspace(3,5,n_grid_elements)/constants.centi**3
Tkin_values = np.linspace(Tmin,Tmax,n_grid_elements)

cloud_kwargs = {'datafilepath':datafilepath,'geometry':geometry,
                'line_profile_type':line_profile_type,'width_v':width_v,
                'use_Ng_acceleration':True,
                'treat_line_overlap':False,
                'warn_negative_tau':False}

print('running pythonradex')
start = time.time()
cloud = radiative_transfer.Cloud(**cloud_kwargs)
#IMPORTANT: put N in innermost loop to improve performance
for coll_dens,Tkin,N in itertools.product(coll_density_values,Tkin_values,N_values):
    collider_densities = {collider:coll_dens for collider in colliders}
    cloud.update_parameters(ext_background=ext_background,Tkin=Tkin,
                            collider_densities=collider_densities,N=N,T_dust=0,
                            tau_dust=0)
    cloud.solve_radiative_transfer()
end = time.time()
pythonradex_time = end-start

print('running pythonradex in grid mode')
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
pythonradex_grid_time = end-start

print('Running RADEX')
start = time.time()
for N,coll_dens,Tkin in itertools.product(N_values,coll_density_values,
                                          Tkin_values):
    collider_densities = {collider:coll_dens for collider in colliders}
    # start_wrapper = time.perf_counter()
    radex_times = mini_radex_wrapper.run_radex(
                    datafilename=data_filename,geometry=geometry,
                    collider_densities=collider_densities,Tkin=Tkin,N=N,
                    width_v=width_v,input_filepath=radex_input_file,
                    output_filepath=radex_output_file)
    # end_wrapper = time.perf_counter()
    # print(radex_times)
    # print(f"wrapper time: {end_wrapper-start_wrapper}")
end = time.time()
RADEX_time = end-start
print(f'time ratio pythonradex/RADEX: {pythonradex_time/RADEX_time:.3g}')
print(f'time ratio pythonradex grid/RADEX: {pythonradex_grid_time/RADEX_time:.3g}')