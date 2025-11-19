#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:50:22 2025

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
import itertools


radex_collider_keys = {'H2':'H2','para-H2':'p-H2','ortho-H2':'o-H2','e':'e',
                       'He':'He'}
ext_background = helpers.generate_CMB_background(z=0)


#can only use LVG slab and uniform sphere, since these are the only two
#where RADEX and pythonradex use same escape probability
geometry = 'uniform sphere'
radex_executable = '../../../tests/Radex/bin/radex_static_sphere'
# geometry = 'LVG slab'
# radex_executable = '../../../tests/Radex/bin/radex_LVG_slab'

n_grid_elements = 20

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

# data_filename = 'c.dat'
# colliders = ['para-H2','ortho-H2']
# log_N_limits = 12+4,18+4
# Tmin,Tmax = 60,250


# #ATTENTION: if no H2 is given, RADEX just puts 1e5 cm-3 by default! WTF!
#RADEX does other strange things with the H2 density:
#see line 112 in io.f, line 168 in io.f, and line 225 in readdata.f
for collider in colliders:
    assert 'H2' in collider

line_profile_type = 'rectangular' #actually, RADEX assumes rectangular, but than converts it Gaussian for the line flux
width_v = 1*constants.kilo

datafilepath = os.path.join(general.lamda_data_folder,data_filename)
radex_input_file = 'radex_test_preformance.inp'


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
    #start_setup = time.time()
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
    #end_setup = time.time()
    #print(f'setup: {end_setup-start_setup}')
    #start_calc = time.time()
    os.system(f'{radex_executable} < {radex_input_file} > /dev/null')
    #end_calc = time.time()
    #print(f'calc: {end_calc-start_calc}')
end = time.time()
RADEX_time = end-start
print(f'time ratio pythonradex/RADEX: {pythonradex_time/RADEX_time:.3g}')
print(f'time ratio pythonradex grid/RADEX: {pythonradex_grid_time/RADEX_time:.3g}')