#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:04:47 2023

@author: gianni
"""

import sys
sys.path.append('/home/gianni/science/projects/code/pythonradex')
from pythonradex import nebula,helpers
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
n_elements = [5,7,10,15] #for Tkin, collider and Ntot
#most optimistic case is if only Ntot is varied because then rate equations don't
#need to be re-calculated every time
vary_only_Ntot = False

data_filename = 'co.dat'
collider = 'para-H2'
log_Ntot_limits = 14+4,18+4
Tmin,Tmax = 20,250
# data_filename = 'hco+.dat'
# collider = 'H2'
# log_Ntot_limits = 10+4,14+4
# data_filename = 'so@lique.dat'
# collider = 'H2'
# log_Ntot_limits = 10+4,12+4
# Tmin,Tmax = 60,250


# 
# Tkin = 50
# collider_densities = {'H2':1e5/constants.centi**3}
# Ntot_values = np.array((1e11,1e12,1e14,1e16,1e18))/constants.centi**2

# data_filename = 'c.dat'
# Tkin = 100
# #ATTENTION: if no H2 is given, RADEX just puts 1e5 cm-3 by default! WTF!
#RADEX does other strange things with the H2 density:
#see line 112 in io.f, line 168 in io.f, and line 225 in readdata.f
# collider_densities = {'e':1e3/constants.centi**3}
# #Ntot_values = np.array((1e11,1e12,1e14,1e16,1e18))/constants.centi**2
# Ntot_values = np.array((1e18,))/constants.centi**2


# data_filename = 'so@lique.dat'
# Tkin = 100
# collider_densities = {'H2':1e3/constants.centi**3}
# Ntot_values = np.array((1e11,1e12,1e14,1e16,1e18))/constants.centi**2


geometry = 'uniform sphere' #verify that the RADEX you are using was compiled with this geometry
line_profile = 'rectangular' #actually, RADEX assumed rectangular, but than converts it Gaussian for the line flux
width_v = 1*constants.kilo
iteration_mode = 'ALI'
use_NG_acceleration = True
average_beta_over_line_profile = False
remove_cache = True


data_folder = '../LAMDA_files'
datafilepath = os.path.join(data_folder,data_filename)
radex_input_file = 'radex_test_preformance.inp'
radex_executable = '../Radex/bin/radex'

pythonradex_times = np.empty(len(n_elements))
RADEX_times = np.empty_like(pythonradex_times)

for i,n in enumerate(n_elements):
    if remove_cache:
        cache_folder = '/home/gianni/science/projects/code/pythonradex/pythonradex/__pycache__'
        if os.path.exists(cache_folder):
            print('removing python cache')
            shutil.rmtree(cache_folder)
    print(f'n elements: {n}')
    if vary_only_Ntot:
        Ntot_values = np.logspace(log_Ntot_limits[0],log_Ntot_limits[1],n**3)
        coll_density_values = [1e4/constants.centi**3,]
        Tkin_values = [(Tmin+Tmax)/2,]
    else:
        Ntot_values = np.logspace(log_Ntot_limits[0],log_Ntot_limits[1],n)
        coll_density_values = np.logspace(3,5,n)/constants.centi**3
        Tkin_values = np.linspace(Tmin,Tmax,n)

    print('running pythonradex')
    start = time.time()
    example_nebula = nebula.Nebula(
                        datafilepath=datafilepath,geometry=geometry,
                        line_profile=line_profile,width_v=width_v,
                        verbose=False,iteration_mode=iteration_mode,
                        use_NG_acceleration=use_NG_acceleration,
                        average_beta_over_line_profile=average_beta_over_line_profile)
    #IMPORTANT: here I put Ntot in the outer loop on purpose to have the worst case
    #if I put Ntot in the innermost loop, performance will be better because
    #rate equations don't need to be re-computed for every iteration
    for Ntot,coll_dens,Tkin in itertools.product(Ntot_values,coll_density_values,
                                                 Tkin_values):
        collider_densities = {collider:coll_dens}
        example_nebula.set_cloud_parameters(
                             ext_background=ext_background,Tkin=Tkin,
                                collider_densities=collider_densities,Ntot=Ntot)
        example_nebula.solve_radiative_transfer()
    end = time.time()
    pythonradex_times[i] = end-start
    
    print('Running RADEX')
    start = time.time()
    for Ntot,coll_dens,Tkin in itertools.product(Ntot_values,coll_density_values,
                                                 Tkin_values):
        collider_densities = {collider:coll_dens}
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
            f.write(f'{Ntot/constants.centi**-2}\n')
            f.write(f'{width_v/constants.kilo}\n')
            f.write('0\n')
        os.system(f'{radex_executable} < {radex_input_file} > /dev/null')
        #os.system(f'radex < {radex_input_file}')
    end = time.time()
    RADEX_times[i] = end-start

    print(f'time ratio pythonradex/RADEX: {pythonradex_times[-1]/RADEX_times[-1]:.3g}')

fig,ax = plt.subplots()
ax.plot(n_elements,pythonradex_times/RADEX_times)
secax = ax.secondary_xaxis('top', functions=(lambda x: x**3, lambda x: x**(1/3)))
secax.set_xlabel('total number of calculations')
plt.xlabel('n_elements')
plt.ylabel('pythonradex time / RADEX time')