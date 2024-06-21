#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:55:42 2024

@author: gianni
"""

#check the times it takes for the different parts of pythonradex to run

import sys
sys.path.append('/home/gianni/science/projects/code/pythonradex')
from pythonradex import nebula,helpers
import os
from scipy import constants
import time
import numpy as np

# data_filename = 'hco+.dat'
# collider_densities = {'H2':1e3/constants.centi**3}
data_filename = 'co.dat'
collider_densities = {'para-H2':1e3/constants.centi**3,
                          'ortho-H2':1e3/constants.centi**3}
data_folder = '/home/gianni/science/LAMDA_database_files'
datafilepath = os.path.join(data_folder,data_filename)

geometry = 'uniform sphere'
ext_background = helpers.zero_background
Ntot = 1e16/constants.centi**2
line_profile = 'Gaussian'
width_v = 1*constants.kilo
iteration_mode = 'ALI'
use_NG_acceleration = True
average_beta_over_line_profile = False
niter = 5
#I choose different Tkin for each iteration to force setting up the
#rate equations for each iteration
Tkin_values = np.linspace(20,40,niter) 

start = time.time()
example_nebula = nebula.Nebula(
                    datafilepath=datafilepath,geometry=geometry,
                    line_profile=line_profile,width_v=width_v,
                    verbose=False,iteration_mode=iteration_mode,
                    use_NG_acceleration=use_NG_acceleration,
                    average_beta_over_line_profile=average_beta_over_line_profile)
end = time.time()
print(f'setup time: {end-start:.3g}')

for i in range(niter):
    tot_time = 0
    print(f'iteration {i+1}')
    start = time.time()
    example_nebula.set_cloud_parameters(
              ext_background=ext_background,Tkin=Tkin_values[i],
              collider_densities=collider_densities,Ntot=Ntot)
    end = time.time()
    tot_time += end-start
    print(f'setup params: {end-start:.3g}')
    start = time.time()
    example_nebula.solve_radiative_transfer()
    end = time.time()
    tot_time += end-start
    print(f'solve time: {end-start:.3g}')
    start = time.time()
    example_nebula.compute_line_fluxes(solid_angle=1)
    end = time.time()
    tot_time += end-start
    print(f'flux time: {end-start:.3g}')
    print(f'total time: {tot_time}\n')