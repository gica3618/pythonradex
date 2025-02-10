#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:55:42 2024

@author: gianni
"""

#check the times it takes for the different parts of pythonradex to run
import sys
sys.path.append('..')
import general
from pythonradex import radiative_transfer
from scipy import constants
import time
import numpy as np

# data_filename = 'hco+.dat'
# collider_densities = {'H2':1e3/constants.centi**3}
data_filename = 'co.dat'
collider_densities = {'para-H2':1e3/constants.centi**3,
                          'ortho-H2':1e3/constants.centi**3}
datafilepath = general.datafilepath(data_filename)

geometry = 'uniform sphere'
ext_background = 0
N = 1e16/constants.centi**2
line_profile_type = 'Gaussian'
width_v = 1*constants.kilo
use_Ng_acceleration = True
treat_line_overlap = False
niter = 5
#I choose different Tkin for each iteration to force setting up the
#rate equations for each iteration
Tkin_values = np.linspace(20,40,niter) 

start = time.time()
cloud = radiative_transfer.Cloud(
                    datafilepath=datafilepath,geometry=geometry,
                    line_profile_type=line_profile_type,width_v=width_v,
                    use_Ng_acceleration=use_Ng_acceleration,
                    treat_line_overlap=treat_line_overlap)
end = time.time()
print(f'setup time: {end-start:.3g}')

for i in range(niter):
    tot_time = 0
    print(f'iteration {i+1}')
    start = time.time()
    cloud.update_parameters(
              ext_background=ext_background,Tkin=Tkin_values[i],
              collider_densities=collider_densities,N=N,T_dust=0,tau_dust=0)
    end = time.time()
    tot_time += end-start
    print(f'setup params: {end-start:.3g}')
    start = time.time()
    cloud.solve_radiative_transfer()
    end = time.time()
    tot_time += end-start
    print(f'solve time: {end-start:.3g}')
    start = time.time()
    cloud.fluxes_of_individual_transitions(solid_angle=1,transitions=None)
    end = time.time()
    tot_time += end-start
    print(f'flux time: {end-start:.3g}')
    print(f'total time: {tot_time}\n')