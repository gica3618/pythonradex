#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:02:57 2024

@author: gianni
"""

from scipy import constants
import sys
sys.path.append('/home/gianni/science/projects/code/pythonradex')
from pythonradex import nebula,helpers
import time

#check how much time the flux calculation takes compared to the time it takes
#to solve the radiative transfer

geometry = 'uniform sphere'
line_profile = 'rectangular'
width_v = 1*constants.kilo
iteration_mode = 'ALI'
use_NG_acceleration = True
ext_background = helpers.generate_CMB_background(z=0)
Tkin = 49
datafilepath = '/home/gianni/science/LAMDA_database_files/co.dat'
Ntot = 1e17/constants.centi**2
collider_densities = {'para-H2':1e3/constants.centi**3}

start = time.time()
example_nebula = nebula.Nebula(
                    datafilepath=datafilepath,geometry=geometry,
                    line_profile=line_profile,width_v=width_v,
                    verbose=False,iteration_mode=iteration_mode,
                    use_NG_acceleration=use_NG_acceleration)
example_nebula.set_cloud_parameters(ext_background=ext_background,Ntot=Ntot,
                                    Tkin=Tkin,collider_densities=collider_densities)
end = time.time()
print(f'setup time: {end-start}')
start = time.time()
example_nebula.solve_radiative_transfer()
end = time.time()
print(f'solve time: {end-start}')
start = time.time()
example_nebula.solve_radiative_transfer()
end = time.time()
print(f'solve time again: {end-start}')
start = time.time()
fluxes = example_nebula.compute_line_fluxes(0.1)
end = time.time()
print(f'flux time: {end-start}')
start = time.time()
fluxes = example_nebula.compute_line_fluxes(0.1)
end = time.time()
print(f'flux time again: {end-start}')