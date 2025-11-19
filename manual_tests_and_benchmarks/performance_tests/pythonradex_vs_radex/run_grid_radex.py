#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 21:40:27 2025

@author: gianni
"""

import grid_definition
import itertools
from scipy import constants
import os
import time

radex_input_file = 'radex_test_preformance.inp'
radex_collider_keys = {'H2':'H2','para-H2':'p-H2','ortho-H2':'o-H2','e':'e',
                       'He':'He'}
radex_executables = {"uniform sphere":'../../../tests/Radex/bin/radex_static_sphere',
                     "LVG slab":'../../../tests/Radex/bin/radex_LVG_slab'}


width_v = grid_definition.width_v/constants.kilo
executable = radex_executables[grid_definition.geometry]
start = time.perf_counter()
for coll_dens,Tkin,N in itertools.product(grid_definition.grid["coll_density_values"],
                                          grid_definition.grid["Tkin_grid"],
                                          grid_definition.grid["N_grid"]):
    #start_setup = time.time()
    #print(N,coll_dens,Tkin)
    collider_densities = {collider:coll_dens for collider in
                          grid_definition.grid["colliders"]}
    with open(radex_input_file,mode='w') as f:
        f.write(grid_definition.grid["datafilename"]+'\n')
        f.write('radex_test_performance.out\n')
        f.write('0 0\n')
        f.write(f'{Tkin}\n')
        f.write(f'{len(collider_densities)}\n')
        for collider,density in collider_densities.items():
            f.write(radex_collider_keys[collider]+'\n')
            f.write(f'{density/constants.centi**-3}\n')
        f.write('2.73\n')
        f.write(f'{N/constants.centi**-2}\n')
        f.write(f'{width_v}\n')
        f.write('0\n')
    #end_setup = time.time()
    #print(f'setup: {end_setup-start_setup}')
    #start_calc = time.time()
    os.system(f'{executable} < {radex_input_file} > /dev/null')
    #end_calc = time.time()
    #print(f'calc: {end_calc-start_calc}')
    #os.system(f'radex < {radex_input_file}')
end = time.perf_counter()
duration = end-start
print(f"duration: {duration} s")