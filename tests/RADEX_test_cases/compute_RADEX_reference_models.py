#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:53:51 2019

@author: gianni
"""
from scipy import constants
import os
folderpath = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(os.path.join(folderpath,'..'))
import RADEX_test_cases
import itertools

radex_input_filename = 'radex.in'
radex_output_filename = 'radex.out'
radex_collider_keys = {'H2':'H2','para-H2':'p-H2','ortho-H2':'o-H2','e':'e',
                       'He':'He'}
executables = {'static sphere':'radex_static_sphere',
               'LVG sphere RADEX':'radex_LVG_sphere',
               'LVG slab RADEX':'radex_LVG_slab'}

exec_paths = {ID:os.path.join(folderpath,f'../Radex/bin/{ex}') for ID,ex in
              executables.items()}

def write_RADEX_input_file(mol_data_filename,Tkin,collider_densities,T_background,
                           N,width_v):
    with open(radex_input_filename,mode='w') as f:
        f.write(mol_data_filename+'\n')
        f.write(radex_output_filename+'\n')
        f.write('0 0\n')
        f.write(f'{Tkin}\n')
        f.write(f'{len(collider_densities)}\n')
        for collider,density in collider_densities.items():
            f.write(radex_collider_keys[collider]+'\n')
            f.write(f'{density/constants.centi**-3}\n')
        f.write(f'{T_background}\n')
        f.write(f'{N/constants.centi**-2}\n')
        f.write(f'{width_v/constants.kilo}\n')
        f.write('0\n')

for test_case in RADEX_test_cases.test_cases:
    filename = test_case['filename']
    specie = filename.split('.')[0]
    for collider_densities,N,Tkin,T_bg in\
                     itertools.product(test_case['collider_densities_values'],
                                       test_case['N_values'],
                                       test_case['Tkin_values'],
                                       test_case["T_background_values"]):
        for geo in ('static sphere','LVG sphere RADEX','LVG slab RADEX'):
            write_RADEX_input_file( 
                   mol_data_filename=filename,Tkin=Tkin,
                   collider_densities=collider_densities,T_background=T_bg,N=N,
                   width_v=RADEX_test_cases.width_v)
            radex_executable = exec_paths[geo]
            os.system(f'{radex_executable} < {radex_input_filename}')
            with open(radex_output_filename,'r') as f:
                for line in f:
                    if 'Geometry' in line:
                        radex_geometry = line.split(':')[1]
                        break
            save_filename = RADEX_test_cases.RADEX_out_filename(
                              radex_geometry=radex_geometry,specie=specie,Tkin=Tkin,
                              T_background=T_bg,N=N,
                              collider_densities=collider_densities)
            os.rename(src=radex_output_filename,dst=save_filename)