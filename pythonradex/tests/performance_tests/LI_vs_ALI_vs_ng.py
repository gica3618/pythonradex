#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:06:20 2024

@author: gianni
"""

#test if the ALI scheme indeed leads to faster convergence compared to standard LAMDA iteration
#in particular, ALI should be faster in the optically thick case
#turns out that if the iteration stop critertion is not strict (e.g. 1e-2), then
#ALI and LI have similar performance. But if we adopt the same criterion as RADEX
#(which seems indeed necessary), then ALI is indeed faster

#test also how large the impact of ng acceleration is

import sys
sys.path.append('/home/gianni/science/projects/code/pythonradex')
from pythonradex import radiative_transfer,helpers
import os
from scipy import constants
import timeit
import numpy as np

# data_filename = 'hco+.dat'
# coll_partner_densities = {'H2':1e3/constants.centi**3}
data_filename = 'co.dat'
collider_densities = {'para-H2':1e4/constants.centi**3}


data_folder = '../LAMDA_files'
datafilepath = os.path.join(data_folder,data_filename)
geometry = 'uniform sphere'
ext_background = helpers.generate_CMB_background(z=0)

N_values = (1e13/constants.centi**2,1e18/constants.centi**2,1e20/constants.centi**2,
               1e22/constants.centi**2)
Tkin = 20
line_profile_type = 'rectangular'
width_v = 1*constants.kilo
timeit_number = 10

for N in N_values:
    print(f'N={N/constants.centi**2:.2g} cm-2')
    for mode in ('ALI','LI'):
        for ng in (True,False):
            cloud = radiative_transfer.Cloud(
                        datafilepath=datafilepath,geometry=geometry,
                        line_profile_type=line_profile_type,width_v=width_v,
                        iteration_mode=mode,warn_negative_tau=False,
                        use_NG_acceleration=ng)
            cloud.set_parameters(
                         ext_background=ext_background,Tkin=Tkin,N=N,
                         collider_densities=collider_densities,)
            cloud.solve_radiative_transfer()
            solve_time = timeit.timeit(cloud.solve_radiative_transfer,
                                       number=timeit_number)
            max_tau = np.max(cloud.tau_nu0)
            Tex_10 = cloud.Tex[0]
            print(f'time for {mode} (ng {ng}), max tau {max_tau:.6g}, '
                  +f'Tex={Tex_10:.6g}: {solve_time:.2g}')
    print('\n')