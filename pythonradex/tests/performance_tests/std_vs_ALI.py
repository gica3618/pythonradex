#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:06:20 2024

@author: gianni
"""

#test if the ALI scheme indeed leads to faster convergence compared to standard LAMDA iteration
#in particular, ALI should be faster in the optically thick case
#turns out that if the iteration stop critertion is not strict (e.g. 1e-2), then
#ALI and std have similar performance. But if we adopt the same criterion as RADEX
#(which seems indeed necessary), then ALI is indeed faster

import sys
sys.path.append('/home/gianni/science/projects/code/pythonradex')
from pythonradex import nebula,helpers
import os
from scipy import constants
import timeit
import numpy as np

# data_filename = 'hco+.dat'
# coll_partner_densities = {'H2':1e3/constants.centi**3}
data_filename = 'co.dat'
collider_densities = {'para-H2':1e4/constants.centi**3}


data_folder = '/home/gianni/science/LAMDA_database_files'
datafilepath = os.path.join(data_folder,data_filename)
geometry = 'uniform sphere'
ext_background = helpers.generate_CMB_background(z=0)

Ntot_values = (1e13/constants.centi**2,1e18/constants.centi**2,1e20/constants.centi**2,
               1e22/constants.centi**2)
Tkin = 20
line_profile = 'rectangular'
width_v = 1*constants.kilo
timeit_number = 10

for mode in ('ALI','std'):
    for Ntot in Ntot_values:
        example_nebula = nebula.Nebula(
                    datafilepath=datafilepath,geometry=geometry,
                    line_profile=line_profile,width_v=width_v,
                    iteration_mode=mode)
        example_nebula.set_cloud_parameters(
                     ext_background=ext_background,Tkin=Tkin,Ntot=Ntot,
                     collider_densities=collider_densities,)
        example_nebula.solve_radiative_transfer()
        solve_time = timeit.timeit(example_nebula.solve_radiative_transfer,
                                   number=timeit_number)
        max_tau = np.max(example_nebula.tau_nu0)
        Tex_10 = example_nebula.Tex[0]
        print(f'time for {mode}, Ntot={Ntot/constants.centi**-2:.2g}, max tau {max_tau:.6g}, '
              +f'Tex={Tex_10:.6g}: {solve_time:.2g}')