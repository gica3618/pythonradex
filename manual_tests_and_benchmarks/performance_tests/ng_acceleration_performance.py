#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:06:20 2024

@author: gianni
"""

#test also how large the impact of ng acceleration is

import sys
sys.path.append('..')
import general
from pythonradex import radiative_transfer,helpers
from scipy import constants
import timeit
import numpy as np

# data_filename = 'hco+.dat'
# coll_partner_densities = {'H2':1e3/constants.centi**3}
data_filename = 'co.dat'
collider_densities = {'para-H2':1e4/constants.centi**3}


data_folder = '../../tests/LAMDA_files'
datafilepath = general.datafilepath(data_filename)
geometry = 'static sphere'
ext_background = helpers.generate_CMB_background(z=0)
#ext_background = helpers.zero_background

N_values = (1e13/constants.centi**2,1e14/constants.centi**2,1e17/constants.centi**2,
            1e20/constants.centi**2)
Tkin = 20
line_profile_type = 'rectangular'
width_v = 1*constants.kilo
timeit_number = 10

for N in N_values:
    print(f'N={N/constants.centi**-2:.2g} cm-2')
    for ng in (True,False):
        cloud = radiative_transfer.Cloud(
                    datafilepath=datafilepath,geometry=geometry,
                    line_profile_type=line_profile_type,width_v=width_v,
                    warn_negative_tau=False,use_Ng_acceleration=ng)
        cloud.update_parameters(
                     ext_background=ext_background,Tkin=Tkin,N=N,
                     collider_densities=collider_densities,T_dust=0,
                     tau_dust=0)
        cloud.solve_radiative_transfer()
        solve_time = timeit.timeit(cloud.solve_radiative_transfer,
                                   number=timeit_number)
        max_tau = np.max(cloud.tau_nu0_individual_transitions)
        Tex_10 = cloud.Tex[0]
        print(f'time for ng {ng}, max tau {max_tau:.4g}, '
              +f'Tex={Tex_10:.4g}: {solve_time:.2g}')
    print('\n')