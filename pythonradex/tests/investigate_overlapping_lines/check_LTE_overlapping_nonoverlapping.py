#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:48:43 2024

@author: gianni
"""

import sys
sys.path.append('../../..')
from pythonradex import radiative_transfer,helpers
from scipy import constants

datafilepath = '../LAMDA_files/cn.dat'
ext_background = helpers.zero_background
N = 1e13/constants.centi**2
Tkin = 300
collider_densities = {'e':0/constants.centi**3}

cld = radiative_transfer.Cloud(
                      datafilepath=datafilepath,geometry='uniform sphere',
                      line_profile_type='Gaussian',width_v=1000*constants.kilo,
                      iteration_mode='LI',use_NG_acceleration=True,
                      average_over_line_profile=True,
                      treat_overlapping_lines=True)
cld.set_parameters(ext_background=ext_background,N=N,Tkin=Tkin,
                   collider_densities=collider_densities)
cld.solve_radiative_transfer()
print(cld.level_pop[0])
print(cld.emitting_molecule.LTE_level_pop(Tkin)[0])