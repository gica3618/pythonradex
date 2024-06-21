#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 08:08:28 2024

@author: gianni
"""

import itertools
from scipy import constants
import sys
sys.path.append('/home/gianni/science/projects/code/pythonradex')
from pythonradex import nebula,helpers
import numpy as np
import time


n = 15
Ntot_values = np.logspace(12,15,n)/constants.centi**2
coll_density_values = np.logspace(3,5,n)/constants.centi**3
Tkin_values = np.linspace(20,100,n)
collider = 'para-H2'
ext_background = helpers.generate_CMB_background(z=0)

neb = nebula.Nebula(
                datafilepath='/home/gianni/science/LAMDA_database_files/co.dat',
                geometry='uniform sphere',line_profile='Gaussian',
                width_v=1*constants.kilo,debug=False,iteration_mode='ALI',
                use_NG_acceleration=True,average_beta_over_line_profile=False,
                warn_negative_tau=False)
#solve a first time to compile the functions:
neb.set_cloud_parameters(
     ext_background=ext_background,Tkin=Tkin_values[0],
     collider_densities={collider:coll_density_values[0]},Ntot=Ntot_values[0])
neb.solve_radiative_transfer()

start = time.time()
for Ntot,coll_dens,Tkin in itertools.product(Ntot_values,coll_density_values,
                                             Tkin_values):
    collider_densities = {collider:coll_dens}
    neb.set_cloud_parameters(
         ext_background=ext_background,Tkin=Tkin,
         collider_densities=collider_densities,Ntot=Ntot)
    neb.solve_radiative_transfer()
end = time.time()
print(f'total time to run grid: {end-start}')