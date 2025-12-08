#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 08:08:28 2024

@author: gianni
"""

import itertools
from scipy import constants
import sys
sys.path.append('..')
import general
from pythonradex import radiative_transfer,helpers
import numpy as np
import time


n = 15
N_values = np.logspace(12,15,n)/constants.centi**2
coll_density_values = np.logspace(3,5,n)/constants.centi**3
Tkin_values = np.linspace(20,100,n)
collider = 'para-H2'
ext_background = helpers.generate_CMB_background(z=0)

source = radiative_transfer.Source(
                datafilepath=general.datafilepath('co.dat'),
                geometry='static sphere',line_profile_type='Gaussian',
                width_v=1*constants.kilo,use_Ng_acceleration=True,
                treat_line_overlap=False,warn_negative_tau=False)
#solve a first time to compile the functions:
source.update_parameters(
     ext_background=ext_background,Tkin=Tkin_values[0],
     collider_densities={collider:coll_density_values[0]},N=N_values[0],
     T_dust=0,tau_dust=0)
source.solve_radiative_transfer()

start = time.time()
for N,coll_dens,Tkin in itertools.product(N_values,coll_density_values,
                                          Tkin_values):
    collider_densities = {collider:coll_dens}
    source.update_parameters(
         ext_background=ext_background,Tkin=Tkin,
         collider_densities=collider_densities,N=N,T_dust=0,tau_dust=0)
    source.solve_radiative_transfer()
end = time.time()
print(f'total time to run grid: {end-start}')