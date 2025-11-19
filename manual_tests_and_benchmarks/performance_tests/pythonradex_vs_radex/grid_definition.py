#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 21:40:10 2025

@author: gianni
"""

from scipy import constants
import numpy as np

#actually, RADEX assumes rectangular, but than converts it Gaussian for the line flux
line_profile_type = 'rectangular'
width_v = 1*constants.kilo
#can only use LVG slab and uniform sphere, since these are the only two
#where RADEX and pythonradex use same escape probability
#geometry = 'uniform sphere'
geometry = 'LVG slab'

n = 20
grid = {"datafilename":'co.dat','colliders':['para-H2','ortho-H2'],
        'N_grid':np.logspace(16+4,16.3+4,n),
        'Tkin_grid':np.linspace(50,51,n),
        'coll_density_values':np.logspace(3,5,n)/constants.centi**3}