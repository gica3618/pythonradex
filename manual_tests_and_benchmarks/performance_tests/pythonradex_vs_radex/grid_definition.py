#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 21:40:10 2025

@author: gianni
"""

from scipy import constants
import numpy as np

# actually, RADEX assumes rectangular, but than converts it Gaussian for the line flux
line_profile_type = "rectangular"
width_v = 1 * constants.kilo
# can only use LVG slab and static sphere, since these are the only two
# where RADEX and pythonradex use same escape probability
# geometry = 'static sphere'
geometry = "LVG slab"
n = 10
coll_density_values = np.logspace(3, 5, n) / constants.centi**3

grid = {
    "datafilename": "co.dat",
    "colliders": ["para-H2", "ortho-H2"],
    "N_grid": np.logspace(13 + 4, 18 + 4, n),
    "Tkin_grid": np.linspace(20, 250, n),
}
# narrow CO range where RADEX does not throw warnings:
# grid = {"datafilename":'co.dat','colliders':['para-H2','ortho-H2'],
#         'N_grid':np.logspace(16+4,16.3+4,n),
#         'Tkin_grid':np.linspace(50,51,n)}

# grid = {"datafilename":'hco+.dat','colliders':['H2',],
#         'N_grid':np.logspace(10+4,14+4,n),
#         'Tkin_grid':np.linspace(20,250,n)}

# grid = {"datafilename":'so@lique.dat','colliders':['H2',],
#         'N_grid':np.logspace(10+4,12+4,n),
#         'Tkin_grid':np.linspace(60,250,n)}

# grid = {"datafilename":'c.dat','colliders':['para-H2','ortho-H2'],
#         'N_grid':np.logspace(12+4,18+4,n),
#         'Tkin_grid':np.linspace(60,250,n)}
