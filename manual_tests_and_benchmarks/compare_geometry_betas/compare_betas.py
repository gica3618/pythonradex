#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:03:21 2024

@author: gianni
"""

import numpy as np
import matplotlib.pyplot as plt
from pythonradex import escape_probability_functions as epf

tau_grid = np.logspace(-3,3,100)

beta_funcs = {'static sphere':epf.beta_static_sphere, #same as 'static sphere RADEX'
              'LVG sphere':epf.beta_LVG_sphere,
              'LVG sphere RADEX':epf.beta_LVG_sphere_RADEX,
              'LVG slab':epf.beta_LVG_slab,
              'static slab':epf.beta_static_slab
              }

fig,ax = plt.subplots()
for ID,func in beta_funcs.items():
    beta = func(tau_grid)
    ax.plot(tau_grid,beta,label=ID)

ax.set_xscale('log')
ax.set_xlabel('tau')
ax.set_ylabel('beta')
ax.legend(loc='best')