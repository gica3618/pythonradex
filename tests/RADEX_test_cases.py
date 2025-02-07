#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:01:09 2024

@author: gianni
"""

import numpy as np
from scipy import constants

width_v = 1*constants.kilo
T_background = 2.73

test_cases = [{'filename':'co.dat',
               'collider_densities_values':[{'ortho-H2':1e2/constants.centi**3,
                                             'para-H2':1e3/constants.centi**3},
                                            {'ortho-H2':1e4/constants.centi**3}],
               'N_values':np.array((1e12,1e15,1e18))/constants.centi**2,
               'Tkin_values':np.array((30,100,200))},
              {'filename':'hcl.dat',
               'collider_densities_values':[{'para-H2':1e4/constants.centi**3},
                                            {'para-H2':1e6/constants.centi**3}],
               'N_values':np.array((1e10,1e12,1e14))/constants.centi**2,
               'Tkin_values':np.array((80,100,200))},
              {'filename':'ocs@xpol.dat',
               'collider_densities_values':[{'H2':1e2/constants.centi**3},
                                            {'H2':1e4/constants.centi**3}],
               'N_values':np.array((1e10,1e13,1e15))/constants.centi**2,
               'Tkin_values':np.array((30,100,200))},
              {'filename':'c.dat',
               #RADEX adds default values of H2, ortho-H2 and para-H2 if not given
               #so need to add
               'collider_densities_values':[{'He':1e2/constants.centi**3,
                                             'e':1/constants.centi**3,
                                             'ortho-H2':1/constants.centi**3,
                                             'para-H2':1/constants.centi**3},
                                            {'He':1e4/constants.centi**3,
                                             'e':1e2/constants.centi**3,
                                             'ortho-H2':1/constants.centi**3,
                                             'para-H2':1/constants.centi**3}],
               'N_values':np.array((1e12,1e15,1e20))/constants.centi**2,
               'Tkin_values':np.array((30,100,140))}]

def RADEX_out_filename(radex_geometry,specie,Tkin,N,collider_densities):
    rg = radex_geometry.replace(' ','').replace('\n','')
    save_filename = f'radex_{rg}_{specie}_Tkin{Tkin}_'\
                    +f'N{N/constants.centi**-2:.1g}'
    for coll,dens in collider_densities.items():
        save_filename += f'_{coll}_{dens/constants.centi**-3:.1g}'
    save_filename += '.out'
    return save_filename