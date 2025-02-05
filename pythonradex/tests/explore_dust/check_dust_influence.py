#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:17:18 2024

@author: gianni
"""

#check if/how the dust influences the level populations

import sys
sys.path.append('/home/gianni/science/projects/code/pythonradex')
from pythonradex import radiative_transfer,helpers
from scipy import constants
import numpy as np
import matplotlib.pyplot as plt


datafilepath = '../LAMDA_files/co.dat'
geometry = 'uniform sphere'
line_profile_type = 'rectangular'
width_v = 1*constants.kilo
use_Ng_acceleration = True
treat_line_overlap_values = (False,True)
ext_background = 0
N = 1e14/constants.centi**2
Tkin = 30
collider_densities = {'para-H2':1/constants.centi**3}

T_dust_value = 200
tau_dust_value = 10
max_plot_level = 50

def T_dust(nu):
    return np.ones_like(nu)*T_dust_value

def tau_dust(nu):
    return np.ones_like(nu)*tau_dust_value

dust_params = {'no dust':{'T_dust':0,'tau_dust':0},
               'with dust':{'T_dust':T_dust,'tau_dust':tau_dust}}

for treat_overlap in treat_line_overlap_values:
    fig,ax = plt.subplots()
    ax.set_title(f'treat lin overlap = {treat_overlap}')
    for d,(dust_case,dust_p) in enumerate(dust_params.items()):
        cld = radiative_transfer.Cloud(
                              datafilepath=datafilepath,geometry=geometry,
                              line_profile_type=line_profile_type,
                              width_v=width_v,
                              use_Ng_acceleration=use_Ng_acceleration,
                              treat_line_overlap=treat_overlap)
        cld.update_parameters(ext_background=ext_background,N=N,Tkin=Tkin,
                              collider_densities=collider_densities,**dust_p)
        cld.solve_radiative_transfer()
        expected_level_pop = cld.emitting_molecule.LTE_level_pop(
                                                      T=T_dust_value)
        ax.plot(cld.level_pop[:max_plot_level],label=dust_case)
        if d==0:
            ax.plot(cld.emitting_molecule.LTE_level_pop(Tkin)[:max_plot_level],
                    label='LTE Tkin',linestyle='dashed')
            ax.plot(cld.emitting_molecule.LTE_level_pop(T_dust_value)[:max_plot_level],
                    label='LTE Tdust',linestyle='dashed')
    ax.set_yscale('log')
    ax.set_xlabel('level')
    ax.set_ylabel('frac population')
    ax.legend(loc='best')