#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:35:49 2025

@author: gianni
"""
import sys
sys.path.append('..')
import general
from pythonradex import radiative_transfer,helpers
import numpy as np
from scipy import constants

#calculate some simple CO models to make sure I understand the inputs of molpop-cep

ref_transitions = [1,2]

def FWHM(Doppler_param):
    return 2*np.sqrt(np.log(2))*Doppler_param

datafilepath = general.datafilepath('co.dat')
width_v = FWHM(1*constants.kilo)
source = radiative_transfer.Source(
                      datafilepath=datafilepath,geometry='static slab',
                      line_profile_type='rectangular',width_v=width_v)

n = 1e4/constants.centi**3
Tkin = 100
ext_background = helpers.generate_CMB_background()
collider_densities = {'para-H2':n/2,'ortho-H2':n/2}
for N in np.array([1e16,1e17,1e18,1e19])/constants.centi**2:
    print(f'N={N/constants.centi**-2:.1g} cm-2')
    source.update_parameters(N=N,Tkin=Tkin,collider_densities=collider_densities,
                            ext_background=ext_background,T_dust=0,tau_dust=0)
    source.solve_radiative_transfer()
    for i in ref_transitions:
        print(f'trans {i}:')
        print(f'Tex={source.Tex[i]:.3g} K')
        print(f'tau_nu0={source.tau_nu0_individual_transitions[i]}')
    print('\n')