#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:08:53 2024

@author: gianni
"""

#Is ALI for overlapping lines really better than LI?

import sys
sys.path.append('/home/gianni/science/projects/code/pythonradex')
from pythonradex import radiative_transfer,helpers
from scipy import constants
import itertools
import os


Tkin = 300
collider_densities = {#'LTE':{'e':1e10/constants.centi**3},
                      'non-LTE':{'e':1e3/constants.centi**3}}
line_profile_types = ['Gaussian',]
                      #'rectangular']
N = {#'thin':1e14/constants.centi**2,'intermediate':1e16/constants.centi**2,
     'thick':1e21/constants.centi**2}
here = os.path.dirname(os.path.abspath(__file__))
#transitions 8 and 10 of CN are separated by ~650 km/s
datafilepath = '../LAMDA_files/cn.dat'
ext_background = helpers.generate_CMB_background()

for lp,(Ntype,N),(coll_dens_type,coll_dens),ng,treat_overlap in\
               itertools.product(line_profile_types,N.items(),collider_densities.items(),
                                 (True,False),(True,False)):
    print(f'Ntype: {Ntype}, coll dens: {coll_dens_type}, line profile: {lp},'
          +f' ng: {ng}, treat overlap: {treat_overlap}')
    for iter_mode in ('LI','ALI'):
        print(iter_mode)
        try:
            cld = radiative_transfer.Cloud(
                                  datafilepath=datafilepath,geometry='uniform sphere',
                                  line_profile_type=lp,width_v=1000*constants.kilo,
                                  iteration_mode=iter_mode,use_NG_acceleration=ng,
                                  average_over_line_profile=True,
                                  treat_line_overlap=treat_overlap)
            cld.set_parameters(ext_background=ext_background,N=N,Tkin=Tkin,
                               collider_densities=coll_dens)
            cld.solve_radiative_transfer()
        except:
            print('could not solve')
            continue
        print(f'{cld.n_iter_convergence} iterations')
    print('\n')
    