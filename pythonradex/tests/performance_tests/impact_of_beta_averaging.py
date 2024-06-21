#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 19:59:09 2024

@author: gianni
"""

#check whether it makes a difference to the results whether beta is averaged over
#the line profile or not

import sys
sys.path.append('/home/gianni/science/projects/code/pythonradex')
from pythonradex import nebula,helpers
from scipy import constants
import numpy as np
import os
import itertools

data_filename = 'co.dat'
Tkin = 100
coll_partner_densities = {'para-H2':1e3/constants.centi**3}
trans_index = 0
Ntot_values = np.array((1e14,1e16,1e18,1e20,1e22))/constants.centi**2
geometry = 'uniform sphere'
line_profile = 'rectangular'
width_v = 1*constants.kilo


test_cases = [{'data_filename':'co.dat',
               'coll_partner':'para-H2',
               'coll_partner_densities':np.array((1e2,1e4,1e6))/constants.centi**3,
               'Ntot_values':np.array((1e14,1e16,1e18,1e20,1e22))/constants.centi**2,
               'Tkin_values':[10,20,50,100]},
              {'data_filename':'c.dat',
               'coll_partner':'e',
               'coll_partner_densities':np.array((1e1,1e2,1e3))/constants.centi**3,
               'Ntot_values':np.array((1e14,1e16,1e18,1e20,1e22))/constants.centi**2,
               'Tkin_values':[10,20,50,100]},
              {'data_filename':'cs.dat',
               'coll_partner':'para-H2',
               'coll_partner_densities':np.array((1e3,1e5))/constants.centi**3,
               'Ntot_values':np.array((1e14,1e16,1e18,1e20,1e22))/constants.centi**2,
               'Tkin_values':[10,20,50,100]},
              {'data_filename':'c+.dat',
               'coll_partner':'e',
               'coll_partner_densities':np.array((1e1,1e2,1e3))/constants.centi**3,
               'Ntot_values':np.array((1e14,1e16,1e18,1e20,1e22))/constants.centi**2,
               'Tkin_values':[10,20,50,100]},]

def relative_diff(x):
    assert len(x) == 2
    return float(np.abs(np.diff(x)/x[0]))

ext_background = helpers.generate_CMB_background(z=0)
data_folder = '/home/gianni/science/LAMDA_database_files'
for test_case in test_cases:
    for coll_density,Ntot,Tkin in itertools.product(test_case['coll_partner_densities'],
                                                    test_case['Ntot_values'],
                                                    test_case['Tkin_values']):
        neb_kwargs = {'datafilepath':os.path.join(data_folder,test_case['data_filename']),
                      'geometry':geometry,'line_profile':line_profile,'width_v':width_v,
                      'verbose':False,'iteration_mode':'ALI',
                      'use_NG_acceleration':True}
        cloud_kwargs = {'ext_background':ext_background,'Tkin':Tkin,
                        'collider_densities':{test_case['coll_partner']:coll_density},
                        'Ntot':Ntot}
        fluxes = []
        Tex = []
        taus = []
        for average in (True,False):
            example_nebula = nebula.Nebula(
                                  average_beta_over_line_profile=average,**neb_kwargs)
            example_nebula.set_cloud_parameters(**cloud_kwargs)
            example_nebula.solve_radiative_transfer()
            example_nebula.compute_line_fluxes(solid_angle=1)
            Tex.append(example_nebula.Tex[trans_index])
            fluxes.append(example_nebula.obs_line_fluxes[trans_index])
            taus.append(example_nebula.tau_nu0[trans_index])
        Tex_relative_diff = relative_diff(Tex)
        flux_relative_diff = relative_diff(fluxes)
        if Tex_relative_diff > 0.1 or flux_relative_diff > 0.1:
            print(f'{test_case["data_filename"]}, {test_case["coll_partner"]}')
            print(f'coll dens = {coll_density/constants.centi**-3:.2g} cm-3, '
                  +f'Ntot = {Ntot/constants.centi**-2:.2g} cm-2, Tkin = {Tkin}')
            print(f'Tex: {Tex}')
            print(f'flux: {fluxes}')
            print(f'tau: {taus}')
            print(f'Tex diff: {Tex_relative_diff*100}%')
            print(f'flux diff: {flux_relative_diff*100}%')