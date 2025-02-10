#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:26:43 2024

@author: gianni
"""

#dust is similar to an overlapping line, so in principle I should average over
#the line profile. check here how much the results are different

import sys
sys.path.append('../../src')
from pythonradex import radiative_transfer
from scipy import constants
import numpy as np
import os
import itertools

trans_index = 0
geometry = 'uniform sphere'
line_profile_type = 'rectangular'
width_v = 1*constants.kilo


test_cases = [{'data_filename':'co.dat',
               'coll_partner':'para-H2',
               'coll_partner_densities':np.array((1e2,1e4,1e6))/constants.centi**3,
               'N_values':np.array((1e14,1e16,1e18,1e20,1e22))/constants.centi**2,
               'Tkin_values':[10,20,50,100]},
              {'data_filename':'c.dat',
               'coll_partner':'e',
               'coll_partner_densities':np.array((1e1,1e2,1e3))/constants.centi**3,
               'N_values':np.array((1e14,1e16,1e18,1e20,1e22))/constants.centi**2,
               'Tkin_values':[10,20,50,100]},
              {'data_filename':'cs.dat',
               'coll_partner':'para-H2',
               'coll_partner_densities':np.array((1e3,1e5))/constants.centi**3,
               'N_values':np.array((1e14,1e16,1e18,1e20,1e22))/constants.centi**2,
               'Tkin_values':[10,20,50,100]},
              {'data_filename':'c+.dat',
               'coll_partner':'e',
               'coll_partner_densities':np.array((1e1,1e2,1e3))/constants.centi**3,
               'N_values':np.array((1e14,1e16,1e18,1e20,1e22))/constants.centi**2,
               'Tkin_values':[10,20,50,100]},]

def relative_diff(x):
    assert len(x) == 2
    return float(np.abs(np.diff(x)/x[0]))

ext_background = 0
data_folder = '../../tests/LAMDA_files'
for test_case in test_cases:
    for coll_density,N,Tkin in itertools.product(test_case['coll_partner_densities'],
                                                 test_case['N_values'],
                                                 test_case['Tkin_values']):
        cloud_kwargs = {'datafilepath':os.path.join(data_folder,test_case['data_filename']),
                        'geometry':geometry,'line_profile_type':line_profile_type,
                        'width_v':width_v,'use_Ng_acceleration':True}
        cloud_params = {'ext_background':ext_background,'Tkin':Tkin,
                        'collider_densities':{test_case['coll_partner']:coll_density},
                        'N':N,'T_dust':2*Tkin,'tau_dust':1}
        Tex = []
        taus = []
        for treat_overlap in (True,False):
            cloud = radiative_transfer.Cloud(
                                  treat_line_overlap=treat_overlap,**cloud_kwargs)
            cloud.update_parameters(**cloud_params)
            cloud.solve_radiative_transfer()
            Tex.append(cloud.Tex[trans_index])
            taus.append(cloud.tau_nu0_individual_transitions[trans_index])
        Tex_relative_diff = relative_diff(Tex)
        print(Tex_relative_diff)
        if Tex_relative_diff > 0.01:
            print(f'{test_case["data_filename"]}, {test_case["coll_partner"]}')
            print(f'coll dens = {coll_density/constants.centi**-3:.2g} cm-3, '
                  +f'N = {N/constants.centi**-2:.2g} cm-2, Tkin = {Tkin}')
            print(f'Tex: {Tex}')
            print(f'tau: {taus}')
            print(f'Tex diff: {Tex_relative_diff*100}%')