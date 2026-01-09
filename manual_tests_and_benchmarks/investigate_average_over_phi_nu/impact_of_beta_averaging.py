#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 19:59:09 2024

@author: gianni
"""

#check whether it makes a difference to the results whether beta is averaged over
#the line profile or not
#conclusion: probably not important

import sys
sys.path.append('..')
import general
from pythonradex import radiative_transfer,helpers
from scipy import constants
import numpy as np
import itertools

trans_index = 0
geometry = 'static sphere'
line_profile_type = 'rectangular'
width_v = 1*constants.kilo


test_cases = [
              {'data_filename':'co.dat',
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

ext_background = helpers.generate_CMB_background(z=0)
data_folder = general.lamda_data_folder
for test_case in test_cases:
    print(test_case)
    for coll_density,N,Tkin in itertools.product(test_case['coll_partner_densities'],
                                                 test_case['N_values'],
                                                 test_case['Tkin_values']):
        cloud_kwargs = {'datafilepath':general.datafilepath(test_case['data_filename']),
                        'geometry':geometry,'line_profile_type':line_profile_type,
                        'width_v':width_v,'use_Ng_acceleration':True}
        cloud_params = {'ext_background':ext_background,'Tkin':Tkin,
                        'collider_densities':{test_case['coll_partner']:coll_density},
                        'N':N,'T_dust':0,'tau_dust':0}
        fluxes = []
        Tex = []
        taus = []
        for treat_overlap in (True,False):
            source = radiative_transfer.Source(
                                  treat_line_overlap=treat_overlap,**cloud_kwargs)
            source.update_parameters(**cloud_params)
            source.solve_radiative_transfer()
            Tex.append(source.Tex[trans_index])
            fluxes.append(source.frequency_integrated_emission_of_individual_transitions(
                                       output_type="flux",solid_angle=1,
                                       transitions=[trans_index,]))
            taus.append(source.tau_nu0_individual_transitions[trans_index])
        Tex_relative_diff = relative_diff(Tex)
        flux_relative_diff = relative_diff(fluxes)
        print(Tex_relative_diff,flux_relative_diff)
        if Tex_relative_diff > 0.1 or flux_relative_diff > 0.1:
            print(f'{test_case["data_filename"]}, {test_case["coll_partner"]}')
            print(f'coll dens = {coll_density/constants.centi**-3:.2g} cm-3, '
                  +f'N = {N/constants.centi**-2:.2g} cm-2, Tkin = {Tkin}')
            print(f'Tex: {Tex}')
            print(f'flux: {fluxes}')
            print(f'tau: {taus}')
            print(f'Tex diff: {Tex_relative_diff*100}%')
            print(f'flux diff: {flux_relative_diff*100}%')