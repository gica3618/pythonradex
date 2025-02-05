#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:34:46 2025

@author: gianni
"""

#benchmark overlapping transitions by using HCN

from scipy import constants
import sys
sys.path.append('/home/gianni/science/projects/code/pythonradex')
from pythonradex import radiative_transfer,helpers

ref_transitions = [3,4,5,6,7,8]
molpop_cep_Tex = {'treat overlap False':[6.29,5.51,6.78,9.36,4.94,5.24],
                  'treat overlap True':[10,10.4,10.6,10.6,6.64,6.88]}

datafilepath = '../LAMDA_files/hcn@hfs.dat'
width_v = 2.5*constants.kilo
N = 3.74E+14/constants.centi**2
Tkin = 25
collider_densities = {'H2':1e5/constants.centi**3}
ext_background = helpers.generate_CMB_background()
for treat_line_overlap in (True,False):
    print(f'treat_line_overlap={treat_line_overlap}')
    cloud = radiative_transfer.Cloud(
                          datafilepath=datafilepath,geometry='uniform slab',
                          line_profile_type='rectangular',width_v=width_v,
                          treat_line_overlap=treat_line_overlap,warn_negative_tau=False)
    for i in ref_transitions:
        assert cloud.emitting_molecule.any_line_has_overlap(line_indices=[i,])
    cloud.update_parameters(N=N,Tkin=Tkin,collider_densities=collider_densities,
                            ext_background=ext_background,T_dust=0,tau_dust=0)
    cloud.solve_radiative_transfer()
    for counter,i in enumerate(ref_transitions):
        Tex_molpop = molpop_cep_Tex[f'treat overlap {treat_line_overlap}'][counter]
        print(f'trans {i}:')
        print(f'Tex={cloud.Tex[i]:.3g} K (molpop: {Tex_molpop:.3g} K)')
        print(f'tau_nu0={cloud.tau_nu0_individual_transitions[i]:.3g}')
    print('\n')