#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:08:41 2025

@author: gianni
"""
#benchmark overlapping transitions by using OH
#the idea is to modify the atomic and collisional data such that treating line
#overlap becomes very important. I do this by considering the overlapping transistions
#of OH: 5->2 and 6->2. I shut down all collisional transitions involving level 5, and
#also all radiative transitions into level 5. Then the only way to excite level 5
#is via photons from the overlapping line
#unfortunately molpop-cep is apparently not able to converge (?) when treating
#line overlap if column density is higher than ~1e11 cm-2.
#if you choose 1e14 cm-2 below, the contrats between treating and not treating
#line overlap is much larger

from scipy import constants
import sys
sys.path.append('/home/gianni/science/projects/code/pythonradex')
from pythonradex import radiative_transfer,helpers

ref_transitions = [4,6]

datafilepath = '../LAMDA_files/oh@hfs_modified.dat'
width_v = 1.8*constants.kilo
N = 2e11/constants.centi**2
#N = 1e14/constants.centi**2
Tkin = 100
collider_densities = {collider:1e9/constants.centi**3 for collider in
                      ('ortho-H2','para-H2')}
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
    for i in ref_transitions:
        print(f'trans {i}:')
        print(f'Tex={cloud.Tex[i]:.3g} K')
        print(f'tau_nu0={cloud.tau_nu0_individual_transitions[i]:.3g}')
    print('\n')