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
#if you choose 1e14 cm-2 below, the contrast between treating and not treating
#line overlap is much larger

from scipy import constants
import sys
sys.path.append('..')
import general
from pythonradex import radiative_transfer,helpers
import numpy as np

ref_transitions = [4,6]
Doppler = 1.081*constants.kilo
molpop_Tex = {"treat_overlap":[7.87E+00,7.46E+01],"ignore_overlap":[2.72E+00,8.56E+01]}
molpop_tau_nu0 = {"treat_overlap":[3.32E-04,3.73E-03],"ignore_overlap":[1.20E-01,1.27E+00]}

datafilepath = general.datafilepath('oh@hfs_modified.dat')
width_v = 2*np.sqrt(np.log(2))*Doppler
#molpop cep column densities are in cm-2/(km/s), so need to convert
#with overlap, molpop-cep only goes up to 2.72e11 cm-2/(km/s)...
molpop_column_densities = {"treat_overlap":2.72E+11/constants.centi**2/constants.kilo*Doppler,
                           "ignore_overlap":1.04E+14/constants.centi**2/constants.kilo*Doppler}
Tkin = 100
collider_densities = {collider:1e9/constants.centi**3 for collider in
                      ('ortho-H2','para-H2')}
ext_background = helpers.generate_CMB_background()
for overlap,N in molpop_column_densities.items():
    if overlap == "treat_overlap":
        treat_line_overlap = True
    elif overlap == "ignore_overlap":
        treat_line_overlap = False
    else:
        raise RuntimeError
    print(f'treat_line_overlap={treat_line_overlap}')
    cloud = radiative_transfer.Cloud(
                          datafilepath=datafilepath,geometry='uniform slab',
                          line_profile_type='Gaussian',width_v=width_v,
                          treat_line_overlap=treat_line_overlap,warn_negative_tau=False)
    for i in ref_transitions:
        assert cloud.emitting_molecule.any_line_has_overlap(line_indices=[i,])
    cloud.update_parameters(N=N,Tkin=Tkin,collider_densities=collider_densities,
                            ext_background=ext_background,T_dust=0,tau_dust=0)
    cloud.solve_radiative_transfer()
    for i,trans_index in enumerate(ref_transitions):
        print(f'trans {trans_index}:')
        print(f'Tex={cloud.Tex[trans_index]:.3g} K (molpop: {molpop_Tex[overlap][i]})')
        print(f'tau_nu0={cloud.tau_nu0_individual_transitions[trans_index]:.3g}'
              +f' (molpop: {molpop_tau_nu0[overlap][i]})')
    print('\n')