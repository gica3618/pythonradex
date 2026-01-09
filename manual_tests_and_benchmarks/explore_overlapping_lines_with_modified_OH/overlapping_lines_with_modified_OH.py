#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 18:24:25 2025

@author: gianni
"""

#the idea is to modify the atomic and collisional data such that treating line
#overlap becomes very important. I do this by considering the overlapping transistions
#of OH: 5->2 and 6->2. I shut down all collisional transitions involving level 5, and
#also all radiative transitions into level 5. Then the only way to excite level 5
#is via photons from the overlapping line

import sys
sys.path.append('..')
import general
from pythonradex import radiative_transfer,helpers,molecule
from scipy import constants

ref_transitions = [4,6]
line_profile_type = 'rectangular'

datafilepath = general.datafilepath('oh@hfs_modified.dat')

cloud_kwargs = {'datafilepath':datafilepath,'geometry':'static slab',
                'line_profile_type':line_profile_type,'warn_negative_tau':False}
param_kwargs = {'Tkin':100,'ext_background':helpers.generate_CMB_background(),
                'T_dust':0,'tau_dust':0}

def run_models(width_v,collider_densities,N,run_only_with_overlap_treatment=False):
    for treat_line_overlap in (False,True):
        if run_only_with_overlap_treatment and not treat_line_overlap:
            continue
        print(f'line overlap: {treat_line_overlap}')
        source = radiative_transfer.Source(**cloud_kwargs,width_v=width_v,
                                         treat_line_overlap=treat_line_overlap)
        source.update_parameters(**param_kwargs,collider_densities=collider_densities,
                                N=N)
        source.solve_radiative_transfer()
        for i in ref_transitions:
            print(f'trans {i}:')
            print(f'Tex={source.Tex[i]:.3g} K')
            print(f'tau_nu0={source.tau_nu0_individual_transitions[i]:.3g}')
            trans = source.emitting_molecule.rad_transitions[i]
            #tau remains constant although Tex changes, impossible? It's because
            #the lower level is very sparsely populated:
            print(f"up,low: {source.level_pop[trans.up.index]:.3g}, {source.level_pop[trans.low.index]:.3g}")
        print('\n')
    print('\n\n')

print('Case 1: linewidth too small for overlap:')
#we see that line overlap treatment has no effect, and that the excitation
#temperature of trans 4 is equal to CMB, as expected
run_models(width_v=0.01*constants.kilo,
           collider_densities={collider:1e5/constants.centi**3 for collider in
                                 ('ortho-H2','para-H2')},
           N=1e12/constants.centi**2)

print('Case 2: partial overlap non-LTE')
#we see that the photons from the overlapping line can raise Tex of trans 4
width_v_partial_overlap = 2*constants.kilo
N_partial_overlap = 1e13/constants.centi**2
mol = molecule.EmittingMolecule(datafilepath=datafilepath,
                                line_profile_type=line_profile_type,
                                width_v=width_v_partial_overlap)
for i in ref_transitions:
    other_trans = [j for j in ref_transitions if j!=i]
    assert len(other_trans) == 1
    for j in other_trans:
        assert j in mol.overlapping_lines[i]
ref_nu0 = mol.rad_transitions[ref_transitions[0]].nu0
Delta_nu0 = ref_nu0 - mol.rad_transitions[ref_transitions[1]].nu0
Delta_v = Delta_nu0/ref_nu0*constants.c
print(f'v distance between the two ref transitions: {Delta_v/constants.kilo} km/s')
coll_dens_nonLTE = {collider:1e5/constants.centi**3 for collider in
                    ('ortho-H2','para-H2')}
run_models(width_v=width_v_partial_overlap,collider_densities=coll_dens_nonLTE,
           N=N_partial_overlap)

print('Case 3: total overlap non-LTE')
#more photons available thanks to total overlap, thus Tex rises even more
width_v_total_overlap = 30*constants.kilo
#to have the same optical depth:
N_total_overlap = N_partial_overlap*width_v_total_overlap/width_v_partial_overlap
run_models(width_v=width_v_total_overlap,collider_densities=coll_dens_nonLTE,
           N=N_total_overlap)

#in LTE, Tex of trans 4 rises even more
print('Case 4: partial overlap LTE')
coll_dens_LTE = {collider:1e9/constants.centi**3 for collider in
                 ('ortho-H2','para-H2')}
run_models(width_v=width_v_partial_overlap,collider_densities=coll_dens_LTE,
           N=N_partial_overlap)

print('Case 5: total overlap LTE')
run_models(width_v=width_v_total_overlap,collider_densities=coll_dens_LTE,
           N=N_total_overlap)

print('Case 6: optically thick')
#completely optically thick, Tex~Tkin, as expected
#running without overlap treatment fails
run_models(width_v=width_v_total_overlap,collider_densities=coll_dens_LTE,
           N=1e17/constants.centi**2,run_only_with_overlap_treatment=True)