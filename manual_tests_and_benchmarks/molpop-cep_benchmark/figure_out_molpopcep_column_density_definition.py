#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:04:49 2025

@author: gianni
"""

#molpopcep gives column density in units of cm-2/(km/s), so to convert this to
#cm-2, I need to multiply by some velocity (I guess Doppler parameter or
#line FWHM)
#try to figure it out by looking at optical depth

#consider an LTE model, to be able to determine level populations analytically
#use HCN_model_no_overlap_LTE.out

#we find that to get the total column density, just multiply N_kms from molpop-cep
#with the Doppler parameter

from pythonradex import molecule
from scipy import constants
import numpy as np
import sys
sys.path.append('..')
import general

datafilepath = general.datafilepath('hcn@hfs.dat')

Doppler = 1.502*constants.kilo
width_v = 2*np.sqrt(np.log(2))*Doppler

#taking the top emitting line for each example column density
cases = [{"N_kms":1.18E+14/constants.centi**2/constants.kilo,"trans_index":30,
          "tau_molpop":0.707},
         {"N_kms":6.66E+12/constants.centi**2/constants.kilo,"trans_index":24,
          "tau_molpop":0.0712},
         {"N_kms":3.74E+17/constants.centi**2/constants.kilo,"trans_index":42,
          "tau_molpop":3.68E+02}]

hcn = molecule.EmittingMolecule(datafilepath=datafilepath,
                                line_profile_type="Gaussian",
                                width_v=width_v)

for case in cases:
    trans = hcn.rad_transitions[case["trans_index"]]
    Boltzmann_level_population = hcn.Boltzmann_level_population(T=25)
    v_multiplier = {"Doppler":Doppler,"FWHM":width_v}
    print(f"tau molpopcep = {case['tau_molpop']}")
    for ID,v_mult in v_multiplier.items():
        N = case["N_kms"]*v_mult
        N1 = N*Boltzmann_level_population[trans.low.index]
        N2 = N*Boltzmann_level_population[trans.up.index]
        tau = trans.tau_nu0(N1=N1,N2=N2)
        print(f"{ID}: {tau:.3g}")