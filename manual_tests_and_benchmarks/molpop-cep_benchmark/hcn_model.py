#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:34:46 2025

@author: gianni
"""

#benchmark overlapping transitions by using HCN
#interestingly, when overlap is off, pythonradex and molpop-cep show quite
#substantial differences in terms of spectra. Tex and tau are generally similar,
#bad small differences can result in quite large differences in the spectra

from scipy import constants
import sys
sys.path.append('..')
import general
from pythonradex import radiative_transfer,helpers,molecule,escape_probability
import numpy as np
import matplotlib.pyplot as plt

ref_transitions = [3,4,5,6,7,8]
Doppler = 1.502*constants.kilo
line_profile_type = "Gaussian"

#case 1: N ~ 4e14 cm-2/(km/s)
# column_densities = {"treat overlap False":3.74E+14/constants.centi**2/constants.kilo * Doppler,
#                     "treat overlap True":3.85E+14/constants.centi**2/constants.kilo * Doppler}
# molpop_cep_Tex = {'treat overlap False':[6.29,5.51,6.78,9.36,4.94,5.24],
#                   'treat overlap True':[10,10.4,10.6,10.6,6.64,6.88]}
# molpop_cep_tau_nu0 = {'treat overlap False':[4.77,5.65,12.5,21.5,0.353,4.69],
#                       'treat overlap True':[3.32,2.71,9.2,17.9,0.279,3.94]}

#case 1: N=2.11e13 cm-2/(km/s)
column_densities = {"treat overlap False":2.11E+13/constants.centi**2/constants.kilo * Doppler,
                    "treat overlap True":2.11E+13/constants.centi**2/constants.kilo * Doppler}
molpop_cep_Tex = {'treat overlap False':[3.74E+00,3.70E+00,3.87E+00,4.28E+00,3.49E+00,3.60E+00],
                  'treat overlap True':[4.44E+00,4.57E+00,4.65E+00,4.71E+00,3.86E+00,4.01E+00]}
molpop_cep_tau_nu0 = {'treat overlap False':[3.53E-01,4.10E-01,9.71E-01,1.90E+00,2.39E-02,3.30E-01],
                      'treat overlap True':[3.30E-01,3.09E-01,8.97E-01,1.81E+00,2.30E-02,3.14E-01]}



datafilepath = general.datafilepath('hcn@hfs.dat')
width_v = (2*np.sqrt(np.log(2))) * Doppler
Tkin = 25
collider_densities = {'H2':1e5/constants.centi**3}
ext_background = helpers.generate_CMB_background()
solid_angle = 1
geometry = 'uniform slab'


#calculate molpop-cep spectra:
hcn = molecule.EmittingMolecule(datafilepath=datafilepath,
                                line_profile_type=line_profile_type,width_v=width_v)
v = np.linspace(-6*width_v,6*width_v,50)
#arbitrary choice of v=0:
nu = hcn.rad_transitions[ref_transitions[2]].nu0*(1-v/constants.c)
molpop_cep_tau = {"treat overlap True":np.zeros_like(nu),"treat overlap False":np.zeros_like(nu)}
molpop_cep_Stot = {"treat overlap True":np.zeros_like(nu),"treat overlap False":np.zeros_like(nu)}
for overlap,tau_nu0s in molpop_cep_tau_nu0.items():
    Texs = molpop_cep_Tex[overlap]
    for trans_index,tau_nu0,Tex in zip(ref_transitions,tau_nu0s,Texs):
        trans = hcn.rad_transitions[trans_index]
        phi_nu = trans.line_profile.phi_nu(nu)
        tau = tau_nu0*phi_nu/np.max(phi_nu)
        molpop_cep_tau[overlap] += tau
        molpop_cep_Stot[overlap] += helpers.B_nu(nu=nu,T=Tex)*tau
molpop_cep_spectrum = {}
for overlap,tau in molpop_cep_tau.items():
    molpop_cep_Stot[overlap] /= tau
    assert geometry == "uniform slab"
    intensity = escape_probability.UniformSlab.intensity(
                          tau_nu=tau,source_function=molpop_cep_Stot[overlap])
    molpop_cep_spectrum[overlap] = intensity*solid_angle

colors = {"pythonradex":"blue","molpop":"red"}
linestyles = {"treat overlap True":"solid","treat overlap False":"dashed"}
for treat_line_overlap in (True,False):
    fig,ax = plt.subplots()
    key = f"treat overlap {treat_line_overlap}"
    print(f'treat_line_overlap={treat_line_overlap}')
    ax.set_title(key)
    source = radiative_transfer.Source(
                          datafilepath=datafilepath,geometry=geometry,
                          line_profile_type=line_profile_type,width_v=width_v,
                          treat_line_overlap=treat_line_overlap,warn_negative_tau=False)
    for i in ref_transitions:
        assert source.emitting_molecule.any_line_has_overlap(line_indices=[i,])
    N = column_densities[key]
    source.update_parameters(N=N,Tkin=Tkin,collider_densities=collider_densities,
                            ext_background=ext_background,T_dust=0,tau_dust=0)
    source.solve_radiative_transfer()
    for counter,i in enumerate(ref_transitions):
        Tex_molpop = molpop_cep_Tex[key][counter]
        tau_molpop = molpop_cep_tau_nu0[key][counter]
        print(f'trans {i}:')
        print(f'Tex={source.Tex[i]:.3g} K (molpop: {Tex_molpop:.3g} K)')
        print(f'tau_nu0={source.tau_nu0_individual_transitions[i]:.3g} (molpop: {tau_molpop:.3g})')
    spec = source.spectrum(solid_angle=solid_angle,nu=nu)
    ax.plot(v/constants.kilo,spec,label="pythonradex",linestyle=linestyles[key],
            color=colors["pythonradex"])
    ax.plot(v/constants.kilo,molpop_cep_spectrum[key],label="molpop",
            linestyle=linestyles[key],color=colors["molpop"])
    ax.legend(loc="best")
    print('\n')