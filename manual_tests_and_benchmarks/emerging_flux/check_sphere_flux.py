#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 18:46:27 2024

@author: gianni
"""

#Here I compare the flux computed by pythonradex to the flux computed with the alternative
#formula where basically A21 is replaced by A21*beta (see Elitzur92, formula 2.6.4)

import sys
sys.path.append('..')
import general
from pythonradex import molecule,atomic_transition,escape_probability,\
             escape_probability_functions,helpers
from scipy import constants
import numpy as np
import matplotlib.pyplot as plt


datafilepath = general.datafilepath('co.dat')
line_profile_type = 'rectangular' #this has to be rectangular for the LVG sphere!
width_v = 1*constants.kilo
T = 50
trans_index = 3

r = 1*constants.au
d = 1*constants.parsec
n_values = np.logspace(-1,5,100)/constants.centi**3

mol = molecule.EmittingMolecule(
            datafilepath=datafilepath,line_profile_type=line_profile_type,width_v=width_v)
level_pop = mol.Boltzmann_level_population(T=T)
trans = mol.rad_transitions[trans_index]
width_nu = width_v/constants.c*trans.nu0
nu = np.linspace(trans.nu0-2*width_nu,trans.nu0+2*width_nu,500)
phi_nu = trans.line_profile.phi_nu(nu)
volume = 4/3*r**3*np.pi
solid_angle = r**2*np.pi/d**2

class LVGSphere1D(escape_probability.Flux1D):
    '''LVG sphere with the wrong flux calculation'''
    pass

beta_funcs = {'static sphere':escape_probability_functions.beta_static_sphere,
              'static sphere RADEX':escape_probability_functions.beta_static_sphere,
              'LVG sphere':escape_probability_functions.beta_LVG_sphere,
              'LVG sphere 1Dflux':escape_probability_functions.beta_LVG_sphere,
              'LVG sphere RADEX':escape_probability_functions.beta_LVG_sphere_RADEX}
geometries = {'static sphere':escape_probability.StaticSphere(),
              'static sphere RADEX':escape_probability.StaticSphereRADEX(),
              'LVG sphere':escape_probability.LVGSphere(),
              'LVG sphere 1Dflux':LVGSphere1D(),
              'LVG sphere RADEX':escape_probability.LVGSphereRADEX()}

pythonradex_fluxes = {ID:np.empty(n_values.size) for ID in beta_funcs.keys()}
beta_fluxes = {ID:np.empty(n_values.size) for ID in beta_funcs.keys()}

for i,n in enumerate(n_values):
    N1 = n*level_pop[trans.low.index]*2*r
    N2 = n*level_pop[trans.up.index]*2*r
    tau_nu = atomic_transition.tau_nu(
               A21=trans.A21,phi_nu=phi_nu,
               g_low=trans.low.g,g_up=trans.up.g,N1=N1,N2=N2,nu=nu)
    source_func = helpers.B_nu(nu=nu,T=T)
    intensity_kwargs = {'tau_nu':tau_nu,'source_function':source_func}
    LVG_sphere_kwargs = {'nu':nu,'nu0':trans.nu0,'V':width_v/2}
    flux_no_beta = volume*n*level_pop[trans.up.index]*trans.A21*trans.Delta_E\
                       /(4*np.pi*d**2) * phi_nu #W/m2/Hz
    if i == 0:
        assert np.max(tau_nu) < 0.01
        thin_flux = np.trapezoid(flux_no_beta,nu)
    for ID,beta_func in beta_funcs.items():
        beta_nu = beta_func(tau_nu=tau_nu)
        beta_fluxes[ID][i] = np.trapezoid(flux_no_beta*beta_nu,nu)
        geo = geometries[ID]
        if ID == 'LVG sphere':
            specific_intensity_pythonradex = geo.specific_intensity(
                                **intensity_kwargs,**LVG_sphere_kwargs)
        else:
            specific_intensity_pythonradex = geo.specific_intensity(**intensity_kwargs)
        pythonradex_fluxes[ID][i] = np.trapezoid(specific_intensity_pythonradex,nu)*solid_angle#W/m2

for ID in beta_funcs.keys():
    fig,ax = plt.subplots()
    ax.set_title(ID)
    ax.plot(n_values/constants.centi**-3,pythonradex_fluxes[ID],label='pythonradex')
    ax.plot(n_values/constants.centi**-3,beta_fluxes[ID],label='flux from beta',
            linestyle='dashed')
    ax.axhline(thin_flux,linestyle='dashed',color='black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('number density [cm-3]')
    ax.set_ylabel('flux')
    ax.legend(loc='best')
    flux_ratio = pythonradex_fluxes[ID]/beta_fluxes[ID]
    max_flux_ratio = np.max(flux_ratio)
    print(f'{ID}: max flux ratio: {max_flux_ratio:.3g}')