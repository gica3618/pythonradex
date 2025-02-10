#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:10:58 2024

@author: gianni
"""

#RADEX uses I=S(1-exp(-tau)) even for a sphere, while I use the formula from
#Osterbrock. The two formulae give different results particularly for the optically
#thin case (see compare_emerging_flux_formulae.py)
#here I test explicitly that the Osterbrock formula indeed gives the flux expected
#in the optically thin case

import numpy as np
from scipy import constants
import sys
sys.path.append('../../src')
from pythonradex import molecule,atomic_transition,escape_probability,helpers
sys.path.append('/home/gianni/science/projects/code/RADEX_wrapper')
import radex_wrapper

r = 1*constants.au
d = 1*constants.parsec
n = 20/constants.centi**3
width_v = 1*constants.kilo
line_profile_type = 'Gaussian'
Tex = 30
trans_index = 1
#make it LTE for the RADEX wrapper so that Tkin=Tex
coll_partner_densities = {'para-H2':1e9/constants.centi**3}
frequency_interval = radex_wrapper.Interval(min=200*constants.giga,
                                            max=240*constants.giga)

datafilepath = '../../tests/LAMDA_files/co.dat'

mol = molecule.EmittingMolecule(
            datafilepath=datafilepath, line_profile_type=line_profile_type,
            width_v=width_v)
level_pop = mol.LTE_level_pop(T=Tex)
trans = mol.rad_transitions[trans_index]
N1 = n*level_pop[trans.low.number]*2*r
N2 = n*level_pop[trans.up.number]*2*r
width_nu = width_v/constants.c*trans.nu0
nu = np.linspace(trans.nu0-2*width_nu,trans.nu0+2*width_nu,500)
phi_nu = trans.line_profile.phi_nu(nu)
tau_nu = atomic_transition.fast_tau_nu(
           A21=trans.A21,phi_nu=phi_nu,
           g_low=trans.low.g,g_up=trans.up.g,N1=N1,N2=N2,nu=nu)
print(f'max tau nu: {np.max(tau_nu):.3g}')

volume = 4/3*r**3*np.pi
solid_angle = r**2*np.pi/d**2
#W/m2
thin_flux = volume*n*level_pop[trans.up.number]*trans.A21*trans.Delta_E/(4*np.pi*d**2)
uniform_sphere = escape_probability.UniformSphere()
source_func = helpers.B_nu(nu=nu,T=Tex)
flux_kwargs = {'tau_nu':tau_nu,'source_function':source_func,'solid_angle':solid_angle}
flux_pythonradex = uniform_sphere.compute_flux_nu(**flux_kwargs)
flux_pythonradex = np.trapz(flux_pythonradex,nu)
flux_0D = escape_probability.Flux1D()
flux_RADEX = flux_0D.compute_flux_nu(**flux_kwargs)
flux_RADEX = np.trapz(flux_RADEX,nu)

radex_input = radex_wrapper.RadexInput(
                     data_filename=datafilepath,
                     frequency_interval=frequency_interval,Tkin=Tex,
                     coll_partner_densities=coll_partner_densities,
                     T_background=0,column_density=n*2*r,
                     Delta_v=width_v)
radex_wrap = radex_wrapper.RadexWrapper()
radex_wrap.compute(radex_input)
output = radex_wrap.compute(radex_input)
#turns out to calculate the flux from the antenna temperature, RADEX simply
#integrates the antenna temperature intensity over all solid angles (not sure why...)
#so need to scale by the solid angle of the source
#see line 334 in io.f
radex_wrapper_flux = output['flux']*solid_angle/(4*np.pi)

for ID,flux in {'thin analytical':thin_flux,'pythonradex formula':flux_pythonradex,
                'RADEX formula':flux_RADEX,'RADEX wrapper':radex_wrapper_flux}.items():
    print(f'flux {ID}: {flux:.3g} W/m2')