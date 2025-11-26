#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:10:58 2024

@author: gianni
"""

#RADEX uses I=S(1-exp(-tau)) even for a sphere, while I use the formula from
#Osterbrock (for static sphere), and another formular for LVG sphere.
#The two formulae give different results
#particularly for the optically thin case (see compare_emerging_flux_formulae.py)
#here I test explicitly that the Osterbrock formula indeed gives the flux expected
#in the optically thin case


import numpy as np
from scipy import constants
import sys
sys.path.append('..')
import general
from pythonradex import molecule,atomic_transition,escape_probability,helpers
sys.path.append('../RADEX_wrapper')
import radex_wrapper

r = 1*constants.au
d = 1*constants.parsec
n = 20/constants.centi**3
width_v = 1*constants.kilo
#actually RADEX does calculation using rectangular, but then applies correction
#factor for Gaussian... but fortunately, for optically thin case, it doesn't matter
line_profile_type = 'rectangular'
Tex = 30
trans_index = 1
#make it LTE for the RADEX wrapper so that Tkin=Tex
coll_partner_densities = {'para-H2':1e9/constants.centi**3}
frequency_interval = radex_wrapper.Interval(min=200*constants.giga,
                                            max=240*constants.giga)

datafilepath = general.datafilepath('co.dat')

mol = molecule.EmittingMolecule(
            datafilepath=datafilepath,line_profile_type=line_profile_type,
            width_v=width_v)
level_pop = mol.LTE_level_pop(T=Tex)
trans = mol.rad_transitions[trans_index]
N1 = n*level_pop[trans.low.number]*2*r
N2 = n*level_pop[trans.up.number]*2*r
width_nu = width_v/constants.c*trans.nu0
nu = np.linspace(trans.nu0-2*width_nu,trans.nu0+2*width_nu,500)
phi_nu = trans.line_profile.phi_nu(nu)
tau_nu = atomic_transition.tau_nu(
           A21=trans.A21,phi_nu=phi_nu,
           g_low=trans.low.g,g_up=trans.up.g,N1=N1,N2=N2,nu=nu)
print(f'max tau nu: {np.max(tau_nu):.3g}')

volume = 4/3*r**3*np.pi
solid_angle = r**2*np.pi/d**2
#W/m2
thin_flux = volume*n*level_pop[trans.up.number]*trans.A21*trans.Delta_E/(4*np.pi*d**2)
print(f"analytical flux: {thin_flux:.3g} W/m2")
source_func = helpers.B_nu(nu=nu,T=Tex)

geometries = {"uniform sphere":escape_probability.UniformSphere(),
              "lvg sphere":escape_probability.UniformLVGSphere(),
              "uniform sphere RADEX":escape_probability.UniformSphereRADEX(),
              "lvg sphere RADEX":escape_probability.LVGSphereRADEX()}
flux_kwargs_template = {'tau_nu':tau_nu,'source_function':source_func,
                        'solid_angle':solid_angle}
flux_kwargs = {geo_name:flux_kwargs_template.copy() for geo_name in
               geometries.keys()}
flux_kwargs["lvg sphere"]["nu"] = nu
flux_kwargs["lvg sphere"]["nu0"] = trans.nu0
flux_kwargs["lvg sphere"]["V"] = width_v/2

for geo_name,geo in geometries.items():
    flux = geo.compute_flux_nu(**flux_kwargs[geo_name])
    flux = np.trapezoid(flux,nu)
    print(f"{geo_name}: {flux:.3g} W/m2")


for radex_geo in ("static sphere","LVG sphere"):
    radex_input = radex_wrapper.RadexInput(
                         data_filename=datafilepath,
                         frequency_interval=frequency_interval,Tkin=Tex,
                         coll_partner_densities=coll_partner_densities,
                         T_background=0,column_density=n*2*r,
                         Delta_v=width_v)
    radex_wrap = radex_wrapper.RadexWrapper(geometry=radex_geo)
    radex_wrap.compute(radex_input)
    output = radex_wrap.compute(radex_input)
    #turns out to calculate the flux from the antenna temperature, RADEX simply
    #integrates the antenna temperature intensity over all solid angles (not sure why...)
    #so need to scale by the solid angle of the source
    #see line 334 in io.f
    #also note that RADEX does some correction to convert from rectangular
    #to Gaussian. but doesn't matter since we are looking at optically thin
    #case here
    radex_wrapper_flux = output['flux']*solid_angle/(4*np.pi)
    print(f"radex wrapper flux ({radex_geo}): {radex_wrapper_flux:.3g} W/m2")