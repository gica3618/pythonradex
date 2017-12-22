# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:09:51 2017

@author: gianni
"""
#test against RADEX

import os
from pythonradex import nebula,helpers
from scipy import constants
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
filenames = ['co.dat','hcl.dat','ocs@xpol.dat']
collider_names = ['ortho-H2','para-H2','H2']

geometry = 'uniform sphere'
line_profile = 'square'
ext_background = helpers.CMB_background

Ntot_values = np.array((1e16,1e14,1e20))/constants.centi**2
width_v_values = np.array((0.5,1,3))*constants.kilo
Tkin_values = np.array((50,100,200))
coll_partner_densities_values = np.array((1e2,1e4,1e8))/constants.centi**3
test_trans = [0,20,37]

#from RADEX:
RADEX_Tex_CO = [7.015,60.955,193.725]
RADEX_tau_nu_CO = [5.696E+00,1.016E-12,3.581E-05]
RADEX_Tex_HCl = [6.127,19.139,198.726]
RADEX_tau_nu_HCl = [2.693E+03,1.265E-04,7.887E+01]
RADEX_Tex_OCS = [-69.517,21.741,199.999]
RADEX_tau_nu_OCS = [-2.160E-01,3.564E-03,1.493E+03]

RADEX_Tex = [RADEX_Tex_CO,RADEX_Tex_HCl,RADEX_Tex_OCS]
RADEX_tau_nu = [RADEX_tau_nu_CO,RADEX_tau_nu_HCl,RADEX_tau_nu_OCS]

radius = 1
distance = radius
surface = 4*np.pi*radius**2

def test_vs_RADEX():
    for filename,coll_ID,Tex_values,tau_nu_values in\
                          zip(filenames,collider_names,RADEX_Tex,RADEX_tau_nu):
        lamda_filepath = os.path.join(here,filename)
        params = zip(Ntot_values,width_v_values,Tkin_values,
                     coll_partner_densities_values,test_trans,Tex_values,tau_nu_values)
        for Ntot,width_v,Tkin,collp_dens,trans_num,Tex,tau_nu in params:
            coll_partner_densities = {coll_ID:collp_dens}
            test_nebula = nebula.Nebula(
                        data_filepath=lamda_filepath,geometry=geometry,
                        ext_background=ext_background,Tkin=Tkin,
                        coll_partner_densities=coll_partner_densities,
                        Ntot=Ntot,line_profile=line_profile,width_v=width_v)
            test_nebula.solve_radiative_transfer()
            fluxes = test_nebula.observed_fluxes(
                                  source_surface=surface,d_observer=distance)
            #see radex_output_readme.txt or RADEX manual for explanation of the follwing formula
            trans = test_nebula.emitting_molecule.rad_transitions[trans_num]
            RADEX_intensity = (helpers.B_nu(nu=trans.nu0,T=Tex)- ext_background(trans.nu0))\
                              * (1-np.exp(-tau_nu))
            RADEX_flux = np.pi*RADEX_intensity*trans.line_profile.width_nu
            pyradex_flux = fluxes[trans_num]
            print('RADEX flux: {:g}; pyradex flux: {:g}'.format(RADEX_flux,pyradex_flux))
            pyradex_tau = test_nebula.tau_nu0[trans_num]
            print('RADEX tau: {:g}; pyradex tau: {:g}'.format(tau_nu,pyradex_tau))
            print('RADEX Tex: {:g} K; pyradex Tex: {:g} K'.format(Tex,test_nebula.Tex[trans_num]))
            assert np.isclose(RADEX_flux,pyradex_flux,atol=0,rtol=0.7)
            assert np.isclose(pyradex_tau,tau_nu,atol=0.01,rtol=0.1)
        print('\n')
