# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:09:51 2017

@author: gianni
"""

import os
from pythonradex import nebula,helpers
from scipy import constants
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
filenames = ['co.dat','hcl.dat','ocs@xpol.dat']
collider_names = ['ortho-H2','para-H2','H2']

line_profile = 'square'
ext_background = helpers.generate_CMB_background()

Ntot_values = np.array((1e16,1e14,1e20))/constants.centi**2
width_v_values = np.array((0.5,1,3))*constants.kilo
Tkin_values = np.array((50,100,200))
coll_partner_densities_values = np.array((1e2,1e4,1e8))/constants.centi**3
test_trans = [0,20,37]

RADEX_sphere = {'co':{'Tex':[7.53,61.111,193.725],'tau':[5.163,1.138e-12,3.582e-05]},
                'hcl':{'Tex':[6.127,19.139,198.726],'tau':[2.693e3,1.265e-4,7.887e1]},
                'ocs@xpol':{'Tex':[-69.517,21.741,199.999],
                            'tau':[-2.160e-1,3.564e-3,1.493e3]}
                }

RADEX_slab = {'co':{'Tex':[12.66,61.112,193.726],'tau':[2.333,1.143e-12,3.651e-5]},
              'hcl':{'Tex':[8.292,19.105,199.714],'tau':[2195,0.0001014,78.61]},
              'ocs@xpol':{'Tex':[51.585,21.691,200],'tau':[0.1952,0.003709,1493]}
              }

RADEX = {'sphere':RADEX_sphere,'slab':RADEX_slab}

radius = 1*constants.au
distance = 1*constants.parsec
slab_surface = (1*constants.au)**2
geometries = list(nebula.Nebula.geometries.keys())
def get_solid_angle(geo):
    if 'sphere' in geo:
        Omega = np.pi*radius**2/distance**2
    elif 'slab' in geo:
        Omega = slab_surface/distance**2
    else:
        raise ValueError('geo: {:s}'.format(geo))
    return Omega


def test_vs_RADEX():
    for filename,coll_ID in zip(filenames,collider_names):
        specie = filename[:-4]
        lamda_filepath = os.path.join(here,filename)
        params = zip(Ntot_values,width_v_values,Tkin_values,
                     coll_partner_densities_values,test_trans)
        for i,(Ntot,width_v,Tkin,collp_dens,trans_num) in enumerate(params):
            coll_partner_densities = {coll_ID:collp_dens}
            for geo in geometries:
                Omega = get_solid_angle(geo)
                print('looking at {:s}, {:s} (case {:d})'.format(specie,geo,i))
                test_nebula = nebula.Nebula(
                            datafilepath=lamda_filepath,geometry=geo,
                            ext_background=ext_background,Tkin=Tkin,
                            coll_partner_densities=coll_partner_densities,
                            Ntot=Ntot,line_profile=line_profile,width_v=width_v)
                test_nebula.solve_radiative_transfer()
                test_nebula.compute_line_fluxes(solid_angle=Omega)
                trans = test_nebula.emitting_molecule.rad_transitions[trans_num]
                if 'sphere' in geo:
                    radex = RADEX['sphere']
                elif 'slab' in geo:
                    radex = RADEX['slab']
                Tex = radex[specie]['Tex'][i]
                tau_nu = radex[specie]['tau'][i]
                #see radex_output_readme.txt or RADEX manual for explanation of the
                #follwing formula
                RADEX_intensity = (helpers.B_nu(nu=trans.nu0,T=Tex)
                                                   -ext_background(trans.nu0))\
                                  * (1-np.exp(-tau_nu))
                RADEX_flux = RADEX_intensity*trans.line_profile.width_nu*Omega
                pyradex_flux = test_nebula.obs_line_fluxes[trans_num]
                print('RADEX flux: {:g}; pyradex flux: {:g}'.format(
                                                   RADEX_flux,pyradex_flux))
                pyradex_tau = test_nebula.tau_nu0[trans_num]
                pyradex_Tex = test_nebula.Tex[trans_num]
                print('RADEX tau: {:g}; pyradex tau: {:g}'.format(tau_nu,pyradex_tau))
                print('RADEX Tex: {:g} K; pyradex Tex: {:g} K'.format(
                                                  Tex,pyradex_Tex))
                if 'RADEX' in geo:
                    atol_flux = 0
                    rtol_flux = 0.3
                    atol_tau = 0.1
                    rtol_tau = 0.1
                    atol_Tex = 1
                    rtol_Tex = 0.2
                else:
                    atol_flux = 0
                    rtol_flux = 0.7
                    atol_tau = 0.1
                    rtol_tau = 0.5
                    atol_Tex = 1
                    rtol_Tex = 0.5
                if pyradex_tau < 0:
                    assert tau_nu < 0
                    assert Tex < 0
                    assert pyradex_Tex < 0
                else:
                    assert np.isclose(pyradex_tau,tau_nu,atol=atol_tau,rtol=rtol_tau)
                    assert np.isclose(RADEX_flux,pyradex_flux,atol=atol_flux,
                                  rtol=rtol_flux)
                    assert np.isclose(Tex,pyradex_Tex,atol=atol_Tex,rtol=rtol_Tex)                    
            print('\n')
        print('\n')