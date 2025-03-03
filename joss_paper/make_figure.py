#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 20:35:38 2025

@author: gianni
"""



from pythonradex import radiative_transfer,helpers
from scipy import constants
import matplotlib.pyplot as plt
import numpy as np

datafilepath = '../tests/LAMDA_files/hcn@hfs.dat'
geometry = 'uniform slab'
line_profile_type = 'Gaussian'
width_v = 2*constants.kilo
cloud_kwargs = {'datafilepath':datafilepath,'geometry':geometry,
                'line_profile_type':line_profile_type,'width_v':width_v,
                'warn_negative_tau':False}
reference_cloud_parameters = {'N':5e14/constants.centi**2,'Tkin':25,
                              'collider_densities':{'H2':1e5/constants.centi**3},
                              'ext_background':0,'T_dust':0,'tau_dust':0}
clouds = {ID:radiative_transfer.Cloud(**cloud_kwargs,treat_line_overlap=treat)
          for ID,treat in zip(('treat overlap','ignore overlap'),(True,False))}

line_indices = [3,4,5,6,7,8]
sigma_v = helpers.FWHM2sigma(FWHM=width_v)


for ID,cloud in clouds.items():
    print(ID)
    cloud.update_parameters(**reference_cloud_parameters)
    cloud.solve_radiative_transfer()
    for i in line_indices:
        print(f'line {i}:')
        print(f'tau_nu0 = {cloud.tau_nu0_individual_transitions[i]:.2g}, '
              +f'Tex = {cloud.Tex[i]:.3g} K')
    print('\n')

nu0s = [clouds['treat overlap'].emitting_molecule.nu0[i] for i in line_indices]
nu0s = sorted(nu0s)
#arbitrarily choose one of the rest frequencies to define the velocity grid:
ref_nu0 = nu0s[2]
v = np.linspace(-4*width_v,4*width_v,150)
def get_nu(v):
    return ref_nu0*(1-v/constants.c)
nu = get_nu(v=v)
solid_angle = (100*constants.au)**2/(15*constants.parsec)**2

linestyles = {'treat overlap':'solid','ignore overlap':'dashed'}
fig,ax = plt.subplots()
ax.set_title('HCN hyperfine spectrum around 177.3 GHz')
for ID,cloud in clouds.items():
    spectrum = cloud.spectrum(solid_angle=solid_angle,nu=nu)
    ax.plot(v/constants.kilo,spectrum*1e26,label=ID,linestyle=linestyles[ID])
ax.legend(loc='best')
for nu0 in nu0s:
    #mark the position of each transition with a vertical line
    v0 = (1-nu0/ref_nu0)*constants.c
    #ax.axvline(v0/constants.kilo,color='black',linestyle='dotted')
    norm_spec = np.exp(-(v-v0)**2/(2*sigma_v)**2)
    ax.plot(v/constants.kilo,norm_spec,color='black',linestyle='dotted')
ax.set_xlabel('v [km/s]')
ax.set_ylabel(r'flux [$10^{-26}$W/m$^2$/Hz]')
plt.savefig("HCN_spec.pdf",format="pdf",bbox_inches="tight") 