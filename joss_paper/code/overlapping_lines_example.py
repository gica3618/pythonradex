#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 20:35:38 2025

@author: gianni
"""

from pythonradex import radiative_transfer,helpers,molecule,escape_probability
from scipy import constants
import matplotlib.pyplot as plt
import numpy as np

save_figure = True

datafilepath = '../../tests/LAMDA_files/hcn@hfs.dat'
geometry = 'uniform slab'
line_profile_type = 'Gaussian'
Doppler = 1.502*constants.kilo
Tkin = 25
nH2 = 1.0e5/constants.centi**3
ext_background = helpers.generate_CMB_background()
#molpop cep gives column densities in cm-2/(km/s), so need to convert
N = 2.11E+14*constants.centi**-2/constants.kilo * Doppler


width_v = Doppler*2*np.sqrt(np.log(2))
cloud_kwargs = {'datafilepath':datafilepath,'geometry':geometry,
                'line_profile_type':line_profile_type,'width_v':width_v,
                'warn_negative_tau':False}
reference_cloud_parameters = {'N':N,'Tkin':Tkin,'collider_densities':{'H2':nH2},
                              'ext_background':ext_background,'T_dust':0,
                              'tau_dust':0}
clouds = {ID:radiative_transfer.Cloud(**cloud_kwargs,treat_line_overlap=treat)
          for ID,treat in zip(('treat overlap','ignore overlap'),(True,False))}

line_indices = [3,4,5,6,7,8]

#from the *.out files:
molpop_cep_tau_nu0 = {"treat overlap":[2.23E+00,1.70E+00,6.05E+00,1.20E+01,1.80E-01,2.50E+00],
                      "ignore overlap":[3.03E+00,3.50E+00,7.98E+00,1.43E+01,2.18E-01,2.90E+00]}
molpop_cep_Tex = {"treat overlap":[8.22E+00,8.55E+00,8.75E+00,8.83E+00,5.56E+00,5.80E+00],
                  "ignore overlap":[5.41E+00,4.86E+00,5.79E+00,7.74E+00,4.39E+00,4.64E+00]}
#sanity check: put molpop cep same as pythonradex for ignore overlap
# molpop_cep_tau_nu0 = {"treat overlap":[2.23E+00,1.70E+00,6.05E+00,1.20E+01,1.80E-01,2.50E+00],
#                       "ignore overlap":[2.6,3.1,6.7,11,0.19,2.5]}
# molpop_cep_Tex = {"treat overlap":[8.22E+00,8.55E+00,8.75E+00,8.83E+00,5.56E+00,5.80E+00],
#                   "ignore overlap":[6.51,5.6,7.08,10.3,4.97,5.3]}

for ID,cloud in clouds.items():
    print(ID)
    cloud.update_parameters(**reference_cloud_parameters)
    cloud.solve_radiative_transfer()
    for i,index in enumerate(line_indices):
        print(f'line {index}:')
        print(f'tau_nu0 = {cloud.tau_nu0_individual_transitions[index]:.2g}, '
              +f'Tex = {cloud.Tex[index]:.3g} K')
        print(f"molpopcep: tau_nu0 = {molpop_cep_tau_nu0[ID][i]:.2g},"
              +f" Tex={molpop_cep_Tex[ID][i]:.2g}")
    print('\n')

nu0s = [clouds['treat overlap'].emitting_molecule.nu0[index] for index in
        line_indices]
nu0s = sorted(nu0s)
#arbitrarily choose one of the rest frequencies to define the velocity grid:
ref_nu0 = nu0s[2]
v = np.linspace(-4*width_v,4*width_v,150)
def get_nu(v):
    return ref_nu0*(1-v/constants.c)
nu = get_nu(v=v)
solid_angle = (100*constants.au)**2/(15*constants.parsec)**2


#calculate molpop-cep spectra:
hcn = molecule.EmittingMolecule(datafilepath=datafilepath,
                                line_profile_type=line_profile_type,width_v=width_v)
molpop_cep_tau = {"treat overlap":np.zeros_like(nu),"ignore overlap":np.zeros_like(nu)}
molpop_cep_Stot = {"treat overlap":np.zeros_like(nu),"ignore overlap":np.zeros_like(nu)}
for overlap,tau_nu0s in molpop_cep_tau_nu0.items():
    Texs = molpop_cep_Tex[overlap]
    for i,tau_nu0,Tex in zip(line_indices,tau_nu0s,Texs):
        trans = hcn.rad_transitions[i]
        phi_nu = trans.line_profile.phi_nu(nu)
        tau = tau_nu0*phi_nu/np.max(phi_nu)
        molpop_cep_tau[overlap] += tau
        molpop_cep_Stot[overlap] += helpers.B_nu(nu=nu,T=Tex)*tau
molpop_cep_spectrum = {}
for overlap,tau in molpop_cep_tau.items():
    molpop_cep_Stot[overlap] /= tau
    assert geometry == "uniform slab"
    molpop_cep_spectrum[overlap] = escape_probability.UniformSlab.compute_flux_nu(
                          tau_nu=tau,source_function=molpop_cep_Stot[overlap],
                          solid_angle=solid_angle)


linestyles = {'treat overlap':'solid','ignore overlap':'dashed'}
colors = {"pythonradex":"blue","molpop-cep":"green"}
fig,ax = plt.subplots()
ax.set_title('HCN hyperfine spectrum around 177.3 GHz')
for ID,cloud in clouds.items():
    spectrum = cloud.spectrum(solid_angle=solid_angle,nu=nu)
    ax.plot(v/constants.kilo,spectrum*1e26,label=ID,linestyle=linestyles[ID],
            color=colors["pythonradex"])
    ax.plot(v/constants.kilo,molpop_cep_spectrum[ID]*1e26,label=f"{ID}\n(MOLPOP-CEP)",
            linestyle=linestyles[ID],color=colors["molpop-cep"])

sigma_v = helpers.FWHM2sigma(FWHM=width_v)
for i,nu0 in enumerate(nu0s):
    v0 = (1-nu0/ref_nu0)*constants.c
    #ax.axvline(v0/constants.kilo,color='black',linestyle='dotted')
    norm_spec = np.exp(-(v-v0)**2/(2*sigma_v)**2)
    if i == 0:
        label = "hyperfine\ncomponents"
    else:
        label = None
    ax.plot(v/constants.kilo,norm_spec,color='black',linestyle='dotted',label=label)
ax.set_xlabel('v [km/s]')
ax.set_ylabel(r'flux [$10^{-26}$W/m$^2$/Hz]')
ax.legend(loc='best')

if save_figure:
    plt.savefig("HCN_spec.pdf",format="pdf",bbox_inches="tight") 