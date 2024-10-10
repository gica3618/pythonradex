#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:10:41 2024

@author: gianni
"""

#here I check the case with a single line and dust
#dust can be treated like an overlapping line, so in principle need to average
#over the line profile. but how much is the difference if I just use the values
#at the line center?


import sys
sys.path.append('/home/gianni/science/projects/code/pythonradex')
from pythonradex import escape_probability,atomic_transition,helpers
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants

Tdust = 100
Iext = helpers.generate_CMB_background()
tau_nu0_values = np.logspace(-2,3,20)
#tau_dust_values = tau_nu0_values.copy()
tau_dust_values = np.logspace(-3,4,30)

esc_probs = {'uniform sphere':escape_probability.UniformSphere(),
             'uniform sphere RADEX':escape_probability.UniformSphereRADEX(),
             'uniform slab':escape_probability.UniformSlab(),
             'LVG slab':escape_probability.UniformLVGSlab(),
             'LVG sphere':escape_probability.UniformLVGSphere(),
             'LVG sphere RADEX':escape_probability.LVGSphereRADEX()}

#for rectangular profile, optical depth of both gas and dust do not change
#over the line profile, so averaging is not necessary. so I only check Gaussian
nu0 = 100*constants.giga
width_v = 1*constants.kilo
line_profile = atomic_transition.GaussianLineProfile(nu0=nu0,width_v=width_v)

TAU,TAU_D = np.meshgrid(tau_nu0_values,tau_dust_values,indexing='ij')

for esc_prob_name,esc_prob in esc_probs.items():
    #consider the factors appearing in the ALI equations for overlapping lines
    def A21_factor(nu,tau_gas,tau_dust):
        tau_g = tau_gas(nu)
        tau_tot = np.atleast_1d(tau_g+tau_dust)
        beta = esc_prob.beta(tau_tot)
        return 1-(1-beta)*tau_g/tau_tot
    
    def B_factor(nu,tau_gas,tau_dust):
        tau_g = tau_gas(nu)
        tau_tot = np.atleast_1d(tau_g+tau_dust)
        beta = esc_prob.beta(tau_tot)
        S_dust = helpers.B_nu(nu=nu,T=Tdust)
        K = tau_dust*S_dust/tau_tot
        return beta*Iext(nu) + (1-beta)*K
    
    computed_A21_factors = {key:np.empty_like(TAU) for key in ('nu0','averaged')}
    computed_B_factors = {key:np.empty_like(TAU) for key in ('nu0','averaged')}
    for i in range(tau_nu0_values.size):
        for j in range(tau_dust_values.size):
            tau_dust = tau_dust_values[j]
            def tau_gas(nu):
                phi_nu = line_profile.phi_nu(nu)
                return phi_nu/np.max(phi_nu) * tau_nu0_values[i]
            def A(nu):
                return A21_factor(nu=nu,tau_gas=tau_gas,tau_dust=tau_dust)
            def B(nu):
                return B_factor(nu=nu,tau_gas=tau_gas,tau_dust=tau_dust)
            computed_A21_factors['nu0'][i,j] = A(nu=nu0)
            computed_A21_factors['averaged'][i,j]\
                             = line_profile.average_over_phi_nu(A)
            computed_B_factors['nu0'][i,j] = B(nu=nu0)
            computed_B_factors['averaged'][i,j]\
                                  = line_profile.average_over_phi_nu(B)
    for ID,factors in {'A21_factor':computed_A21_factors,
                       'B_factor':computed_B_factors}.items():
        differences = {'relative diff':helpers.relative_difference(factors['nu0'],factors['averaged']),
                       'abs diff':np.abs(factors['nu0']-factors['averaged'])}
        fig,axes = plt.subplots(2,2)
        fig.suptitle(f'{ID} ({esc_prob_name})')
        for i,mode in enumerate(('nu0','averaged')):
            ax = axes[0,i]
            im = ax.pcolormesh(TAU,TAU_D,factors[mode])
            fig.colorbar(im,ax=ax)
            ax.set_title(mode)
        for i,(diffID,diff) in enumerate(differences.items()):
            ax = axes[1,i]
            ax.set_title(diffID)
            im = ax.pcolormesh(TAU,TAU_D,diff)
            fig.colorbar(im,ax=ax)
            ax.set_title(diffID)
        for ax in axes.ravel():
            ax.set_xscale('log')
            ax.set_yscale('log')