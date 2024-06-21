#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:23:14 2024

@author: gianni
"""

from scipy import constants
import numpy as np
import sys
sys.path.append('/home/gianni/science/projects/code/pythonradex')
from pythonradex import escape_probability
import matplotlib.pyplot as plt

nu0 = 200*constants.giga
width_v = 1*constants.kilo

width_nu = width_v/constants.c*nu0
width_sigma = width_nu/np.sqrt(8*np.log(2))
nu = np.linspace(nu0-1.5*width_nu,nu0+1.5*width_nu,100)
tau_nu0_values = np.logspace(-2,3,20)

esc_probs = {'uniform sphere':escape_probability.UniformSphere(),
             'uniform sphere RADEX':escape_probability.UniformSphereRADEX(),
             'uniform slab':escape_probability.UniformSlab(),
             'LVG slab':escape_probability.UniformLVGSlab(),
             'LVG sphere':escape_probability.UniformLVGSphere(),
             'LVG sphere RADEX':escape_probability.LVGSphereRADEX()}

phi_nu_norm = 1/(np.sqrt(2*np.pi)*width_sigma)
phi_nu = np.exp(-(nu-nu0)**2/(2*width_sigma**2)) * phi_nu_norm
#print(np.trapz(phi_nu,nu))

for esc_prob_name,esc_prob in esc_probs.items():
    beta_nu0 = np.empty(tau_nu0_values.size)
    beta_averaged = np.empty_like(beta_nu0)
    fig,axes = plt.subplots(2)
    fig.suptitle(esc_prob_name)
    for i,tau_nu0 in enumerate(tau_nu0_values):
        tau_nu = tau_nu0*phi_nu/phi_nu_norm
        beta_nu0[i] = esc_prob.beta(np.array((tau_nu0,)))
        beta_tau = esc_prob.beta(tau_nu)
        beta_averaged[i] = np.trapz(beta_tau*phi_nu,nu)
    relative_diff = np.abs((beta_nu0-beta_averaged)/beta_nu0)
    axes[0].plot(tau_nu0_values,beta_nu0,label='beta nu0')
    axes[0].plot(tau_nu0_values,beta_averaged,label='beta averaged')
    axes[0].set_ylabel('beta')
    axes[0].legend(loc='best')
    axes[1].plot(tau_nu0_values,relative_diff)
    axes[1].set_ylabel('relative diff beta_nu0 vs averaged_beta')
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('tau_nu0')
    