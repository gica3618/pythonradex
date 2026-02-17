#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:23:14 2024

@author: gianni
"""

from scipy import constants
import numpy as np
from pythonradex import escape_probability, atomic_transition
import matplotlib.pyplot as plt

nu0 = 200 * constants.giga
width_v = 1 * constants.kilo

width_nu = width_v / constants.c * nu0
width_sigma = width_nu / np.sqrt(8 * np.log(2))
nu = np.linspace(nu0 - 1.5 * width_nu, nu0 + 1.5 * width_nu, 100)
tau_nu0_values = np.logspace(-2, 3, 20)

esc_probs = {
    "static sphere": escape_probability.StaticSphere(),
    "static sphere RADEX": escape_probability.StaticSphereRADEX(),
    "static slab": escape_probability.StaticSlab(),
    "LVG slab": escape_probability.LVGSlab(),
    "LVG sphere": escape_probability.LVGSphere(),
    "LVG sphere RADEX": escape_probability.LVGSphereRADEX(),
}

line_profile = atomic_transition.GaussianLineProfile(nu0=nu0, width_v=width_v)

for esc_prob_name, esc_prob in esc_probs.items():
    beta_nu0 = np.empty(tau_nu0_values.size)
    beta_averaged = np.empty_like(beta_nu0)
    fig, axes = plt.subplots(2)
    fig.suptitle(esc_prob_name)
    for i, tau_nu0 in enumerate(tau_nu0_values):

        def beta_func(nu):
            phi_nu = line_profile.phi_nu(nu)
            tau = np.atleast_1d(phi_nu / np.max(phi_nu) * tau_nu0)
            return esc_prob.beta(tau)

        beta_nu0[i] = beta_func(nu=nu0)[0]
        beta_averaged[i] = line_profile.average_over_phi_nu(beta_func)
    relative_diff = np.abs((beta_nu0 - beta_averaged) / beta_nu0)
    axes[0].plot(tau_nu0_values, beta_nu0, label="beta nu0")
    axes[0].plot(tau_nu0_values, beta_averaged, label="beta averaged")
    axes[0].set_ylabel("beta")
    axes[0].legend(loc="best")
    axes[1].plot(tau_nu0_values, relative_diff)
    axes[1].set_ylabel("relative diff beta_nu0 vs averaged_beta")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("tau_nu0")
