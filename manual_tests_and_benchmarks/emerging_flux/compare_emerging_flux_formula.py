#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:58:53 2024

@author: gianni
"""

# RADEX uses B(Tex)*(1-exp(-tau)) for the emerging flux even for a static or LVG sphere
# this is not correct, but let's check how big the difference actually is

# assume rectangular line profile, to include LVG sphere

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from pythonradex import helpers

nu0 = 100 * constants.giga
Tex = 30
source_function_nu0 = helpers.B_nu(nu=nu0, T=Tex)
solid_angle = (1 * constants.au / (1 * constants.parsec)) ** 2
width_v = 1 * constants.kilo


width_nu = width_v / constants.c * nu0
V = width_v / 2


def flux_static_sphere(tau_nu0):
    flux_nu0 = (
        2
        * np.pi
        * source_function_nu0
        / tau_nu0**2
        * (tau_nu0**2 / 2 - 1 + (tau_nu0 + 1) * np.exp(-tau_nu0))
        * solid_angle
        / np.pi
    )
    # assume rectangular profile:
    return flux_nu0 * width_nu


def flux_lvg_sphere(tau_nu0):
    # see LVG_sphere.pdf
    return (
        nu0
        / constants.c
        * solid_angle
        * source_function_nu0
        * (1 - np.exp(-tau_nu0))
        * 4
        / 3
        * V
    )


def flux_0D(tau_nu0):
    return source_function_nu0 * (1 - np.exp(-tau_nu0)) * solid_angle * width_nu


tau_values = np.logspace(-2, 2, 100)

pythonradex_fluxes = {
    "static sphere": flux_static_sphere(tau_nu0=tau_values),
    "LVG sphere": flux_lvg_sphere(tau_nu0=tau_values),
}
f_0D = flux_0D(tau_nu0=tau_values)


# in the optically thin case, the ratio should be (volume of cylinder) / (volume of sphere)
# (because the 0D formula corresponds to intensity independent over the emitting area),
# which is  (r**2*pi * 2r) / (4/3 r**3 pi) = 3/2
expected_ratio_thin = 3 / 2


for ID, pythonradex_flux in pythonradex_fluxes.items():
    relative_diff = np.abs((pythonradex_flux - f_0D) / pythonradex_flux)
    ratio = f_0D / pythonradex_flux

    fig, axes = plt.subplots(3)
    fig.suptitle(ID)
    axes[0].plot(tau_values, pythonradex_flux, label="pythonradex")
    axes[0].plot(tau_values, f_0D, label="0D (RADEX)")
    axes[0].set_ylabel("flux")
    axes[0].legend(loc="best")
    axes[0].set_yscale("log")
    axes[1].plot(tau_values, relative_diff * 100)
    axes[1].set_ylabel("relative difference [%]")
    axes[2].plot(tau_values, ratio)
    axes[2].set_ylabel("f_0D/f_sphere")
    axes[2].axhline(expected_ratio_thin, color="black", linestyle="dashed")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("tau")

#for JOSS paper and documentation:
fig, ax = plt.subplots(figsize=[6.4, 3])
fig.suptitle("Impact of using slab formula to calculate sphere flux")
for geo_name, pythonradex_flux in pythonradex_fluxes.items():
    ax.plot(tau_values, f_0D / pythonradex_flux, label=geo_name)
ax.legend(loc="best")
ax.set_ylabel("flux (slab formula) / flux (sphere formula)")
ax.set_xscale("log")
ax.set_xlabel("optical depth")
fig.tight_layout()
#for the paper:
plt.savefig("../../joss_paper/flux_comparison_spherical_geometries.pdf",
            format="pdf", bbox_inches="tight")
#for the docs:
plt.savefig("../../docs/images/flux_comparison_spherical_geometries.png", format="png",
            bbox_inches="tight")