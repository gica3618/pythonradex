#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 13:26:05 2026

@author: gianni
"""


# 2026/01/30: copied this script from manual_tests_and_benchmarks/compare_emerging_flux_formula.py
# and modified to produce paper figure

# assume rectangular line profile, to include LVG sphere

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from pythonradex import helpers

save_figure = True


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

fig, ax = plt.subplots(figsize=[6.4, 3])
fig.suptitle("Impact of using slab formula to calculate sphere flux")
for geo_name, pythonradex_flux in pythonradex_fluxes.items():
    ax.plot(tau_values, f_0D / pythonradex_flux, label=geo_name)
ax.legend(loc="best")
ax.set_ylabel("flux (slab formula) / flux (sphere formula)")
ax.set_xscale("log")
ax.set_xlabel("optical depth")
fig.tight_layout()
if save_figure:
    print("saving figure")
    plt.savefig("flux_comparison_spherical_geometries.pdf", format="pdf",
                bbox_inches="tight")
else:
    print("figure will not be saved")