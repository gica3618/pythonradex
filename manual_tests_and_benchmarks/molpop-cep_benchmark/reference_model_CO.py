#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:35:49 2025

@author: gianni
"""
import sys

sys.path.append("..")
import general
from pythonradex import radiative_transfer, helpers
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

# calculate some simple CO models to make sure I understand the inputs of molpop-cep

ref_transitions = [1, 2]
Doppler = 1 * constants.kilo
datafilepath = general.datafilepath("co.dat")
n = 1e4 / constants.centi**3
Tkin = 100
ext_background = helpers.generate_CMB_background()
solid_angle = 1

width_v = 2 * np.sqrt(np.log(2)) * Doppler
source = radiative_transfer.Source(
    datafilepath=datafilepath,
    geometry="static slab",
    line_profile_type="Gaussian",
    width_v=width_v,
)

molpop_cep_N = np.array((1e16, 1e17, 1e18, 1e19)) * constants.centi**-2 / constants.kilo

# from the .out file of molpop-cep:
molpop_cep_tau_nu0 = [
    [1.54e-01, 1.09e00, 6.30e00, 4.97e01],
    [4.35e-01, 2.26e00, 1.25e01, 9.76e01],
]
molpop_cep_Tex = [
    [6.89e01, 7.36e01, 9.18e01, 9.85e01],
    [4.18e01, 6.53e01, 8.94e01, 9.82e01],
]

collider_densities = {"para-H2": n / 2, "ortho-H2": n / 2}
for i, N_molpop in enumerate(molpop_cep_N):
    N = N_molpop * Doppler
    print(f"N={N/constants.centi**-2:.1g} cm-2")
    source.update_parameters(
        N=N,
        Tkin=Tkin,
        collider_densities=collider_densities,
        ext_background=ext_background,
        T_dust=0,
        tau_dust=0,
    )
    source.solve_radiative_transfer()
    fig, axes = plt.subplots(2)
    fig.suptitle(f"{N/constants.centi**-2:.2g} cm-2")
    for j, trans_index in enumerate(ref_transitions):
        print(f"trans {trans_index}:")
        print(f"Tex={source.Tex[trans_index]:.3g} K (molpop: {molpop_cep_Tex[j][i]})")
        print(
            f"tau_nu0={source.tau_nu0_individual_transitions[trans_index]}"
            + f" (molpop: {molpop_cep_tau_nu0[j][i]})"
        )
        v = np.linspace(-3 * width_v, 3 * width_v, 100)
        trans = source.emitting_molecule.rad_transitions[trans_index]
        nu = trans.nu0 * (1 - v / constants.c)
        spec = source.spectrum(
            output_type="flux density", solid_angle=solid_angle, nu=nu
        )
        ax = axes[j]
        ax.set_title(f"trans {trans_index}")
        ax.plot(v / constants.kilo, spec, label="pythonradex")
        width_nu = width_v / constants.c * trans.nu0
        sigma_nu = helpers.FWHM2sigma(width_nu)
        molpop_tau_nu = molpop_cep_tau_nu0[j][i] * np.exp(
            -((nu - trans.nu0) ** 2) / (2 * sigma_nu**2)
        )
        molpop_cep_spec = helpers.B_nu(nu=nu, T=molpop_cep_Tex[j][i]) * (
            1 - np.exp(-molpop_tau_nu)
        )
        ax.plot(v / constants.kilo, molpop_cep_spec, label="molpop")
        if j == 0:
            ax.legend(loc="best")
    print("\n")
