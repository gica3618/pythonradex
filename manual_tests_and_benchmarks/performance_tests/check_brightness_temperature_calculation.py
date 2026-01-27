#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 21:36:52 2025

@author: gianni
"""

from pythonradex import helpers, radiative_transfer
import time
from scipy import constants
import sys

sys.path.append("..")
import general
import numpy as np


geometry = "static sphere"
line_profile_type = "rectangular"
width_v = 1 * constants.kilo
use_Ng_acceleration = True
ext_background = helpers.generate_CMB_background(z=0)
Tkin = 49
datafilepath = general.datafilepath("co.dat")
N = 1e17 / constants.centi**2
collider_densities = {"para-H2": 1e3 / constants.centi**3}


source = radiative_transfer.Source(
    datafilepath=datafilepath,
    geometry=geometry,
    line_profile_type=line_profile_type,
    width_v=width_v,
    use_Ng_acceleration=use_Ng_acceleration,
)
source.update_parameters(
    ext_background=ext_background,
    N=N,
    Tkin=Tkin,
    collider_densities=collider_densities,
    T_dust=0,
    tau_dust=0,
)

# warm up
source.solve_radiative_transfer()
start = time.time()
source.solve_radiative_transfer()
end = time.time()
print(f"solve time: {end-start:.3g}")
v = np.linspace(-3 * width_v, 3 * width_v, 50)
nu = source.emitting_molecule.rad_transitions[1].nu0 * v / constants.c
source.spectrum(solid_angle=1, nu=nu)

# the fast method works only if there is no dust and no overlapping lines (i.e.
# source function is just B_nu(Tex)))
start = time.time()
nu0 = source.emitting_molecule.nu0
source_function = helpers.B_nu(nu=nu0, T=source.Tex)
intensity = source.geometry.intensity(
    tau_nu=source.tau_nu0_individual_transitions, source_function=source_function
)
T_RJ_fast = intensity * constants.c**2 / (2 * nu0**2 * constants.k)
end = time.time()
fast_time = end - start
print(f"fast method: {fast_time:.3g}")

# the method I implemented
# warm up
T_RJ_imp = source.brightness_temperature_nu0(
    transitions=np.arange(source.emitting_molecule.n_rad_transitions),
    temperature_type="Rayleigh-Jeans",
)
start = time.time()
T_RJ_imp = source.brightness_temperature_nu0(
    transitions=np.arange(source.emitting_molecule.n_rad_transitions),
    temperature_type="Rayleigh-Jeans",
)
end = time.time()
imp_time = end - start
print(f"implemented: {imp_time:.3g}")
print(f"imp/fast time ratio: {imp_time/fast_time:.3g}")

start = time.time()
mock_solid_angle = 1
nu0 = source.emitting_molecule.nu0
intensity = (
    source.spectrum(solid_angle=mock_solid_angle, nu=nu0) / mock_solid_angle
)  # W/m2/Hz/sr
T_RJ_lazy = intensity * constants.c**2 / (2 * nu0**2 * constants.k)
end = time.time()
lazy_time = end - start
print(f"lazy method: {lazy_time:.3g}")

assert np.allclose(T_RJ_fast, T_RJ_lazy, rtol=1e-4, atol=0)

print(f"lazy/fast time ratio: {lazy_time/fast_time:.3g}")
