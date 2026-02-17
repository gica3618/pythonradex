#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:34:53 2025

@author: gianni
"""

# this is just to check how much CPU pythonradex uses
# in terminal, do something like /usr/bin/time -v python check_pythonradex_CPU.py


from scipy import constants
from pythonradex import radiative_transfer, helpers
import sys

sys.path.append("../..")
import general
import os
import time

# datafilepath = os.path.join(general.lamda_data_folder,"co.dat")
# collider_densities = {"ortho-H2":1e4*constants.centi**-3}
datafilepath = os.path.join(general.lamda_data_folder, "c+.dat")
collider_densities = {"e": 1e4 * constants.centi**-3}

cloud_kwargs = {}
source = radiative_transfer.Source(
    datafilepath=datafilepath,
    geometry="static sphere",
    line_profile_type="Gaussian",
    width_v=1 * constants.kilo,
)
ext_background = helpers.generate_CMB_background()

start = time.time()
for i in range(1000):
    # start_i = time.time()
    Tkin = 30
    N = 1e15 * constants.centi**-2
    source.update_parameters(
        ext_background=ext_background,
        Tkin=Tkin,
        collider_densities=collider_densities,
        N=N,
        T_dust=0,
        tau_dust=0,
    )
    source.solve_radiative_transfer()
    # end_i = time.time()
    # print(f"took {end_i-start_i}")
end = time.time()
print(f"total took {end-start}")
