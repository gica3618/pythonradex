#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:03:58 2021

@author: cataldi
"""

from scipy import constants
import numpy as np
from pythonradex import radiative_transfer, helpers
import os
import itertools

here = os.path.dirname(os.path.abspath(__file__))
geometries = list(radiative_transfer.Source.geometries.keys())


def test_negative_tau():
    # Previously, I found that Tkin=130, width_v=0.5 km/s, N=[1e24.2,1e25,1e26] and
    # ortho-H2 = 1e4 or 1e1 cm-3 produce negative optical depth that makes the
    # code crash. So here I test if this does not occure anymore
    ncoll_values = np.array((1e1, 1e4)) / constants.centi**3
    Tkin_values = np.array((10, 50, 130, 200))
    width_v_values = np.array([0.1, 0.5]) * constants.kilo
    N_values = np.array([10**24.2, 1e25, 1e26])
    ext_background = helpers.generate_CMB_background()
    for geo in geometries:
        for ncoll, Tkin, width_v, N in itertools.product(
            ncoll_values, Tkin_values, width_v_values, N_values
        ):
            collider_densities = {"ortho-H2": ncoll}
            source = radiative_transfer.Source(
                datafilepath=os.path.join(here, "LAMDA_files/co.dat"),
                geometry=geo,
                line_profile_type="rectangular",
                width_v=width_v,
            )
            source.update_parameters(
                ext_background=ext_background,
                N=N,
                Tkin=Tkin,
                collider_densities=collider_densities,
                T_dust=0,
                tau_dust=0,
            )
            source.solve_radiative_transfer()
