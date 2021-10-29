#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:03:58 2021

@author: cataldi
"""

from scipy import constants
import numpy as np
from pythonradex import nebula,helpers
import os

here = os.path.dirname(os.path.abspath(__file__))
geometries = list(nebula.Nebula.geometries.keys())

def test_negative_tau():
    #Previously, I found that Tkin=130, width_v=0.5 km/s, Ntot=1e25 and
    #ortho-H2 = 1e4 or 1e1 cm-3 produce negative optical depth that makes the
    #code crash. So here I test if this does not occure anymore
    ncoll_values = np.array((1e1,1e4))/constants.centi**3
    Tkin = 130
    width_v = 0.5*constants.kilo
    Ntot = 1e26
    ext_background = helpers.generate_CMB_background()
    for geo in geometries:
        for ncoll in ncoll_values:
            coll_partner_densities = {'ortho-H2':ncoll}
            test_nebula = nebula.Nebula(
                            datafilepath=os.path.join(here,'co.dat'),
                            geometry=geo,ext_background=ext_background,Tkin=Tkin,
                            coll_partner_densities=coll_partner_densities,
                            Ntot=Ntot,line_profile='square',width_v=width_v)
            test_nebula.solve_radiative_transfer()