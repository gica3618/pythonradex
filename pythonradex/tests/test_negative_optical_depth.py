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
import itertools

here = os.path.dirname(os.path.abspath(__file__))
geometries = list(nebula.Nebula.geometries.keys())

def test_negative_tau():
    #Previously, I found that Tkin=130, width_v=0.5 km/s, Ntot=[1e24.2,1e25,1e26] and
    #ortho-H2 = 1e4 or 1e1 cm-3 produce negative optical depth that makes the
    #code crash. So here I test if this does not occure anymore
    ncoll_values = np.array((1e1,1e4))/constants.centi**3
    Tkin_values = np.array((10,50,130,200))
    width_v_values = np.array([0.1,0.5])*constants.kilo
    Ntot_values = np.array([10**24.2,1e25,1e26])
    ext_background = helpers.generate_CMB_background()
    for geo in geometries:
        for ncoll,Tkin,width_v,Ntot in itertools.product(ncoll_values,Tkin_values,
                                                         width_v_values,Ntot_values):
            coll_partner_densities = {'ortho-H2':ncoll}
            test_nebula = nebula.Nebula(
                            datafilepath=os.path.join(here,'co.dat'),
                            geometry=geo,ext_background=ext_background,Tkin=Tkin,
                            coll_partner_densities=coll_partner_densities,
                            Ntot=Ntot,line_profile='square',width_v=width_v)
            test_nebula.solve_radiative_transfer()