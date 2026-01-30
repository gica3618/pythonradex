#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 10:35:28 2026

@author: gianni
"""

# RADEX converts optical depth and flux from rectangular to Gaussian using
# a conversion factor sqrt(pi)/(2*sqrt(ln2))
# however, this is exact to convert a rectangular to Gaussian, but in optically
# thick case, the spectrum is not Gaussian (while optical depth always is)
# let's check the difference

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

width_v = 1*constants.kilo
sigma_v = width_v/(2*np.sqrt(2*np.log(2)))
v = np.linspace(-3*width_v,3*width_v,150)
nu0 = 100*constants.giga
width_nu = width_v/constants.c*nu0
sigma_nu = sigma_v/constants.c*nu0
nu = nu0*(1-v/constants.c)
tau_nu0 = np.logspace(-2,3,50)
tau = tau_nu0[:,None]*np.exp(-(nu[None,:]-nu0)**2/(2*sigma_nu**2))


flux_density = (1-np.exp(-tau))

#check that even for the highest optical depth, the v grid is large enough:
fig,ax = plt.subplots()
ax.plot(v/constants.kilo, flux_density[-1,:])
ax.set_title("most optically thick model")
ax.set_xlabel("v [km/s]")
ax.set_ylabel("flux density")

flux = -np.trapezoid(flux_density,nu,axis=1)
flux_RADEX = (1-np.exp(-tau_nu0))*width_nu*np.sqrt(np.pi)/(2*np.sqrt(np.log(2)))

ratio = flux_RADEX/flux

fig,ax = plt.subplots()
ax.plot(tau_nu0,ratio)
ax.set_xscale("log")
ax.set_xlabel("optical depth")
ax.set_ylabel("RADEX flux / true flux")