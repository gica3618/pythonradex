#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 07:37:49 2025

@author: gianni
"""

from scipy import interpolate
import numpy as np
import time
import numba as nb


def beta_sphere(tau_nu):
    return 1.5/tau_nu*(1-2/tau_nu**2+(2/tau_nu+2/tau_nu**2)*np.exp(-tau_nu))

@nb.jit(nopython=True,cache=True)
def beta_sphere_comp(tau_nu):
    return 1.5/tau_nu*(1-2/tau_nu**2+(2/tau_nu+2/tau_nu**2)*np.exp(-tau_nu))

tau_data = np.logspace(-3,3,300)
beta_data = beta_sphere(tau_nu=tau_data)

lin_interp = interpolate.interp1d(x=tau_data,y=beta_data,kind='linear',assume_sorted=True)
near_interp = interpolate.interp1d(x=tau_data,y=beta_data,kind='nearest',assume_sorted=True)

tau = np.logspace(-2,2,42)
np.random.shuffle(tau)
for i in range(5):
    start = time.time()
    out = lin_interp(tau)
    end = time.time()
    print(f'linear: {end-start:.3g}')
    
    start = time.time()
    out = near_interp(tau)
    end = time.time()
    print(f'nearest: {end-start:.3g}')
    
    start = time.time()
    out = beta_sphere(tau)
    end = time.time()
    print(f'direct: {end-start:.3g}')
    
    start = time.time()
    out = beta_sphere_comp(tau)
    end = time.time()
    print(f'compiled: {end-start:.3g}\n')