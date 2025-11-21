#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:58:53 2024

@author: gianni
"""

#RADEX uses B(Tex)*(1-exp(-tau)) for the emerging flux even for a uniform sphere
#but looking at Osterbrock, this is not correct, but let's compare how big the difference
#actually is


import numpy as np
import matplotlib.pyplot as plt

def flux_uniform_sphere(tau_nu,source_function,solid_angle):
    return 2*np.pi*source_function/tau_nu**2\
               *(tau_nu**2/2-1+(tau_nu+1)*np.exp(-tau_nu)) *solid_angle/np.pi

def flux_0D(tau_nu,source_function,solid_angle):
    return source_function*(1-np.exp(-tau_nu))*solid_angle


tau_values = np.logspace(-2,2,100)
source_function = 1
solid_angle = 1

f_sphere = flux_uniform_sphere(tau_nu=tau_values,source_function=source_function,
                               solid_angle=solid_angle)
f_0D = flux_0D(tau_nu=tau_values,source_function=source_function,solid_angle=solid_angle)
relative_diff = np.abs((f_sphere-f_0D)/f_sphere)
ratio = f_sphere/f_0D
#in the optically thin case, the ratio should be (volume of sphere) / (volume of cylinder)
#(because the 0D formula corresponds to intensity independent over the emitting area),
#which is (4/3 r**3 pi) / (r**2*pi * 2r) = 2/3
expected_ratio_thin = 2/3


fig,axes = plt.subplots(3)
axes[0].plot(tau_values,f_sphere,label='sphere')
axes[0].plot(tau_values,f_0D,label='0D (RADEX)')
axes[0].set_ylabel('flux')
axes[0].legend(loc='best')
axes[0].set_yscale('log')
axes[1].plot(tau_values,relative_diff*100)
axes[1].set_ylabel('relative difference [%]')
axes[2].plot(tau_values,ratio)
axes[2].set_ylabel('f_sphere/f_0D')
axes[2].axhline(expected_ratio_thin,color='black',linestyle='dashed')
for ax in axes:
    ax.set_xscale('log')
    ax.set_xlabel('tau_nu')

fig,ax = plt.subplots()
fig.suptitle("Static sphere: flux comparison between pythonradex and RADEX")
ax.plot(tau_values,relative_diff*100)
ax.set_ylabel('$(F_\mathrm{pythonradex}-F_\mathrm{RADEX})/F_\mathrm{pythonradex}$ [%]')
ax.set_xscale('log')
ax.set_xlabel('optical depth')