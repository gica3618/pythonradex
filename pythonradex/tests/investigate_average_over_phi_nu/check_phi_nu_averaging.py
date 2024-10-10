#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:05:36 2024

@author: gianni
"""
#2024/07/30: I used this script to investigate the problems with the averaging over phi nu
#the problems turned out to be mainly related to beta, which was unstable for small tau
#(uniform sphere) or had strong discontinuities (uniform slab, LVG sphere RADEX)


import sys
sys.path.append('../../..')
from pythonradex import radiative_transfer,helpers
from scipy import constants
import numpy as np
import matplotlib.pyplot as plt

datafilepath = '../LAMDA_files/co.dat'
ext_background = helpers.generate_CMB_background()
T = 45
N = 1e15/constants.centi**2

cloud = radiative_transfer.Cloud(
                    datafilepath=datafilepath,geometry='LVG sphere RADEX',
                    line_profile_type='Gaussian',width_v=1*constants.kilo,
                    iteration_mode='ALI',use_NG_acceleration=True,
                    average_over_line_profile=True)
cloud.set_parameters(ext_background=ext_background,N=N,Tkin=T,
                     collider_densities={'para-H2':1e4/constants.centi**3})
level_pop = cloud.emitting_molecule.LTE_level_pop(T=T)

calculated_Jbar = cloud.Jbar_alllines_averaged(level_population=level_pop)
expected_Jbar = []
for i,line in enumerate(cloud.emitting_molecule.rad_transitions):
    n_low = line.low.number
    n_up = line.up.number
    x1 = level_pop[n_low]
    x2 = level_pop[n_up]
    N1 = N*x1
    N2 = N*x2
    nu0 = line.nu0
    width_nu = line.line_profile.width_nu
    if cloud.emitting_molecule.line_profile_type == 'rectangular':
        nu = np.linspace(nu0-width_nu/2,nu0+width_nu/2,300)
    else:
        nu = np.linspace(nu0-width_nu*2,nu0+width_nu*2,400)
    phi_nu = line.line_profile.phi_nu(nu)
    tau_nu = line.tau_nu(N1=N1,N2=N2,nu=nu)
    beta_nu = cloud.geometry.beta(tau_nu)
    Iext_nu = cloud.ext_background(nu)
    S = line.A21*x2/(x1*line.B12-x2*line.B21)
    Jbar = np.trapz((beta_nu*Iext_nu+(1-beta_nu)*S)*phi_nu,nu)
    Jbar /= np.trapz(phi_nu,nu)
    print(Jbar,calculated_Jbar[i])
    residual = np.abs((Jbar-calculated_Jbar[i])/calculated_Jbar[i])
    if residual > 1e-2 or i==6:
        fig,axes = plt.subplots(2,2)
        components = {'phi_nu':phi_nu,'tau_nu':tau_nu,'beta_nu':beta_nu,
                      'Iext_nu':Iext_nu}
        for ax,(ID,comp) in zip(axes.ravel(),components.items()):
            ax.plot(nu,comp,label=ID)
            ax.legend(loc='best')