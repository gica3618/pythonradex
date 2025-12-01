#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 13:51:56 2025

@author: gianni
"""

import sys
sys.path.append('../../manual_tests_and_benchmarks/RADEX_wrapper')
import radex_wrapper
from scipy import constants
from pythonradex import radiative_transfer,helpers
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm


line_profile_type = "rectangular"
data_filename = "co.dat"
width_v = 1*constants.kilo
datafilepath = os.path.join("../../tests/LAMDA_files",data_filename)

T_background = 2.73
ext_background = helpers.generate_CMB_background()
#can only use LVG slab and uniform sphere
geometry = "LVG slab"
geometry_radex = "LVG slab"
# geometry = "uniform sphere"
# geometry_radex = "static sphere"

#don't use LVG sphere because RADEX uses a different formula for the escape
#probability
assert geometry != "LVG sphere" and geometry_radex != "LVG sphere"
#also RADEX does not support uniform slab
assert geometry != "uniform slab"

cloud = radiative_transfer.Cloud(datafilepath=datafilepath,geometry=geometry,
                                 line_profile_type=line_profile_type,
                                 width_v=width_v)
def compute_pythonradex_model(N,Tkin,coll_partner_densities):
    cloud.update_parameters(ext_background=ext_background,N=N,Tkin=Tkin,
                            collider_densities=coll_partner_densities,
                            T_dust=0,tau_dust=0)
    cloud.solve_radiative_transfer()
    return cloud.Tex,cloud.tau_nu0_individual_transitions

def compute_RADEX_model(N,Tkin,coll_partner_densities,transitions):
    Tex = []
    tau_nu0 = []
    epsilon = 1*constants.giga
    for trans in transitions:
        freq_interval = radex_wrapper.Interval(min=trans.nu0-epsilon,
                                               max=trans.nu0+epsilon)
        radex_input = radex_wrapper.RadexInput(
                         data_filename=data_filename,frequency_interval=freq_interval,
                         Tkin=Tkin,coll_partner_densities=coll_partner_densities,
                         T_background=T_background,column_density=N,Delta_v=width_v)
        wrapper = radex_wrapper.RadexWrapper(geometry=geometry_radex)
        results = wrapper.compute(radex_input)
        Tex.append(results["Tex"])
        tau_nu0.append(results["tau"])
    return Tex,tau_nu0

#plot comparison for a single model:
plot_max_trans_index = 25
colors = {"pythonradex":"blue","RADEX":"red"}
fig,axes = plt.subplots(nrows=2)
example_N = 1e14*constants.centi**-2
example_Tkin = 50
example_coll_dens = {'ortho-H2':1e4/constants.centi**3,'para-H2':1e4/constants.centi**3}
radex_model = compute_RADEX_model(N=example_N,Tkin=example_Tkin,
                                  coll_partner_densities=example_coll_dens,
                                  transitions=cloud.emitting_molecule.rad_transitions)
pythonradex_model = compute_pythonradex_model(
                         N=example_N,Tkin=example_Tkin,
                         coll_partner_densities=example_coll_dens)
x = [trans.up.E/constants.k for trans in
     cloud.emitting_molecule.rad_transitions[:plot_max_trans_index+1]]
for ax,y,y_radex in zip(axes,pythonradex_model,radex_model):
    ax.plot(x,y[:plot_max_trans_index+1],marker="o",linestyle='None',label="pythonradex",
            color=colors["pythonradex"])
    ax.plot(x,y_radex[:plot_max_trans_index+1],marker="x",linestyle='None',label="RADEX",
            color=colors["RADEX"])
axes[0].set_xticklabels([])
axes[0].set_ylabel("$T_\mathrm{ex}$ [K]")
axes[0].legend(loc="best")
axes[1].set_xlabel("$E_u$ [K]")
axes[1].set_ylabel(r"$\log(\tau(\nu_0))$")
axes[1].set_yscale("log")


#grid flux comparison
save_figure = True
plot_trans_index = 1
plot_trans = cloud.emitting_molecule.rad_transitions[plot_trans_index]
solid_angle = 1

grid_size = 25
#grid_size = 3
N_grid = np.logspace(13,18,grid_size)*constants.centi**-2
Tkin_grid = np.logspace(np.log10(5),np.log10(500),grid_size)
N_GRID,TKIN_GRID = np.meshgrid(N_grid,Tkin_grid,indexing='ij')

nH2_grid = np.array((1e2,2e3,5e4,1e6))*constants.centi**-3
flux_residuals = []

v = np.linspace(-width_v*3,width_v*3,30)
nu = plot_trans.nu0*(1-v/constants.c)

vmin,vmax = -10,10
cmap = plt.get_cmap("PuOr").copy()
cmap.set_bad("red")

n_fluxes = {"radex":[],"pythonradex":[]}
n_negative_fluxes = {"radex":[],"pythonradex":[]}
fig,axes = plt.subplots(2,2,constrained_layout=True)
for i,nH2 in enumerate(nH2_grid):
    ax = axes.ravel()[i]
    fluxes = {"radex":np.empty((N_grid.size,Tkin_grid.size)),
              "pythonradex":np.empty((N_grid.size,Tkin_grid.size))}
    coll_densities = {'ortho-H2':nH2/2,'para-H2':nH2/2}
    for j,N in enumerate(N_grid):
        for k,Tkin in enumerate(Tkin_grid):
            print(f"nH2 = {nH2/constants.centi**-3:.2g} cm-3,"
                  +f" N = {N/constants.centi**-2:.2g}, Tkin={Tkin:.2g} K")
            radex_model = compute_RADEX_model(
                              N=N,Tkin=Tkin,coll_partner_densities=coll_densities,
                              transitions=[plot_trans,])
            radex_model = radex_model[0][0],radex_model[1][0]
            pythonradex_model = compute_pythonradex_model(
                                     N=N,Tkin=Tkin,
                                     coll_partner_densities=coll_densities)
            pythonradex_model = pythonradex_model[0][plot_trans_index],\
                                pythonradex_model[1][plot_trans_index]
            models = {"radex":radex_model,"pythonradex":pythonradex_model}
            #don't take the flux from RADEX directly (since it does some corrections
            #of rectangular to Gaussian), instead calculate it:
            for ID,model in models.items():
                Tex,tau_nu0 = model
                S = helpers.B_nu(nu=nu,T=Tex)
                phi_nu = plot_trans.line_profile.phi_nu(nu)
                tau_nu = phi_nu/np.max(phi_nu)*tau_nu0
                flux = cloud.geometry.compute_flux_nu(
                             tau_nu=tau_nu,source_function=S,solid_angle=solid_angle)
                flux = -np.trapz(flux,nu)
                fluxes[ID][j,k] = flux
    # for flux in fluxes.values():
    #     assert np.all(flux>0)
    for code,flux in fluxes.items():
        n_fluxes[code].append(len(flux))
        n_negative_fluxes[code].append((flux<0).sum())
    flux_residual = (fluxes["pythonradex"]-fluxes["radex"])/fluxes["pythonradex"]
    flux_residuals.append(flux_residual)
    masked_flux_residual = flux_residual.copy()*100
    assert -vmin == vmax
    masked_flux_residual[np.abs(masked_flux_residual) > vmax] = np.nan
    im = ax.pcolormesh(N_GRID/constants.centi**-2,TKIN_GRID,masked_flux_residual,
                       norm=SymLogNorm(linthresh=0.1,vmin=vmin,vmax=vmax),
                       shading='auto',cmap=cmap)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(np.min(N_grid)/constants.centi**-2,np.max(N_GRID)/constants.centi**-2)
    ax.set_ylim(np.min(Tkin_grid),np.max(Tkin_grid))
    nH2_string = f"{nH2/constants.centi**-3:.1e}" # '2.3e+02'
    mant, exp = nH2_string.split("e")
    nH2_latex_str = rf"$n(H_2) = {mant}\times10^{{{int(exp)}}}$ cm$^{-3}$"
    ax.set_title(nH2_latex_str)
    if i in (0,1):
        ax.set_xticklabels([])
    if i in (1,3):
        ax.set_yticklabels([])
    if i in (0,2):
        ax.set_ylabel("$T_\mathrm{kin}$ [K]")
    if i in (2,3):
        ax.set_xlabel("CO column density [cm$^{-2}$]")
cbar = fig.colorbar(im, ax=axes.ravel(), orientation='horizontal', location='top')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks_position('top')
colorbar_label = f"CO {plot_trans.name}\n"\
                +r"$(F_\mathrm{pythonradex}-F_\mathrm{RADEX})/F_\mathrm{pythonradex}$ [%]"
cbar.set_label(colorbar_label, labelpad=10)
# cbar_tick_labels = np.logspace(vmin,vmax,5)
# cbar_ticks = np.log10(cbar_tick_labels)
# cbar.set_ticks(cbar_ticks)
# cbar.set_ticklabels([round(t) for t in cbar_tick_labels])

print("max flux residual:")
for i,(nH2,flux_residual) in enumerate(zip(nH2_grid,flux_residuals)):
    print(f"nH2 = {nH2/constants.centi**-3} cm-3: {np.max(flux_residual)}")
    for code,n_neg in n_negative_fluxes.items():
        print(f"{code}: {n_neg[i]}/{n_fluxes[code][i]} fluxes are negative")

if save_figure:
    plt.savefig("pythonradex_vs_radex.pdf",format="pdf",bbox_inches="tight") 