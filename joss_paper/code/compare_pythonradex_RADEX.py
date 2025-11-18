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
from matplotlib.colors import LogNorm


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
# geometry_radex = "uniform sphere"

#don't use LVG sphere because RADEX uses a different formula for the escape
#probability
assert geometry == geometry_radex != "LVG sphere"
#also RADEX does not support uniform slab
assert geometry == geometry_radex != "uniform slab"

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
N_grid = np.logspace(13,18,grid_size)*constants.centi**-2
Tkin_grid = np.logspace(np.log10(5),np.log10(1000),grid_size)
N_GRID,TKIN_GRID = np.meshgrid(N_grid,Tkin_grid,indexing='ij')

nH2_grid = np.array((1e3,1e4,1e6,1e8))*constants.centi**-3
flux_residuals = []

v = np.linspace(-width_v*3,width_v*3,30)
nu = plot_trans.nu0*(1-v/constants.c)

vmin,vmax = 1,50
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("red")

fig,axes = plt.subplots(2,2,constrained_layout=True)
for i,nH2 in enumerate(nH2_grid):
    ax = axes.ravel()[i]
    fluxes = {"radex":np.empty((N_grid.size,Tkin_grid.size)),
              "pythonradex":np.empty((N_grid.size,Tkin_grid.size))}
    coll_densities = {'ortho-H2':nH2/2,'para-H2':nH2/2}
    for j,N in enumerate(N_grid):
        for k,Tkin in enumerate(Tkin_grid):
            radex_model = compute_RADEX_model(
                              N=N,Tkin=Tkin,coll_partner_densities=coll_densities,
                              transitions=[plot_trans,])
            pythonradex_model = compute_pythonradex_model(
                                     N=N,Tkin=Tkin,coll_partner_densities=coll_densities)
            models = {"radex":radex_model,"pythonradex":pythonradex_model}
            for ID,model in models.items():
                Tex,tau_nu0 = model
                if ID == "pythonradex":
                    Tex,tau_nu0 = Tex[plot_trans_index],tau_nu0[plot_trans_index]
                elif ID == "radex":
                    Tex,tau_nu0 = Tex[0],tau_nu0[0]
                else:
                    raise RuntimeError
                S = helpers.B_nu(nu=nu,T=Tex)
                phi_nu = plot_trans.line_profile.phi_nu(nu)
                tau_nu = phi_nu/np.max(phi_nu)*tau_nu0
                flux = cloud.geometry.compute_flux_nu(
                             tau_nu=tau_nu,source_function=S,solid_angle=solid_angle)
                flux = np.trapz(flux,nu)
                fluxes[ID][j,k] = flux
    mean_flux = (fluxes["pythonradex"]+fluxes["radex"])/2
    flux_residual = np.abs((fluxes["pythonradex"]-fluxes["radex"])/mean_flux)
    flux_residuals.append(flux_residual)
    masked_flux_residual = flux_residual.copy()*100
    masked_flux_residual[masked_flux_residual < vmin] = vmin
    masked_flux_residual[masked_flux_residual > vmax] = np.nan
    im = ax.pcolormesh(N_GRID/constants.centi**-2,TKIN_GRID,masked_flux_residual,
                       norm=LogNorm(vmin=vmin,vmax=vmax),shading='auto',cmap=cmap)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(np.min(N_grid)/constants.centi**-2,np.max(N_GRID)/constants.centi**-2)
    ax.set_ylim(np.min(Tkin_grid),np.max(Tkin_grid))
    assert np.log10(nH2).is_integer
    ax.set_title(f"$n(H_2) = 10^{int(np.log10(nH2))-6}$ cm$^{{-3}}$")
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
cbar.set_label(f"relative flux difference (CO {plot_trans.name}) [%]", labelpad=10)
# cbar_tick_labels = np.logspace(vmin,vmax,5)
# cbar_ticks = np.log10(cbar_tick_labels)
# cbar.set_ticks(cbar_ticks)
# cbar.set_ticklabels([round(t) for t in cbar_tick_labels])

print("max flux residual:")
for nH2,flux_residual in zip(nH2_grid,flux_residuals):
    print(f"nH2 = {nH2/constants.centi**-3} cm-3: {np.max(flux_residual)}")


if save_figure:
    plt.savefig("pythonradex_vs_radex.pdf",format="pdf",bbox_inches="tight") 


'''

#calculate a grid of models

coll_densities = {'ortho-H2':1e4/constants.centi**3,'para-H2':1e4/constants.centi**3}
Tex_grid = np.empty((N_grid.size,Tkin_grid.size,n_trans))
Tex_grid_radex = Tex_grid.copy()
for i,N in enumerate(N_grid):
    for j,Tkin in enumerate(Tkin_grid):
        radex_model = compute_RADEX_model(N=N,Tkin=Tkin,
                                          coll_partner_densities=coll_densities)
        pythonradex_model = compute_pythonradex_model(
                                 N=N,Tkin=Tkin,coll_partner_densities=coll_densities)
        Tex_pythonradex = pythonradex_model[0]
        Tex_grid[i,j,:] = Tex_pythonradex
        Tex_radex = radex_model[0]
        Tex_grid_radex[i,j,:] = Tex_radex
Tex_residual = np.abs((Tex_grid-Tex_grid_radex)/Tex_grid_radex)
Tex_residual = np.mean(Tex_residual,axis=-1)

N_GRID,TKIN_GRID = np.meshgrid(N_grid,Tkin_grid,indexing='ij')
fig,ax = plt.subplots()
im = ax.pcolormesh(N_GRID,TKIN_GRID,np.log10(Tex_residual))
fig.colorbar(im,ax=ax)
ax.set_xscale("log")
ax.set_yscale("log")
'''