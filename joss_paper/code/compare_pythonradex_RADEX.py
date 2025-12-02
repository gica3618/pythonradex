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

#TODO either plot Tex instead of flux, or calculate the flux by using the level
#populations, not tau (since tau given by RADEX contains a "correction" for
#Gaussian line shape
#TODO: check the values in the region where there is strong disagreement

data_filename = "co.dat"
#can only use LVG slab and uniform sphere
geometry = "LVG slab"
geometry_radex = "LVG slab"
# geometry = "uniform sphere"
# geometry_radex = "static sphere"

#grid flux comparison
plot_trans_index = 1
grid_size = 25
#grid_size = 3

vmin,vmax = -100,100
linthresh = 1
cmap = plt.get_cmap("PuOr").copy()
cmap.set_bad("red")


##############################################################

#for consistency with RADEX:
T_background = 2.73
ext_background = helpers.generate_CMB_background()
line_profile_type = "rectangular"

solid_angle = 1
width_v = 1*constants.kilo
datafilepath = os.path.join("../../tests/LAMDA_files",data_filename)

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
    pop_up = [cloud.level_pop[t.up.number] for t in
              cloud.emitting_molecule.rad_transitions]
    pop_low = [cloud.level_pop[t.low.number] for t in
               cloud.emitting_molecule.rad_transitions]
    return cloud.Tex,pop_up,pop_low

def compute_RADEX_model(N,Tkin,coll_partner_densities,transitions):
    Tex = []
    pop_up = []
    pop_low = []
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
        pop_up.append(results["pop_up"])
        pop_low.append(results["pop_low"])
    return Tex,pop_up,pop_low

#plot comparison for a single model:
plot_max_trans_index = 25
colors = {"pythonradex":"blue","RADEX":"red"}
fig,axes = plt.subplots(nrows=3)
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
axes[1].set_ylabel(r"$\log(f_\mathrm{up})$")
axes[2].set_ylabel(r"$\log(f_\mathrm{low})$")
axes[2].set_xlabel("$E_u$ [K]")
for ax in axes[1:]:
    ax.set_yscale("log")


#now calculate the grid...
plot_trans = cloud.emitting_molecule.rad_transitions[plot_trans_index]
N_grid = np.logspace(13,18,grid_size)*constants.centi**-2
Tkin_grid = np.logspace(np.log10(5),np.log10(500),grid_size)
N_GRID,TKIN_GRID = np.meshgrid(N_grid,Tkin_grid,indexing='ij')

nH2_grid = np.array((1e2,2e3,5e4,1e6))*constants.centi**-3
residuals = {"Tex":[],"flux":[]}

v = np.linspace(-width_v*3,width_v*3,30)
nu = plot_trans.nu0*(1-v/constants.c)

excitation_temp = {"radex":np.empty((nH2_grid.size,N_grid.size,Tkin_grid.size)),
                   "pythonradex":np.empty((nH2_grid.size,N_grid.size,Tkin_grid.size))}
flux = {"radex":np.empty((nH2_grid.size,N_grid.size,Tkin_grid.size)),
        "pythonradex":np.empty((nH2_grid.size,N_grid.size,Tkin_grid.size))}
for i,nH2 in enumerate(nH2_grid):
    coll_densities = {'ortho-H2':nH2/2,'para-H2':nH2/2}
    for j,N in enumerate(N_grid):
        for k,Tkin in enumerate(Tkin_grid):
            print(f"nH2 = {nH2/constants.centi**-3:.2g} cm-3,"
                  +f" N = {N/constants.centi**-2:.2g}, Tkin={Tkin:.2g} K")
            radex_model = compute_RADEX_model(
                              N=N,Tkin=Tkin,coll_partner_densities=coll_densities,
                              transitions=[plot_trans,])
            radex_model = radex_model[0][0],radex_model[1][0],radex_model[2][0]
            pythonradex_model = compute_pythonradex_model(
                                     N=N,Tkin=Tkin,
                                     coll_partner_densities=coll_densities)
            pythonradex_model = pythonradex_model[0][plot_trans_index],\
                                          pythonradex_model[1][plot_trans_index],\
                                          pythonradex_model[2][plot_trans_index]
            models = {"radex":radex_model,"pythonradex":pythonradex_model}
            #don't take the flux or tau from RADEX directly (since it does some corrections
            #of rectangular to Gaussian), instead calculate it:
            for ID,model in models.items():
                Tex,pop_up,pop_low = model
                S = helpers.B_nu(nu=nu,T=Tex)
                tau_nu = plot_trans.tau_nu(N1=N*pop_low,N2=N*pop_up,nu=nu)
                f = cloud.geometry.compute_flux_nu(
                             tau_nu=tau_nu,source_function=S,solid_angle=solid_angle)
                f = -np.trapz(f,nu) #nu is decreasing, so need to take minus
                flux[ID][i,j,k] = f
                excitation_temp[ID][i,j,k] = Tex

for quantity,values in zip(("flux","Tex"),(flux,excitation_temp)):
    print(quantity)
    fig,axes = plt.subplots(2,2,constrained_layout=True)
    for code,val in values.items():
        n_negative = (val<0).sum()
        print(f"{quantity} {code}: {n_negative}/{val.size} are negative")
    residual = (values["pythonradex"]-values["radex"])/values["pythonradex"]
    masked_residual = residual.copy()*100
    assert -vmin == vmax
    mask = np.abs(masked_residual) > vmax
    masked_residual[mask] = np.nan
    print(f"{quantity} with large difference between pythonradex and RADEX")
    for i,j,k in zip(*mask.nonzero()):
        print(f"nH2={nH2_grid[i]/constants.centi**-3} cm-3, "
              +f"N={N_grid[j]/constants.centi**-2} cm-2, Tkin={Tkin_grid[k]:.3} K")
        for code,val in values.items():
            print(f"{code}: {quantity} = {val[i,j,k]:.3g}")
    for i,nH2 in enumerate(nH2_grid):
        ax = axes.ravel()[i]
        norm = SymLogNorm(linthresh=linthresh,vmin=vmin,vmax=vmax)
        im = ax.pcolormesh(N_GRID/constants.centi**-2,TKIN_GRID,masked_residual[i,:,:],
                           norm=norm,shading='auto',cmap=cmap)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(np.min(N_grid)/constants.centi**-2,np.max(N_GRID)/constants.centi**-2)
        ax.set_ylim(np.min(Tkin_grid),np.max(Tkin_grid))
        nH2_string = f"{nH2/constants.centi**-3:.1e}" # '2.3e+02'
        mant, exp = nH2_string.split("e")
        nH2_latex_str = rf"$n(H_2) = {mant}\times10^{{{int(exp)}}}$ cm$^{{-3}}$"
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
    if quantity == "flux":
        quantity_symb = "F"
    elif quantity == "Tex":
        quantity_symb = "T_\mathrm{ex}"
    else:
        raise RuntimeError
    colorbar_label = f"CO {plot_trans.name}\n"\
                    +f"$({quantity_symb}^\mathrm{{pythonradex}}"\
                    +f"-{quantity_symb}^\mathrm{{RADEX}})/{quantity_symb}^\mathrm{{pythonradex}}$ [%]"
    cbar.set_label(colorbar_label, labelpad=10)
    if True:
        plt.savefig(f"{quantity}_pythonradex_vs_radex.pdf",format="pdf",
                    bbox_inches="tight")


"""
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
                #tau_nu = phi_nu/np.max(phi_nu)*tau_nu0
                flux = cloud.geometry.compute_flux_nu(
                             tau_nu=tau_nu,source_function=S,solid_angle=solid_angle)
                flux = -np.trapz(flux,nu) #nu is decreasing, so need to take minus
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
    nH2_latex_str = rf"$n(H_2) = {mant}\times10^{{{int(exp)}}}$ cm$^{{-3}}$"
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

if True:
    plt.savefig("pythonradex_vs_radex.pdf",format="pdf",bbox_inches="tight")
"""