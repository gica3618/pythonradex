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


data_filename = "co.dat"
#can only use LVG slab and uniform sphere
geometry = "LVG slab"
geometry_radex = "LVG slab"
# geometry = "uniform sphere"
# geometry_radex = "static sphere"

#for grid comparison:
plot_trans_index = 1
grid_size = 25
#grid_size = 5

vmin,vmax = 0.95,1.05
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
#also, RADEX does not support uniform slab
assert geometry != "uniform slab"


source = radiative_transfer.Source(datafilepath=datafilepath,geometry=geometry,
                                   line_profile_type=line_profile_type,
                                   width_v=width_v)
def compute_pythonradex_model(N,Tkin,coll_partner_densities):
    source.update_parameters(ext_background=ext_background,N=N,Tkin=Tkin,
                             collider_densities=coll_partner_densities,
                             T_dust=0,tau_dust=0)
    source.solve_radiative_transfer()
    pop_up = [source.level_pop[t.up.number] for t in
              source.emitting_molecule.rad_transitions]
    pop_low = [source.level_pop[t.low.number] for t in
               source.emitting_molecule.rad_transitions]
    return source.Tex,pop_up,pop_low

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
                                  transitions=source.emitting_molecule.rad_transitions)
pythonradex_model = compute_pythonradex_model(
                         N=example_N,Tkin=example_Tkin,
                         coll_partner_densities=example_coll_dens)
x = [trans.up.E/constants.k for trans in
     source.emitting_molecule.rad_transitions[:plot_max_trans_index+1]]
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
plot_trans = source.emitting_molecule.rad_transitions[plot_trans_index]
N_grid = np.logspace(13,18,grid_size)*constants.centi**-2
Tkin_grid = np.logspace(np.log10(5),np.log10(500),grid_size)
N_GRID,TKIN_GRID = np.meshgrid(N_grid,Tkin_grid,indexing='ij')

nH2_grid = np.array((1e3,1e4,1e5))*constants.centi**-3
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
                intensity = source.geometry.intensity(
                             tau_nu=tau_nu,source_function=S)
                f = -np.trapz(intensity*solid_angle,nu) #nu is decreasing, so need to take minus
                flux[ID][i,j,k] = f
                excitation_temp[ID][i,j,k] = Tex

nrows = nH2_grid.size
ncols = 2
fig,axes = plt.subplots(nrows=nrows,ncols=ncols,constrained_layout=True,
                        figsize=(6,8))
fig.suptitle(f"CO {plot_trans.name}")

for column,(quantity,values) in enumerate(zip(("Tex","flux"),(excitation_temp,flux))):
    print(quantity)
    for code,val in values.items():
        n_negative = (val<0).sum()
        print(f"{quantity} {code}: {n_negative}/{val.size} are negative")
        n_zero = (val==0).sum()
        print(f"{quantity} {code}: {n_zero}/{val.size} are 0")
    ratio = values["pythonradex"]/values["radex"]
    #assert np.all(np.isfinite(ratio))
    masked_ratio = ratio.copy()
    mask = (masked_ratio < vmin) | (masked_ratio > vmax)
    masked_ratio[mask] = np.nan
    print(f"{quantity} with large difference between pythonradex and RADEX:")
    for i,j,k in zip(*mask.nonzero()):
        print(f"nH2={nH2_grid[i]/constants.centi**-3:.3g} cm-3, "
              +f"N={N_grid[j]/constants.centi**-2:.3g} cm-2, Tkin={Tkin_grid[k]:.3} K")
        for code,val in values.items():
            print(f"{code}: {quantity} = {val[i,j,k]:.3g}")
    for row,nH2 in enumerate(nH2_grid):
        ax = axes[row,column]
        im = ax.pcolormesh(N_GRID/constants.centi**-2,TKIN_GRID,masked_ratio[row,:,:],
                           shading='auto',cmap=cmap,vmin=vmin,vmax=vmax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(np.min(N_grid)/constants.centi**-2,np.max(N_GRID)/constants.centi**-2)
        ax.set_ylim(np.min(Tkin_grid),np.max(Tkin_grid))
        nH2_string = f"{nH2/constants.centi**-3:.1e}" # '2.3e+02'
        mant, exp = nH2_string.split("e")
        nH2_latex_str = rf"$n(H_2) = {mant}\times10^{{{int(exp)}}}$ cm$^{{-3}}$"
        ax.set_title(nH2_latex_str)
        if row < nrows-1:
            ax.set_xticklabels([])
        if column > 0:
            ax.set_yticklabels([])
        if column == 0:
            ax.set_ylabel("kinetic temperature [K]")
        if row == nrows-1:
            ax.set_xlabel("CO column density [cm$^{-2}$]")
    cbar = fig.colorbar(im, ax=axes[:,column], orientation='horizontal', location='top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('top')
    if quantity == "flux":
        colorbar_label = "$F_\mathrm{{pythonradex}}/F_\mathrm{{RADEX}}$"
    elif quantity == "Tex":
        colorbar_label = "$T_\mathrm{{ex}}^\mathrm{{pythonradex}}/T_\mathrm{{ex}}^\mathrm{{RADEX}}$"
    else:
        raise RuntimeError
    cbar.set_label(colorbar_label, labelpad=10)
if True:
    plt.savefig("pythonradex_vs_radex.pdf",format="pdf",bbox_inches="tight")