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

save_figure = True

data_filename = "co.dat"
#can only use static sphere
#(for LVG sphere, RADEX uses different escape prob.; RADEX does not support
#static slab; LVG slab needs rectangular profile)
geometry = "static sphere"
geometry_radex = "static sphere"

#for grid comparison:
plot_trans_index = 1
#grid_size = 25
grid_size = 10

vmin,vmax = 0.998,1.002
cmap = plt.get_cmap("PuOr").copy()
cmap.set_bad("red")


##############################################################

#for consistency with RADEX:
T_background = 2.73
ext_background = helpers.generate_CMB_background()
#RADEX computes using rectangular, but uses a correction factor to convert
#to Gaussian for flux and optical depth
line_profile_type = "Gaussian"

width_v = 1*constants.kilo
datafilepath = os.path.join("../../tests/LAMDA_files",data_filename)

#don't use LVG sphere because RADEX uses a different formula for the escape
#probability
assert geometry != "LVG sphere" and geometry_radex != "LVG sphere"
#also, RADEX does not support static slab
assert geometry != "static slab"
#LVG slab needs rectangular profile
assert geometry != "LVG slab"


source = radiative_transfer.Source(datafilepath=datafilepath,geometry=geometry,
                                   line_profile_type=line_profile_type,
                                   width_v=width_v)
#set ext background, dust:
source.update_parameters(ext_background=ext_background,N=1e15/constants.centi**2,
                         Tkin=94,collider_densities={"ortho-H2":1},
                         T_dust=0,tau_dust=0)
def compute_pythonradex_model(N,Tkin,coll_partner_densities):
    source.update_parameters(N=N,Tkin=Tkin,
                             collider_densities=coll_partner_densities)
    source.solve_radiative_transfer()
    return source.Tex,source.upper_level_population,source.lower_level_population,\
            source.tau_nu0_individual_transitions

def compute_RADEX_model(N,Tkin,coll_partner_densities,transitions):
    Tex = []
    pop_up = []
    pop_low = []
    tau = []
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
        tau.append(results["tau"])
    return Tex,pop_up,pop_low,tau

#plot comparison for a single model:
plot_max_trans_index = 25
colors = {"pythonradex":"blue","RADEX":"red"}
fig,axes = plt.subplots(nrows=4)
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
axes[3].set_ylabel("tau")
axes[3].set_xlabel("$E_u$ [K]")
for ax in axes[1:]:
    ax.set_yscale("log")
fig.tight_layout()


#now calculate the grid...
plot_trans = source.emitting_molecule.rad_transitions[plot_trans_index]
N_grid = np.logspace(13,18,grid_size)*constants.centi**-2
nH2_grid = np.logspace(2,6,grid_size)*constants.centi**-3
NH2_GRID,N_GRID = np.meshgrid(nH2_grid,N_grid,indexing='ij')

Tkin_grid = np.array((20,50,200),dtype=float)

residuals = {"Tex":[],"tau":[]}

shape = (Tkin_grid.size,nH2_grid.size,N_grid.size)
excitation_temp = {"radex":np.empty(shape),"pythonradex":np.empty(shape)}
optical_depth = {"radex":np.empty(shape),"pythonradex":np.empty(shape)}
failed_models = 0
for i,Tkin in enumerate(Tkin_grid):
    for j,nH2 in enumerate(nH2_grid):
        coll_densities = {'ortho-H2':3*nH2/4,'para-H2':nH2/4}
        for k,N in enumerate(N_grid):
            print(f"Tkin={Tkin:.2g} K, nH2 = {nH2/constants.centi**-3:.2g} cm-3,"
                  +f" N = {N/constants.centi**-2:.2g} cm-2, ")
            radex_model = compute_RADEX_model(
                              N=N,Tkin=Tkin,coll_partner_densities=coll_densities,
                              transitions=[plot_trans,])
            radex_model = [radex_model[i][0] for i in range(4)]
            try:
                pythonradex_model = compute_pythonradex_model(
                                         N=N,Tkin=Tkin,
                                         coll_partner_densities=coll_densities)
            except RuntimeError as e:
                if str(e) == "maximum number of iterations reached":
                    for code in excitation_temp.keys():
                        excitation_temp[code][i,j,k] = np.inf
                        optical_depth[code][i,j,k] = np.inf
                    failed_models += 1
                    continue
                else:
                    raise
            pythonradex_model = [pythonradex_model[i][plot_trans_index] for i
                                 in range(4)]
            models = {"radex":radex_model,"pythonradex":pythonradex_model}
            for ID,model in models.items():
                Tex,pop_up,pop_low,tau = model
                excitation_temp[ID][i,j,k] = Tex
                optical_depth[ID][i,j,k] = tau

def color_pixel(ax,X,Y,j,k,color):
    x0 = X[j,k]
    y0 = Y[j,k]
    try:
        width = X[j+1,k] - X[j,k]
        height = Y[j,k+1] - Y[j,k]
    except IndexError:
        width = X[j,k] - X[j-1,k]
        height = Y[j,k] - Y[j,k-1]
    ax.add_patch(plt.Rectangle((x0-width/2,y0-height/2),width,height,
                               color=color))

nrows = Tkin_grid.size
ncols = 2
fig,axes = plt.subplots(nrows=nrows,ncols=ncols,constrained_layout=True,
                        figsize=(6,8))
fig.suptitle(f"CO {plot_trans.name}")

X = np.log10(NH2_GRID/constants.centi**-3)
Y = np.log10(N_GRID/constants.centi**-2)

for column,(quantity,values) in enumerate(zip(("Tex","tau"),
                                              (excitation_temp,optical_depth))):
    print(quantity)
    both_negative = (values["pythonradex"] < 0) & (values["radex"] < 0)
    print(f"{quantity} both negative: {both_negative.sum()}/{both_negative.size}")
    for code,val in values.items():
        n_negative = (val<0).sum()
        print(f"{quantity} {code}: {n_negative}/{val.size} are negative")
        n_zero = (val==0).sum()
        print(f"{quantity} {code}: {n_zero}/{val.size} are 0")
    ratio = values["pythonradex"]/values["radex"]
    either_negative = (values["pythonradex"] < 0) | (values["radex"] < 0)
    exceeding_vmin_vmax = (ratio < vmin) | (ratio > vmax)

    #neither_negative = (values["pythonradex"] >= 0) & (values["radex"] >= 0)
    #ratio = np.where(neither_negative,values["pythonradex"]/values["radex"],np.nan)
    
    # masked_ratio = ratio.copy()
    # mask = (masked_ratio < vmin) | (masked_ratio > vmax)
    # masked_ratio[mask] = np.nan
    print(f"{quantity} with large difference between pythonradex and RADEX:")
    for i,j,k in zip(*np.isnan(ratio).nonzero()):
        print(f"Tkin={Tkin_grid[i]:.3} K, nH2={nH2_grid[j]/constants.centi**-3:.3g} cm-3, "
              +f"N={N_grid[k]/constants.centi**-2:.3g} cm-2, ")
        for code,val in values.items():
            print(f"{code}: {quantity} = {val[i,j,k]:.3g}")
    for row,Tkin in enumerate(Tkin_grid):
        ax = axes[row,column]
        im = ax.pcolormesh(X,Y,ratio[row,:,:],shading='nearest',cmap=cmap,vmin=vmin,
                           vmax=vmax)
        for j in range(nH2_grid.size):
            for k in range(N_grid.size):
                if False:#either_negative[row,j,k]:
                    color_pixel(ax=ax,X=X,Y=Y,j=j,k=k,color="red")
                elif exceeding_vmin_vmax[row,j,k]:
                    color_pixel(ax=ax,X=X,Y=Y,j=j,k=k,color="black")
                    ax.text(X[j,k], Y[j,k],f"{ratio[row,j,k]:.4g}",
                            ha="center", va="center",color="white",fontsize=5)
        ax.set_xlim(np.log10(np.min(nH2_grid)/constants.centi**-3),
                    np.log10(np.max(nH2_grid)/constants.centi**-3))
        ax.set_ylim(np.log10(np.min(N_grid)/constants.centi**-2),
                    np.log10(np.max(N_grid)/constants.centi**-2))
        if column == 0:
            Tkin_string = r"$T_\mathrm{kin}$"+f" = {Tkin:.0f} K"
            ax.set_title(Tkin_string)
        if row < nrows-1:
            ax.set_xticklabels([])
        if column > 0:
            ax.set_yticklabels([])
        if column == 0 and row == nrows-1:
            ax.set_xlabel("log H$_2$ density [log cm$^{-3}$]")
            ax.set_ylabel("log CO column density [log cm$^{-2}$]")
    cbar = fig.colorbar(im, ax=axes[:,column], orientation='horizontal', location='top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('top')
    if quantity == "tau":
        colorbar_label = r"$\tau_\mathrm{{pythonradex}}/\tau_\mathrm{{RADEX}}$"
    elif quantity == "Tex":
        colorbar_label = "$T_\mathrm{{ex}}^\mathrm{{pythonradex}}/T_\mathrm{{ex}}^\mathrm{{RADEX}}$"
    else:
        raise RuntimeError
    cbar.set_label(colorbar_label, labelpad=10)

n_models = nH2_grid.size*N_grid.size*Tkin_grid.size
print(f"{failed_models}/{n_models} pythonradex models failed")

if save_figure:
    print("saving figure")
    plt.savefig("pythonradex_vs_radex.pdf",format="pdf",bbox_inches="tight")
else:
    print("will not save figure")