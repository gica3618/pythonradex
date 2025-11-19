#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 21:40:18 2025

@author: gianni
"""

import time
import grid_definition
import sys
sys.path.append('../..')
import general
from pythonradex import radiative_transfer,helpers
import itertools


ext_background = helpers.generate_CMB_background()

cloud = radiative_transfer.Cloud(
          datafilepath=general.datafilepath(grid_definition.grid["datafilename"]),
          geometry=grid_definition.geometry,
          line_profile_type=grid_definition.line_profile_type,
          width_v=grid_definition.width_v,use_Ng_acceleration=True,
          treat_line_overlap=False,warn_negative_tau=False)
#warm up
start_warmup = time.perf_counter()
cloud.update_parameters(
      ext_background=ext_background,Tkin=grid_definition.grid["Tkin_grid"][0],
      collider_densities={collider:grid_definition.coll_density_values[0] for collider in
                          grid_definition.grid["colliders"]},
      N=grid_definition.grid["N_grid"][0],T_dust=0,tau_dust=0)
cloud.solve_radiative_transfer()
end_warmup = time.perf_counter()
print(f"warm up: {end_warmup-start_warmup} s")

start = time.perf_counter()
#IMPORTANT: put N in innermost loop to improve performance
for coll_dens,Tkin,N in itertools.product(grid_definition.coll_density_values,
                                          grid_definition.grid["Tkin_grid"],
                                          grid_definition.grid["N_grid"]):
    collider_densities = {collider:coll_dens for collider in
                          grid_definition.grid["colliders"]}
    cloud.update_parameters(ext_background=ext_background,Tkin=Tkin,
                            collider_densities=collider_densities,N=N)
    cloud.solve_radiative_transfer()
end = time.perf_counter()
duration = end-start
print(f"duration: {duration} s")