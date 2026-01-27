#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 21:40:27 2025

@author: gianni
"""

import grid_definition
import itertools
import time
import numpy as np
import mini_radex_wrapper

input_filepath = "radex_test_preformance.inp"
output_filepath = "radex_test_performance.out"

width_v = grid_definition.width_v
datafilename = grid_definition.grid["datafilename"]
geometry = grid_definition.geometry

func_times = []
setup_times = []
calc_times = []
assert_times = []
start = time.perf_counter()
for coll_dens, Tkin, N in itertools.product(
    grid_definition.coll_density_values,
    grid_definition.grid["Tkin_grid"],
    grid_definition.grid["N_grid"],
):
    collider_densities = {
        collider: coll_dens for collider in grid_definition.grid["colliders"]
    }
    start_func = time.perf_counter()
    times = mini_radex_wrapper.run_radex(
        datafilename=datafilename,
        geometry=geometry,
        collider_densities=collider_densities,
        Tkin=Tkin,
        N=N,
        width_v=width_v,
        input_filepath=input_filepath,
        output_filepath=output_filepath,
    )
    end_func = time.perf_counter()
    setup_times.append(times["setup"])
    calc_times.append(times["calc"])
    assert_times.append(times["assert"])
    func_times.append(end_func - start_func)
end = time.perf_counter()
duration = end - start
print(f"duration: {duration} s")
for ID, times in zip(
    ("setup", "calc", "assert", "func"),
    (setup_times, calc_times, assert_times, func_times),
):
    print(
        f"{ID} times: {np.mean(times):.3g} +- {np.std(times):.3g}"
        + f" (min={np.min(times):.3g}, max={np.max(times):.3g})"
    )
    print(f"total {ID} time: {np.sum(times):.3g}")
