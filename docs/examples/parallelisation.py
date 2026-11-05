#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:23:25 2026

@author: gianni
"""

from pythonradex import radiative_transfer, helpers
import time
import numpy as np
from scipy import constants
import itertools
from concurrent.futures import ProcessPoolExecutor


source_kwargs = {"datafilepath":"./co.dat",
                 "geometry":"static sphere",
                 "line_profile_type":"Gaussian",
                 "width_v":1 * constants.kilo,
                 "use_datafile_frequencies":True,
                 "warn_negative_tau":False}

source = radiative_transfer.Source(**source_kwargs)

initial_parameters = {"N":1e17 * constants.centi**-2,#arbitrary
                      "Tkin":100,#arbitrary
                      "collider_densities":{"ortho-H2": 1e4 * constants.centi**-3,
                                              "para-H2": 1e4 * constants.centi**-3},#arbitrary
                      "ext_background":helpers.generate_CMB_background(z=0),
                      "T_dust":0,
                      "tau_dust":0}

source.update_parameters(**initial_parameters)
#solve once, to force JIT (numba) compilation, so that compilation
#does not artificially inflate the execution time of the grid search
source.solve_radiative_transfer()

nH2_grid = np.logspace(3, 7, 20) * constants.centi**-3
Tkin_grid = np.logspace(1, np.log10(300), 20)
N_grid = np.logspace(13, 17, 20) * constants.centi**-2

grid = [{"Tkin": Tkin, "collider_densities":{"ortho-H2":nH2/2,"para-H2":nH2/2},
         "N":N}
        for Tkin,nH2,N in itertools.product(Tkin_grid, nH2_grid, N_grid)
        ]


#standard, non-parallel approach: a single Source object is used to walk through
#the grid
start = time.time()
print("running grid non-parallel")
intensities_nonparallel = []
for params in grid:
    source.update_parameters(**params)
    source.solve_radiative_transfer()
    #record the intensity of CO 1-0:
    intensity = source.frequency_integrated_emission(
                     output_type="intensity", transitions=0)
    intensities_nonparallel.append(intensity)
end = time.time()
duration_nonparallel = end - start
print(f"non-parallel: {duration_nonparallel:.2g} s")


#parallelisation: each worker has its own Source object
def init_worker():
    global worker_source
    worker_source = radiative_transfer.Source(**source_kwargs)
    worker_source.update_parameters(**initial_parameters)

def run_single(new_parameters):
    worker_source.update_parameters(**new_parameters)
    worker_source.solve_radiative_transfer()
    #return the intensity of CO 1-0:
    return worker_source.frequency_integrated_emission(
                 output_type="intensity", transitions=0)

start = time.time()
print("running grid in parallel")
#on MacOS and Windows, it might be necessary to do if __name__ == '__main__'
if __name__ == '__main__':
    with ProcessPoolExecutor(initializer=init_worker) as executor:
        intensities_parallel = list(executor.map(run_single, grid))
end = time.time()
duration_parallel = end - start
print("parallel (separate Source for each worker): "
      +f"{duration_parallel:.2g} s")

print(f"non-parallel/parallel ratio: {duration_nonparallel/duration_parallel:.2g}")


#verify that both methods give the same results:
intensities_serial = np.array(intensities_nonparallel)
if np.any(intensities_serial != intensities_parallel):
    raise RuntimeError