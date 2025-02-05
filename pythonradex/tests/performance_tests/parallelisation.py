#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 20:41:03 2024

@author: gianni
"""

#asked question on stackoverflow, why when creating 8 processes, each only
#used 25% CPU. First, this laptop has only 4 CPUs (although multiprocessing.cpu_count() 
#says 8). Second, CPU usage percentage does not directly translate to execution speed.
#For example, if the CPU temperature gets high, the CPU frequency is reduced
#In summary, there might be some benefit in using multiprocessing, but it's nowhere
#near a factor 8, maybe a factor 2 at most...

import sys
sys.path.append('/home/gianni/science/projects/code/pythonradex')
from pythonradex import radiative_transfer,helpers
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from scipy import constants
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
import os

#os.system("taskset -p 0xff %d" % os.getpid())
#os.sched_setaffinity(0,range(8))

#TODO compare to SpectralRadex, which has a parallel grid exploration

#chunk_sizes = [1,5,10,50,100,300,800]
chunk_sizes = [100,]

def generate_new_cloud():
    return radiative_transfer.Cloud(
                    datafilepath='/home/gianni/science/LAMDA_database_files/co.dat',
                    geometry='uniform sphere',line_profile_type='Gaussian',
                    width_v=1*constants.kilo,use_Ng_acceleration=True,
                    treat_line_overlap=False)

ext_background = helpers.generate_CMB_background(z=0)
collider = 'para-H2'

n_processes = [1,2,4,8]
n = 15
N_values = np.logspace(12,15,n)/constants.centi**2
coll_density_values = np.logspace(3,5,n)/constants.centi**3
Tkin_values = np.linspace(20,100,n)
print(f'number of calculations: {n**3}')
print(f'chunk sizes: {chunk_sizes}')
print(f'n_processes: {n_processes}')

#need to do a first calculation to compile everything
cloud = generate_new_cloud()
cloud.update_parameters(
      ext_background=ext_background,Tkin=20,
      collider_densities={collider:1e4/constants.centi**3},N=1e13/constants.centi**2,
      T_dust=0,tau_dust=0)
cloud.solve_radiative_transfer()

print('running without multiprocessing')
start = time.time()
cloud = generate_new_cloud()
for N,coll_dens,Tkin in itertools.product(N_values,coll_density_values,
                                             Tkin_values):
    collider_densities = {collider:coll_dens}
    cloud.update_parameters(
         ext_background=ext_background,Tkin=Tkin,
         collider_densities=collider_densities,N=N,T_dust=0,tau_dust=0)
    cloud.solve_radiative_transfer()
end = time.time()
time_without_multiprocessing = end-start
print(f'without multiprocessing: {time_without_multiprocessing:.3g}')

for chunksize in chunk_sizes:
    print(f'doing chunk size {chunksize}')
    multiprocessing_times = []
    for n_proc in n_processes:
        print(f'doing n_proc = {n_proc}')
        param_iterator = itertools.product(N_values,coll_density_values,Tkin_values)
        start = time.time()
        cloud = generate_new_cloud()
        def wrapper(params):
            N,coll_dens,Tkin = params
            collider_densities = {collider:coll_dens}
            cloud.update_parameters(
                  ext_background=ext_background,Tkin=Tkin,
                  collider_densities=collider_densities,N=N,T_dust=0,tau_dust=0)
            cloud.solve_radiative_transfer()
        if __name__ == '__main__':
            p = Pool(n_proc)
            p.map(wrapper,param_iterator,chunksize=chunksize)
        # with ProcessPoolExecutor() as executor:
        #     executor.map(wrapper,param_iterator,chunksize=chunk_size)
        end = time.time()
        exec_time = end-start
        multiprocessing_times.append(exec_time)
        print(f'n_proc = {n_proc}: {exec_time:.3g}')
    
    fig,ax = plt.subplots()
    ax.set_title(f'chunk size: {chunksize}')
    ax.plot(n_processes,multiprocessing_times)
    ax.axhline(time_without_multiprocessing,color='black',linestyle='dashed',
               label='without multiprocessing')
    ax.set_xlabel('n processes')
    ax.set_ylabel('time')
    ax.legend(loc='best')