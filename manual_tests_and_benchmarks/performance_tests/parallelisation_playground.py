#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 19:01:54 2024

@author: gianni
"""

from multiprocessing import Pool
import numpy as np
import time

# the execution speeds observed using this script are not always the same; this might
# be an effect of CPU temperature? Also, sometimes I see 4 processes each using 100% CPU,
# (and faster), while sometime they use only 50% and same speed as with 1 process


matrix_dim = 100
n_processes = [1, 2, 4, 8]
n_iter = 100
chunksize = 10


def test_function():
    for i in range(500):
        matrix = np.random.rand(matrix_dim, matrix_dim)
        np.linalg.solve(matrix, np.ones(matrix_dim))


for n_proc in n_processes:
    print(f"calculating with {n_proc} processes")
    start = time.time()
    p = Pool(n_proc)
    p.starmap(test_function, [() for _ in range(n_iter)])
    end = time.time()
    print(f"time using {n_proc} processes: {end-start:.3g}")

print("now calculating without multiprocessing")
start = time.time()
[test_function() for _ in range(n_iter)]
end = time.time()
print(f"without multiprocess: {end-start:.3g}")
