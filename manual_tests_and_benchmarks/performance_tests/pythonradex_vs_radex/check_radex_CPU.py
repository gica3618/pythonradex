#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:33:10 2025

@author: gianni
"""

#in terminal, do something like /usr/bin/time -v python check_radex_CPU.py

import os
import time

radex_executable = '../../../tests/Radex/bin/radex_static_sphere'
radex_input_file = 'radex_test_preformance.inp'

for i in range(1000):
    #start = time.time()
    os.system(f'{radex_executable} < {radex_input_file} > /dev/null')
    #end = time.time()
    #print(f"took {end-start}")