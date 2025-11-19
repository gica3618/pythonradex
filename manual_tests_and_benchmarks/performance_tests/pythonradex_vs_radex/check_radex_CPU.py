#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:33:10 2025

@author: gianni
"""

#this is just to check how much CPU radex uses
#in terminal, do something like /usr/bin/time -v python check_radex_CPU.py

import os
import time


for i in range(1000):
    #start = time.time()
    os.system('../../../tests/Radex/bin/radex_static_sphere'
              +' < radex_test_preformance.inp > /dev/null')
    #end = time.time()
    #print(f"took {end-start}")