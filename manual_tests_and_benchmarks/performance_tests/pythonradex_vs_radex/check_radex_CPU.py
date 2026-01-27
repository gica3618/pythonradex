#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:33:10 2025

@author: gianni
"""

# this is just to check how much CPU radex uses
# in terminal, do something like /usr/bin/time -v python check_radex_CPU.py

import os
import time

in_file = "check_radex_CPU.inp"
with open(in_file, "r") as f:
    out_file = f.readlines()[1].strip()
assert out_file == "check_radex_CPU.out", out_file

# command = '../../../tests/Radex/bin/radex_static_sphere < radex_test_preformance.inp > /dev/null'
# safer without dev/null, to see any error messages; performance seems not strongly affected
command = f"../../../tests/Radex/bin/radex_static_sphere < {in_file}"

for i in range(1000):
    # start = time.time()
    os.system(command)
    # end = time.time()
    # print(f"took {end-start}")
    # make sure RADEX ran successfully:
    assert os.path.exists(out_file)
