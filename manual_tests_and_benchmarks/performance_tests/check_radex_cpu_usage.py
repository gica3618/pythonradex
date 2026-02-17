#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:04:37 2025

@author: gianni
"""

import os

radex_executable = "../../tests/Radex/bin/radex_static_sphere"
radex_input_file = "radex_test_preformance.inp"

while True:
    os.system(f"{radex_executable} < {radex_input_file} > /dev/null")
