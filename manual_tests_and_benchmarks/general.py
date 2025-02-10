#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 20:07:06 2025

@author: gianni
"""

import os

lamda_data_folder = '../../tests/LAMDA_files'

def datafilepath(filename):
    return os.path.join(lamda_data_folder,filename)