#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 20:07:06 2025

@author: gianni
"""

import os

this_filepath = os.path.abspath(__file__)
dirpath = os.path.dirname(this_filepath)
lamda_data_folder = os.path.join(dirpath,'../tests/LAMDA_files')

def datafilepath(filename):
    return os.path.join(lamda_data_folder,filename)