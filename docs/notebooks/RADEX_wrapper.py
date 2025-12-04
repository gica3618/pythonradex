#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 18:19:04 2025

@author: gianni
"""

import os
from scipy import constants
import subprocess
import time
import pandas as pd


radex_path = "/home/gianni/Applications/Radex/bin/radex"


radex_collider_keys = {'H2':'H2','para-H2':'p-H2','ortho-H2':'o-H2','e':'e',
                       'He':'He'}


def write_radex_input_file(datafilename,collider_densities,Tkin,T_background,
                           N,width_v,input_filepath,output_filepath):
    with open(input_filepath,mode='w') as f:
        f.write(datafilename+'\n')
        f.write(f'{output_filepath}\n')
        f.write('0 0\n') #output all transitions
        f.write(f'{Tkin}\n')
        f.write(f'{len(collider_densities)}\n')
        for collider,density in collider_densities.items():
            f.write(radex_collider_keys[collider]+'\n')
            f.write(f'{density/constants.centi**-3}\n')
        f.write(f'{T_background}\n')
        f.write(f'{N/constants.centi**-2}\n')
        f.write(f'{width_v/constants.kilo}\n')
        f.write('0\n')

def run_radex(input_filepath,output_filepath,verbose=False):
    start = time.time()
    if os.path.exists(output_filepath):
        os.remove(output_filepath)
    execution_start = time.time
    subprocess.run(f'{radex_path} < {input_filepath} > radex_log.txt', shell=True)
    execution_end = time.time()
    assert os.path.exists(output_filepath),"Radex crashed?"
    end = time.time()
    if verbose:
        print(f"execution time: {execution_end-execution_start:.2g}")
        print(f"total run time: {end-start:.2g}")

def read_radex_output(output_filepath):
    rows = []
    with open(output_filepath) as f:
        read = False
        for i,line in enumerate(f):
            if not read:
                if "(erg/cm2/s)" in line:
                    read = True
                continue
            if read:
                components = line.split()
                row = {'flux':float(components[-1])*constants.erg/constants.centi**2,
                       "pop_low":float(components[-3]),"pop_up":float(components[-4]),
                       'TR':float(components[-5]),'tau':float(components[-6]),
                       'Tex':float(components[-7]),
                       "wavelength":float(components[-8])*constants.micro,
                       "freq":float(components[-9])*constants.giga,
                       "Eup":float(components[-10])*constants.k}
                line_name = "".join(components[:-10])
                row["line_name"] = line_name
                rows.append(row)
    return pd.DataFrame(rows)

def run(datafilename,collider_densities,Tkin,T_background,N,width_v,input_filepath,
        output_filepath):
    write_radex_input_file(
          datafilename=datafilename,collider_densities=collider_densities,
          Tkin=Tkin,T_background=T_background,N=N,width_v=width_v,input_filepath=input_filepath,
          output_filepath=output_filepath)
    run_radex(input_filepath=input_filepath,output_filepath=output_filepath)
    return read_radex_output(output_filepath=output_filepath)


if __name__ == "__main__":
    datafilename = "co.dat"
    collider_densities = {"ortho-H2":1e5*constants.centi**-3,
                          "para-H2":1e5*constants.centi**-3}
    Tkin = 101
    T_background = 2.73
    N = 1e16*constants.centi**-2
    width_v = 2.3*constants.kilo
    input_filepath = "test.inp"
    output_filepath = "test.out"
    test = run(datafilename=datafilename,collider_densities=collider_densities,
               Tkin=Tkin,T_background=T_background,N=N,width_v=width_v,input_filepath=input_filepath,
               output_filepath=output_filepath)