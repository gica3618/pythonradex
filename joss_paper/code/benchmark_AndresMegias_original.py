#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PythonRadex test

Andrés
"""

import io
import time
import subprocess
import numpy as np
import pandas as pd
import pythonradex as pradex
from scipy import constants
# import matplotlib.pyplot as plt

radex_path = '/Users/andres/Applications/RADEX/bin/radex'

def radex(molecule, freq_range, backgr_temp, kin_temp, molec_col_dens,
          collider_num_dens, line_width):
    """
    Perform the RADEX calculation with the given parameters.

    Parameters
    ----------
    molecule : str
        RADEX name of the molecule.
    freq_range : list (float)
        Minimum and maximum frequencies to compute (GHz).
    backgr_temp : str
        Background temperature (K).
    molec_kin_temp : str / array
        Kinetic temperature (K).
    molec_col_dens : str / array
        Column density of the molecule (/cm2).
    collider_num_dens : dict(float)
        collider number densities (/cm3).
    line_width : str
        Width of the lines of the molecule (km/s).

    Returns
    -------
    transitions_df : dataframe
        List of calculated transitions by RADEX on-line
    """
    radex_name = species_abreviations[molecule]
    def create_radex_input(molecule, min_freq, max_freq, kin_temp, backgr_temp,
                           collider_num_dens, molec_col_dens, line_width):
        """
        Create the RADEX input file with the given parameters.
        """
        text = []
        radex_name = species_abreviations[molecule]
        if type(molec_col_dens) in (str, int, float, dict):
            molec_col_dens = [molec_col_dens]
        if type(collider_num_dens) in (str, int, float, dict):
            collider_num_dens = [collider_num_dens]
        if type(kin_temp) in (str, int, float, dict):
            kin_temp = [kin_temp]
        num_calcs = 0
        for kin_temp_i in kin_temp:
            for collider_num_dens_i in collider_num_dens:
                for col_dens_i in molec_col_dens:
                    num_collision_partners = len(collider_num_dens_i)
                    text += [f'{radex_name}.dat']
                    text += [f'{radex_name}.out']
                    text += [f'{min_freq} {max_freq}']
                    text += [f'{kin_temp_i}']
                    text += [f'{num_collision_partners}']
                    for (species,num_dens_i) in collider_num_dens_i.items():
                        text += [f'{species}', f'{float(num_dens_i):e}']
                    text += [f'{backgr_temp}']
                    text += [f'{float(col_dens_i):e}']
                    text += [f'{line_width}']
                    text += ['1']
                    num_calcs += 1
        text[-1] = text[-1].replace('1', '0')
        for i in range(len(text) - 1):
            text[i] += '\n'
        with open(f'{radex_name}.inp', 'w') as file:
            file.writelines(text)
        return num_calcs
    min_freq, max_freq = freq_range
    num_calcs = create_radex_input(molecule, min_freq, max_freq, kin_temp,
                                   backgr_temp, collider_num_dens,
                                   molec_col_dens, line_width)
    subprocess.run(f'{radex_path} < {radex_name}.inp > radex_log.txt', shell=True)
    with open(f'{radex_name}.out', 'r') as file:
        output_text = file.readlines()
    i = -1
    write_line = False
    tables = [[] for j in range(num_calcs)]
    for line in output_text:
        if line.startswith('* Radex version'):
            write_line = False
            if i >= 0:
                del tables[i][1]
            i += 1
        if write_line:
            tables[i] += [line]
        if line.startswith('Calculation finished'):
            write_line = True
    del tables[num_calcs-1][1]
    transitions_df = [[] for j in range(num_calcs)]
    for i in range(num_calcs):
        tables[i] = ''.join(tables[i])
        result_df = pd.read_csv(io.StringIO(tables[i]), delimiter='\\s+')
        result_df = result_df.astype(object)
        transitions1 = [idx[0] for idx in list(result_df.index)]
        for (j,(idx,row)) in enumerate(result_df.iterrows()):
            result_df.at[idx,'LINE'] = f"{transitions1[j]} -- {result_df['LINE'][idx]}"
        result_df.index = np.arange(len(transitions1))+1
        transitions_dict = {'transition': result_df['LINE'].values,
                            'freq. (GHz)': result_df['FREQ'].values,
                            'ex. temp. (K)': result_df['T_EX'].values,
                            'opt. depth': result_df['TAU'].values,
                            'intens. (K)': result_df['T_R'].values}
        transitions_df[i] = pd.DataFrame(transitions_dict, index=result_df.index)

    return transitions_df

species_abreviations = {
    'CO': 'co',
    '13CO': '13co',
    'C18O': 'c18o',
    'C17O': 'c17o',
    'CS': 'cs',
    'p-H2S': 'ph2s',
    'o-H2S': 'oh2s',
    'HCO+': 'hco+',
    'DCO+': 'dco+',
    'H13CO+': 'h13co+',
    'HC18O+': 'hc18o+',
    'HC17O+': 'hc17o+',
    'Oatom': 'oatom',
    'Catom': 'catom',
    'C+ion': 'c+',
    'N2H+': 'n2h+',
    'HCN': 'hcn@hfs',
    'H13CN': 'h13cn',
    'HC15N': 'hc15n',
    'HC3N': 'hc3n',
    'HNC': 'hnc',
    'SiO': 'sio',
    '29SiO': '29sio',
    'SiS': 'sis',
    'O2': 'o2',
    'CN': 'cn',
    'SO': 'so',
    'SO2': 'so2',
    'o-SiC2': 'o-sic2',
    'OCS': 'ocs',
    'HCS+': 'hcs+',
    'o-H2CO': 'o-h2co',
    'p-H2CO': 'p-h2co',
    'o-H2CS': 'oh2cs',
    'p-H2CS': 'ph2cs',
    'CH3OH-E': 'e-ch3oh',
    'CH3OH-A': 'a-ch3oh',
    'CH3CN': 'ch3cn',
    'o-C3H2': 'o-c3h2',
    'p-C3H2': 'p-c3h2',
    'OH': 'oh',
    'o-H2O': 'o-h2o',
    'p-H2O': 'p-h2o',
    'HDO': 'hdo',
    'HCl': 'hcl@hfs',
    'o-NH3': 'o-nh3',
    'p-NH3': 'p-nh3',
    'o-H3O+': 'o-h3o+',
    'p-H3O+': 'p-h3o+',
    'HNCO': 'hnco',
    'NO': 'no',
    'HF': 'hf'
     }


#%%

print('PythonRADEX benchmark')
print('---------------------')
print()

datafilepath = './co.dat' #file downloaded from EMAA or LAMDA database
geometry = 'uniform sphere'
line_profile_type = 'Gaussian'
width_v = 1*constants.kilo

N = molec_col_dens = 1e16/constants.centi**2
Tkin = kin_temp = 120
collider_densities = {'ortho-H2': 2e2/constants.centi**3,
                      'para-H2': 6e2/constants.centi**3}
T_dust = 0
tau_dust = 0
index_21 = 1

num_runs = 1000

print('Individual run :')

t1 = time.time()

cloud = pradex.radiative_transfer.Cloud(
                          datafilepath=datafilepath,geometry=geometry,
                          line_profile_type=line_profile_type,width_v=width_v)
ext_background = pradex.helpers.generate_CMB_background(z=0)
cloud.update_parameters(N=N, Tkin=Tkin, collider_densities=collider_densities,
                        ext_background=ext_background, T_dust=T_dust,
                        tau_dust=tau_dust)
cloud.solve_radiative_transfer()

nu0_21 = cloud.emitting_molecule.nu0[index_21]
nu0 = float(nu0_21)

tau0 = cloud.tau_nu(nu=nu0)
Tex = cloud.Tex[index_21]
I0 = (pradex.helpers.B_nu(nu0, Tex) - ext_background(nu0)) * (1 - np.exp(-tau0))
# I0 = cloud.spectrum(1., nu=np.array([nu0]))
# I0 -= ext_background(nu0)
Tb0 = I0 * constants.c**2 / (2 * nu0**2 * constants.Boltzmann)

t2 = time.time()
dt = t2 - t1

v = np.linspace(-width_v, width_v, 20)
nu = nu0_21 * (1- v/constants.c)

print(f'- PythonRADEX: {dt*1e3:.3f} ms')

t1 = time.time()

radex_results = radex('CO', freq_range=[nu.min()/1e9,nu.max()/1e9], backgr_temp=2.73,
                      kin_temp=Tkin, collider_num_dens={'ortho-H2': 2e2, 'para-H2': 6e2},
                      molec_col_dens=N*constants.centi**2, line_width=width_v/1e3)

t2 = time.time()
dt = t2 - t1

print(f'- RADEX: {dt*1e3:.3f} ms')

print(f'Multiple run (x {num_runs}) :')

t1 = time.time()

cloud = pradex.radiative_transfer.Cloud(
                          datafilepath=datafilepath,geometry=geometry,
                          line_profile_type=line_profile_type,width_v=width_v)
ext_background = pradex.helpers.generate_CMB_background(z=0)

for i in range(num_runs):
    cloud.update_parameters(N=N, Tkin=Tkin, collider_densities=collider_densities,
                            ext_background=ext_background, T_dust=T_dust,
                            tau_dust=tau_dust)
    cloud.solve_radiative_transfer()
    nu0_21 = cloud.emitting_molecule.nu0[index_21]
    nu0 = float(nu0_21)
    tau0 = cloud.tau_nu(nu=nu0)
    I0 = (pradex.helpers.B_nu(nu0, Tex) - ext_background(nu0)) * (1 - np.exp(-tau0))
    Tb0 = I0 * constants.c**2 / (2 * nu0**2 * constants.Boltzmann)

t2 = time.time()
dt = t2 - t1

v = np.linspace(-width_v, width_v, 20)
nu = nu0_21 * (1- v/constants.c)

dt_pradex = dt

print(f'- PythonRADEX: {dt*1e3:.0f} ms')

t1 = time.time()

Tkin = [Tkin for i in range(num_runs)]

radex_results = radex('CO', freq_range=[nu.min()/1e9,nu.max()/1e9], backgr_temp=2.73,
                      kin_temp=Tkin, collider_num_dens={'ortho-H2': 2e2, 'para-H2': 6e2},
                      molec_col_dens=N*constants.centi**2, line_width=width_v/1e3)

t2 = time.time()
dt = t2 - t1

dt_radex = dt
ratio = dt_radex / dt_pradex

print(f'- RADEX: {dt*1e3:.0f} ms')
print(f' * Ratio: {ratio:.1f}')










