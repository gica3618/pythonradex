#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:37:03 2024

@author: gianni
"""

import os
from pythonradex import radiative_transfer, helpers
from scipy import constants
import numpy as np
import RADEX_test_cases
import itertools
import pytest

# RADEX does the calculation with rectangular, but then applies a correction factor
# to convert optical depth, flux etc
line_profile_type = "Gaussian"

# following shows the tolerance that needs to be adopted to pass this test
rtol_Tex = {"static sphere RADEX": 5e-2, "LVG slab": 10e-2, "LVG sphere RADEX": 20e-2}
rtol_level_pop = {
    "static sphere RADEX": 5e-2,
    "LVG slab": 5e-2,
    "LVG sphere RADEX": 20e-2,
}
rtol_TR_flux = {
    "static sphere RADEX": 5e-2,
    "LVG slab": 5e-2,
    "LVG sphere RADEX": 30e-2,
}
rtol_tau = {"static sphere RADEX": 5e-2, "LVG slab": 10e-2, "LVG sphere RADEX": 20e-2}


frac_max_level_pop_to_consider = {
    "static sphere RADEX": 1e-5,
    "LVG slab": 1e-5,
    "LVG sphere RADEX": 0.01,
}

RADEX_geometry = {
    "static sphere RADEX": "Uniform sphere",
    "LVG slab": "Plane parallel slab",
    "LVG sphere RADEX": "Expanding sphere",
}

radex_output_collider_keys = {
    "H2": "H2",
    "pH2": "para-H2",
    "oH2": "ortho-H2",
    "e-": "e",
    "He": "He",
}

distance = 1 * constants.parsec
Omega = 1 * constants.au**2 / distance**2


def read_RADEX_output(filepath, molecule):
    output_file = open(filepath)
    output_lines = output_file.readlines()
    output_file.close()
    collider_densities = {}
    tau = []
    flux = []
    T_R = []
    Tex = []
    level_pop = np.ones(molecule.n_levels) * np.inf
    trans_counter = 0
    for line in output_lines:
        if "T(kin)" in line:
            Tkin = line.split(":")[1].replace(" ", "")
            Tkin = float(Tkin)
            continue
        if "Column density" in line:
            column_density = line.split(":")[1].replace(" ", "")
            column_density = float(column_density) / constants.centi**2
            continue
        if "Line width" in line:
            width_v = line.split(":")[1].replace(" ", "")
            width_v = float(width_v) * constants.kilo
            continue
        if "Density" in line:
            density = line.split(":")[1].replace(" ", "")
            density = float(density) / constants.centi**3
            collider = line.split()[3]
            collider = radex_output_collider_keys[collider]
            collider_densities[collider] = density
            continue
        if line[0].isdigit():
            linedata = line.split()
            flux.append(float(linedata[-1]) * constants.erg / constants.centi**2)
            T_R.append(float(linedata[-5]))
            tau.append(float(linedata[-6]))
            Tex.append(float(linedata[-7]))
            trans = molecule.rad_transitions[trans_counter]
            freq = float(linedata[-9]) * constants.giga
            assert np.isclose(trans.nu0, freq, atol=0, rtol=1e-3)
            trans_up_x = float(linedata[-4])
            trans_low_x = float(linedata[-3])
            if np.isfinite(level_pop[trans.up.index]):
                assert level_pop[trans.up.index] == trans_up_x
            else:
                level_pop[trans.up.index] = trans_up_x
            if np.isfinite(level_pop[trans.low.index]):
                assert level_pop[trans.low.index] == trans_low_x
            else:
                level_pop[trans.low.index] = trans_low_x
            trans_counter += 1
    assert np.all(np.isfinite(level_pop))
    return {
        "Tkin": Tkin,
        "column_density": column_density,
        "width_v": width_v,
        "collider_densities": collider_densities,
        "flux": np.array(flux),
        "T_R": np.array(T_R),
        "tau": np.array(tau),
        "Tex": np.array(Tex),
        "level_pop": np.array(level_pop),
    }


here = os.path.dirname(os.path.abspath(__file__))
LAMDA_folder = os.path.join(here, "LAMDA_files")
RADEX_output_folder = os.path.join(here, "RADEX_test_cases")


@pytest.mark.filterwarnings("ignore:some lines are overlapping")
@pytest.mark.filterwarnings("ignore:negative optical depth")
def test_vs_RADEX():
    max_taus = []
    for geo, geo_RADEX in RADEX_geometry.items():
        for test_case in RADEX_test_cases.test_cases:
            filename = test_case["filename"]
            datafilepath = os.path.join(LAMDA_folder, filename)
            specie = filename.split(".")[0]
            width_v = RADEX_test_cases.width_v
            for collider_densities, N, Tkin, T_bg in itertools.product(
                test_case["collider_densities_values"],
                test_case["N_values"],
                test_case["Tkin_values"],
                test_case["T_background_values"],
            ):
                # need to enter test mode to allow Gaussian line profile with LVG slab:
                source = radiative_transfer.Source(
                    datafilepath=datafilepath,
                    geometry=geo,
                    line_profile_type=line_profile_type,
                    width_v=width_v,
                    use_Ng_acceleration=True,
                    treat_line_overlap=False,
                    test_mode=True,
                )
                if T_bg == 0:
                    ext_background = lambda nu: np.zeros_like(nu)
                else:
                    ext_background = lambda nu: helpers.B_nu(nu=nu, T=T_bg)
                print(T_bg)
                source.update_parameters(
                    ext_background=ext_background,
                    N=N,
                    Tkin=Tkin,
                    collider_densities=collider_densities,
                    T_dust=0,
                    tau_dust=0,
                )
                source.solve_radiative_transfer()
                # print(f'tau: {np.min(source.tau_nu0)}, {np.max(source.tau_nu0)}')
                RADEX_output_filename = RADEX_test_cases.RADEX_out_filename(
                    radex_geometry=geo_RADEX,
                    specie=specie,
                    Tkin=Tkin,
                    T_background=T_bg,
                    N=N,
                    collider_densities=collider_densities,
                )
                RADEX_results_filepath = os.path.join(
                    RADEX_output_folder, RADEX_output_filename
                )
                RADEX_results = read_RADEX_output(
                    filepath=RADEX_results_filepath, molecule=source.emitting_molecule
                )
                assert RADEX_results["Tkin"] == Tkin
                # if collider is ortho-H2 or para-H2, RADEX automatically also
                # adds H2, although it will not use it (as long as H2 is not defined
                # in the LAMDA file, I think...)
                cleaned_RADEX_colliders = RADEX_results["collider_densities"].copy()
                for opH2 in ("ortho-H2", "para-H2"):
                    if opH2 in collider_densities:
                        if (
                            "H2" not in collider_densities
                            and "H2" in cleaned_RADEX_colliders
                        ):
                            del cleaned_RADEX_colliders["H2"]
                assert cleaned_RADEX_colliders == collider_densities
                assert RADEX_results["column_density"] == N
                assert RADEX_results["width_v"] == width_v
                print(specie)
                print(geo)
                print(N, Tkin, collider_densities)
                level_pop_selection = source.level_pop > frac_max_level_pop_to_consider[
                    geo
                ] * np.max(source.level_pop)
                # check Tex, tau, TR, flux for individual transitions:
                taus = []
                for i, trans in enumerate(source.emitting_molecule.rad_transitions):
                    if level_pop_selection[trans.up.index]:
                        tau_i = source.tau_nu0_individual_transitions[i]
                        taus.append(tau_i)
                        if RADEX_results["Tex"][i] < 0 or RADEX_results["tau"][i] < 0:
                            print("negative RADEX Tex or tau, skipping")
                            continue
                        assert np.isclose(
                            RADEX_results["Tex"][i],
                            source.Tex[i],
                            atol=0,
                            rtol=rtol_Tex[geo],
                        )
                        assert np.isclose(
                            RADEX_results["tau"][i], tau_i, atol=0, rtol=rtol_tau[geo]
                        )
                        B_Tex = helpers.B_nu(nu=trans.nu0, T=source.Tex[i])
                        I_bg = ext_background(trans.nu0)
                        I_bg_sub = (B_Tex - I_bg) * (1 - np.exp(-tau_i))
                        T_R = helpers.RJ_brightness_temperature(
                            specific_intensity=I_bg_sub, nu=trans.nu0
                        )
                        assert np.isclose(
                            RADEX_results["T_R"][i], T_R, atol=0, rtol=rtol_TR_flux[geo]
                        )
                        # apply same correction factor as RADEX, and multiply by 4pi:
                        flux = (
                            I_bg_sub * 1.0645 * 4 * np.pi * trans.line_profile.width_nu
                        )
                        assert np.isclose(
                            RADEX_results["flux"][i],
                            flux,
                            atol=0,
                            rtol=rtol_TR_flux[geo],
                        )
                if len(taus) > 0:
                    max_taus.append(np.max(taus))
                # check level pop:
                assert np.allclose(
                    RADEX_results["level_pop"][level_pop_selection],
                    source.level_pop[level_pop_selection],
                    atol=0,
                    rtol=rtol_level_pop[geo],
                )
    print(f"max(taus): {np.max(max_taus)}")
