#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:11:07 2024

@author: gianni
"""


from pythonradex import radiative_transfer
from scipy import constants
import os
import numpy as np
import itertools
import pytest


tau_dust_values = {"thick": 10, "thin": 1e-4}
default_T_dust = 100
Tkin = 30
geometries = ("static sphere", "static slab")
line_profile_types = ("rectangular", "Gaussian")


def generate_cloud(
    datafilename,
    geometry,
    line_profile_type,
    width_v,
    treat_line_overlap,
    N,
    collider_densities,
    tau_dust,
    T_dust,
):
    here = os.path.dirname(os.path.abspath(__file__))
    datafilepath = os.path.join(here, f"LAMDA_files/{datafilename}")
    src = radiative_transfer.Source(
        datafilepath=datafilepath,
        geometry=geometry,
        line_profile_type=line_profile_type,
        width_v=width_v,
        use_Ng_acceleration=True,
        treat_line_overlap=treat_line_overlap,
    )
    src.update_parameters(
        ext_background=0,
        N=N,
        Tkin=Tkin,
        collider_densities=collider_densities,
        T_dust=T_dust,
        tau_dust=tau_dust,
    )
    src.solve_radiative_transfer()
    return src


class TestDust:

    width_v = {"co": 1 * constants.kilo, "cn": 1000 * constants.kilo}
    treat_line_overlap = {"co": False, "cn": True}
    datafilenames = {"co": "co.dat", "cn": "cn.dat"}

    def cloud_iterator(self, N, collider_densities, tau_dust, T_dust, molecule_name):
        for geo, lp in itertools.product(geometries, line_profile_types):
            treat_line_overlap = self.treat_line_overlap[molecule_name]
            yield generate_cloud(
                datafilename=self.datafilenames[molecule_name],
                geometry=geo,
                line_profile_type=lp,
                width_v=self.width_v[molecule_name],
                treat_line_overlap=treat_line_overlap,
                N=N,
                collider_densities=collider_densities,
                tau_dust=tau_dust,
                T_dust=T_dust,
            )

    def test_thin_dust_thick_gas(self):
        # expect that dust does not have any effect
        gas_params = {
            "co": {
                "N": 1e16 / constants.centi**2,
                "collider_densities": {"ortho-H2": 1e5 / constants.centi**3},
            },
            "cn": {
                "N": 1e15 / constants.centi**2,
                "collider_densities": {"e": 1e3 / constants.centi**3},
            },
        }
        tau_dust = tau_dust_values["thin"]
        for mol_name, params in gas_params.items():
            cloud_iterator_with_dust = self.cloud_iterator(
                **params,
                tau_dust=tau_dust,
                T_dust=default_T_dust,
                molecule_name=mol_name,
            )
            cloud_iterator_wo_dust = self.cloud_iterator(
                **params, tau_dust=0, T_dust=0, molecule_name=mol_name
            )
            for dust_cloud, no_dust_cloud in zip(
                cloud_iterator_with_dust, cloud_iterator_wo_dust
            ):
                assert np.allclose(
                    dust_cloud.level_pop, no_dust_cloud.level_pop, atol=1e-4, rtol=1e-2
                )

    @pytest.mark.filterwarnings("ignore:negative optical depth")
    def test_thick_dust_thin_gas(self):
        # expect LTE at T_dust
        # for this test to pass I need to use a relatively generous atol
        tau_dust = tau_dust_values["thick"]
        T_dust = default_T_dust
        gas_params = {
            "co": {
                "N": 1e12 / constants.centi**2,
                "collider_densities": {"ortho-H2": 1e1 / constants.centi**3},
            },
            "cn": {
                "N": 1e11 / constants.centi**2,
                "collider_densities": {"e": 1e-1 / constants.centi**3},
            },
        }
        for mol_name, params in gas_params.items():
            src_iter = self.cloud_iterator(
                **params, tau_dust=tau_dust, T_dust=T_dust, molecule_name=mol_name
            )
            for source in src_iter:
                assert (
                    source.rate_equations.Tkin != T_dust
                ), "if Tkin=Tdust, cannot say if LTE is caused by gas or dust"
                expected_level_pop = (
                    source.emitting_molecule.Boltzmann_level_population(T=T_dust)
                )
                assert np.allclose(source.level_pop, expected_level_pop, atol=5e-2)
