#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:34:44 2024

@author: gianni
"""

import numpy as np
from scipy import constants
import numba as nb
import pytest
from pythonradex import (
    radiative_transfer,
    helpers,
    intensity,
    molecule,
    escape_probability,
)
import itertools
import os


def allowed_param_combination(geometry, line_profile_type):
    if "LVG" in geometry and line_profile_type == "Gaussian":
        return False
    else:
        return True


here = os.path.dirname(os.path.abspath(__file__))
CO_datafilepath = os.path.join(here, "LAMDA_files/co.dat")
HCl_datafilepath = os.path.join(here, "LAMDA_files/hcl.dat")
line_profile_types = ("rectangular", "Gaussian")
zero = lambda nu: np.zeros_like(nu)


class TestFastIntensities:

    test_transitions = nb.typed.List([0, 1, 3, 4, 5, 8])
    molecules = {
        line_profile_type: molecule.EmittingMolecule(
            datafilepath=CO_datafilepath,
            line_profile_type=line_profile_type,
            width_v=1 * constants.kilo,
        )
        for line_profile_type in ("rectangular", "Gaussian")
    }
    test_tau_values = [1e-5, 1e-2, 0.5, 10, 100]

    @staticmethod
    def calculate_nu_and_tau(lp, nu0, width_v, peak_tau):
        width_nu = width_v / constants.c * nu0
        if lp == "rectangular":
            nu = np.linspace(nu0 - 0.55 * width_nu, nu0 + 0.55 * width_nu, 10000)
            tau = np.where(np.abs(nu - nu0) < width_nu / 2, peak_tau, 0)
        elif lp == "Gaussian":
            nu = np.linspace(nu0 - 3 * width_nu, nu0 + 3 * width_nu, 10000)
            sigma_nu = helpers.FWHM2sigma(width_nu)
            tau = np.exp(-((nu - nu0) ** 2) / (2 * sigma_nu**2))
            tau *= peak_tau / np.max(tau)
        return nu, tau

    def test_fast_intensities(self):
        for (geo_name, geo), lp in itertools.product(
            radiative_transfer.Source.geometries.items(), ("Gaussian", "rectangular")
        ):
            if not allowed_param_combination(geometry=geo_name, line_profile_type=lp):
                continue
            is_LVG_sphere = geo_name == "LVG sphere"
            mol = self.molecules[lp]
            V_LVG_sphere = mol.width_v / 2
            T = 123
            Tex = np.ones(mol.n_rad_transitions, dtype=float) * T
            for test_tau in self.test_tau_values:
                expected_intensity = []
                for t in self.test_transitions:
                    S = helpers.B_nu(nu=mol.nu0[t], T=Tex[t])
                    nu, tau = self.calculate_nu_and_tau(
                        lp=lp, nu0=mol.nu0[t], width_v=mol.width_v, peak_tau=test_tau
                    )
                    intensity_kwargs = {"tau": tau, "source_function": S}
                    if is_LVG_sphere:
                        intensity_kwargs["nu"] = nu
                        intensity_kwargs["nu0"] = mol.nu0[t]
                        intensity_kwargs["V"] = V_LVG_sphere
                    expc_specific_I = geo.specific_intensity(**intensity_kwargs)
                    expected_intensity.append(np.trapezoid(expc_specific_I, nu))
                tau_nu0 = np.ones(mol.n_rad_transitions) * test_tau
                intensity_calculator = intensity.IntensityCalculator(
                    emitting_molecule=mol,
                    level_population=mol.Boltzmann_level_population(T),
                    geometry_name=geo_name,
                    specific_intensity=geo.specific_intensity,
                    tau_nu0_individual_transitions=tau_nu0,
                    tau_dust=zero,
                    S_dust=zero,
                    V_LVG_sphere=V_LVG_sphere,
                )
                if lp == "Gaussian":
                    calculated_intensity = intensity_calculator.fast_line_intensities_Gaussian_without_overlap(
                        transitions=self.test_transitions
                    )
                elif lp == "rectangular":
                    calculated_intensity = intensity_calculator.fast_line_intensities_rectangular_without_overlap(
                        transitions=self.test_transitions
                    )
                else:
                    raise ValueError
                assert np.allclose(
                    expected_intensity, calculated_intensity, atol=0, rtol=5e-3
                )

    def test_rectangular_tau_param_constructor(self):
        line_index = 3
        mol = self.molecules["rectangular"]
        test_line = mol.rad_transitions[line_index]
        N = 1e12 / constants.centi**2
        level_pop = mol.Boltzmann_level_population(33)
        N1 = N * level_pop[test_line.low.index]
        N2 = N * level_pop[test_line.up.index]
        width_nu = mol.width_v / constants.c * test_line.nu0
        # make nu asymmetric on purpose...
        nu = np.linspace(test_line.nu0 - 2 * width_nu, test_line.nu0 + width_nu, 400)
        # add the edge points, otherwise the interpolation will mess things up...
        nu = np.append(nu, (test_line.nu0 - width_nu / 2, test_line.nu0 + width_nu / 2))
        nu.sort()
        expected_tau = test_line.tau(N1=N1, N2=N2, nu=nu)
        geo_name = "static slab"
        geo = radiative_transfer.Source.geometries[geo_name]
        tau_nu0_individual_transitions = mol.get_tau_nu0_lines(
            N=N, level_population=level_pop
        )
        intensity_calculator = intensity.IntensityCalculator(
            emitting_molecule=mol,
            level_population=level_pop,
            geometry_name=geo_name,
            V_LVG_sphere=mol.width_v / 2,
            specific_intensity=geo.specific_intensity,
            tau_nu0_individual_transitions=tau_nu0_individual_transitions,
            tau_dust=zero,
            S_dust=zero,
        )
        transitions = [
            line_index,
        ]
        constructed_nu, constructed_tau, constructed_source_function = (
            intensity_calculator.get_intensity_parameters_for_rectangular(
                transitions=transitions
            )
        )
        interp_tau = np.interp(x=constructed_nu, xp=nu, fp=expected_tau)
        assert np.allclose(constructed_tau, interp_tau, atol=0, rtol=1e-3)
        Tex = mol.get_Tex(level_population=level_pop)[line_index]
        expected_S = helpers.B_nu(T=Tex, nu=nu)
        interp_S = np.interp(x=constructed_nu, xp=nu, fp=expected_S)
        assert np.allclose(interp_S, constructed_source_function, atol=0, rtol=1e-3)


class TestInvalidIntensityRequest:

    molecule_with_overlap = molecule.EmittingMolecule(
        datafilepath=HCl_datafilepath,
        line_profile_type="rectangular",
        width_v=30 * constants.kilo,
    )
    test_overlapping_transitions = [0, 1, 2]  # these are overlapping

    def generate_test_intensity_calculator(
        self, emitting_molecule, tau_nu0_individual_transitions, tau_dust, S_dust
    ):
        geometry_name = "static sphere"
        geometry = escape_probability.StaticSphere()
        level_population = emitting_molecule.Boltzmann_level_population(T=45)
        intensitycalculator = intensity.IntensityCalculator(
            emitting_molecule=emitting_molecule,
            level_population=level_population,
            geometry_name=geometry_name,
            V_LVG_sphere=1 * constants.kilo,
            specific_intensity=geometry.specific_intensity,
            tau_nu0_individual_transitions=tau_nu0_individual_transitions,
            tau_dust=tau_dust,
            S_dust=S_dust,
        )
        return intensitycalculator

    def test_tau_dust_condition(self):
        emitting_molecule = molecule.EmittingMolecule(
            datafilepath=CO_datafilepath,
            line_profile_type="rectangular",
            width_v=1 * constants.kilo,
        )
        tau_nu0_individual_transitions = np.ones(emitting_molecule.n_rad_transitions)
        S_dust = lambda nu: np.ones_like(nu)
        for tau_dust in (1e-2, 0.2, 1, 10):

            def tau_dust_func(nu):
                return np.ones_like(nu) * tau_dust

            intensitycalculator = self.generate_test_intensity_calculator(
                emitting_molecule=emitting_molecule,
                tau_nu0_individual_transitions=tau_nu0_individual_transitions,
                tau_dust=tau_dust_func,
                S_dust=S_dust,
            )
            all_transitions = np.arange(
                intensitycalculator.emitting_molecule.n_rad_transitions
            )
            if tau_dust > 0.1:
                with pytest.raises(ValueError):
                    intensitycalculator.intensities_of_individual_transitions(
                        transitions=all_transitions
                    )
            else:
                intensitycalculator.intensities_of_individual_transitions(
                    transitions=all_transitions
                )

        def variable_tau_dust(nu):
            return np.where(nu <= 1.1 * emitting_molecule.nu0[0], 10, 0.05)

        intensitycalculator = self.generate_test_intensity_calculator(
            emitting_molecule=emitting_molecule,
            tau_nu0_individual_transitions=tau_nu0_individual_transitions,
            tau_dust=variable_tau_dust,
            S_dust=S_dust,
        )
        with pytest.raises(ValueError):
            intensitycalculator.intensities_of_individual_transitions(
                transitions=[
                    0,
                ]
            )
        intensitycalculator.intensities_of_individual_transitions(transitions=[1, 2, 3])

    def test_total_tau_overlapping_lines(self):
        for tt in self.test_overlapping_transitions:
            assert self.molecule_with_overlap.overlapping_lines[tt] == [
                t for t in self.test_overlapping_transitions if t != tt
            ]
        test_tau = np.zeros(self.molecule_with_overlap.n_rad_transitions)
        test_tau[:3] = [1.2, 0.3, 4]
        S_dust = lambda nu: np.zeros_like(nu)
        tau_dust = lambda nu: np.zeros_like(nu)
        intensitycalculator = self.generate_test_intensity_calculator(
            emitting_molecule=self.molecule_with_overlap,
            tau_nu0_individual_transitions=test_tau,
            tau_dust=tau_dust,
            S_dust=S_dust,
        )
        tau_tot = intensitycalculator.determine_tau_of_overlapping_lines(
            transitions=self.test_overlapping_transitions
        )
        assert np.all(tau_tot == np.array((0.3 + 4, 1.2 + 4, 1.2 + 0.3)))

    def test_overlap_line_condition(self):
        S_dust = lambda nu: np.zeros_like(nu)
        tau_dust = lambda nu: np.zeros_like(nu)
        # optical depths of the test transitions:
        all_cases = {
            "valid": [[1e-3, 1e-2, 1e-3], [1e-3, 0, 1e-4]],
            "invalid": [[1, 1, 1], [0, 0, 1], [1e-3, 2e-1, 1e-2]],
        }
        for validity, cases in all_cases.items():
            tau_nu0 = np.zeros(self.molecule_with_overlap.n_rad_transitions)
            for case in cases:
                tau_nu0[:3] = case
                intensitycalculator = self.generate_test_intensity_calculator(
                    emitting_molecule=self.molecule_with_overlap,
                    tau_nu0_individual_transitions=tau_nu0,
                    tau_dust=tau_dust,
                    S_dust=S_dust,
                )
                all_transitions = np.arange(
                    intensitycalculator.emitting_molecule.n_rad_transitions
                )
                if validity == "invalid":
                    with pytest.raises(ValueError):
                        intensitycalculator.intensities_of_individual_transitions(
                            transitions=all_transitions
                        )
                else:
                    intensitycalculator.intensities_of_individual_transitions(
                        transitions=all_transitions
                    )


class TestVarious:

    width_v = 1 * constants.kilo
    N = 1e12 / constants.centi**2
    T = 344

    def generate_intensity_calculator(self, geo_name, tau_dust, S_dust):
        geo = radiative_transfer.Source.geometries[geo_name]
        for lp in ("rectangular", "Gaussian"):
            mol = molecule.EmittingMolecule(
                datafilepath=CO_datafilepath, line_profile_type=lp, width_v=self.width_v
            )
            level_population = mol.Boltzmann_level_population(self.T)
            tau_nu0 = mol.get_tau_nu0_lines(N=self.N, level_population=level_population)
            intensitycalculator = intensity.IntensityCalculator(
                emitting_molecule=mol,
                level_population=level_population,
                geometry_name=geo_name,
                specific_intensity=geo.specific_intensity,
                tau_nu0_individual_transitions=tau_nu0,
                tau_dust=tau_dust,
                S_dust=S_dust,
                V_LVG_sphere=mol.width_v / 2,
            )
            yield {
                "mol": mol,
                "intensitycalculator": intensitycalculator,
                "tau_nu0": tau_nu0,
                "level_population": level_population,
            }

    def test_tau_constructor(self):
        test_transitions = [0, 3, 10]
        for f in self.generate_intensity_calculator(
            geo_name="static sphere", tau_dust=zero, S_dust=zero
        ):
            for t in test_transitions:
                line = f["mol"].rad_transitions[t]
                width_nu = f["mol"].width_v / constants.c * line.nu0
                nu = np.linspace(line.nu0 - 2 * width_nu, line.nu0 + 2 * width_nu, 200)
                constructed_tau = f[
                    "intensitycalculator"
                ].construct_tau_individual_line(
                    line=line, nu=nu, tau_nu0=f["tau_nu0"][t]
                )
                N1 = self.N * f["level_population"][line.low.index]
                N2 = self.N * f["level_population"][line.up.index]
                expected_tau = line.tau(N1=N1, N2=N2, nu=nu)
                assert np.allclose(
                    constructed_tau, expected_tau, atol=0, rtol=1e-3
                )

    def test_line_identification(self):
        trans_indices = [2, 3]
        for f in self.generate_intensity_calculator(
            geo_name="static sphere", tau_dust=zero, S_dust=zero
        ):
            intensitycalculator = f["intensitycalculator"]
            transitions = [f["mol"].rad_transitions[i] for i in trans_indices]
            freq_widths = [
                helpers.Delta_nu(Delta_v=self.width_v, nu0=t.nu0) for t in transitions
            ]
            for i, trans, width_nu in zip(trans_indices, transitions, freq_widths):
                include = [
                    np.array((trans.nu0,)),
                    np.array((trans.nu0 - 0.1 * width_nu, trans.nu0 + 0.1 * width_nu)),
                    # now ranges that do not enclose nu0:
                    np.array((trans.nu0 + 0.1 * width_nu, trans.nu0 + 0.4 * width_nu)),
                    # include nu outside of nu_min,nu_max:
                    np.array(
                        (
                            trans.nu0 + 0.1 * width_nu,
                            trans.nu0 + 0.4 * width_nu,
                            trans.nu0 + 100 * width_nu,
                        )
                    ),
                    np.array(
                        (
                            trans.line_profile.nu_min - width_nu / 3,
                            trans.nu0 + 0.1 * width_nu,
                            trans.nu0 + 0.4 * width_nu,
                            trans.nu0 + 100 * width_nu,
                        )
                    ),
                    np.array((trans.line_profile.nu_max - width_nu / 300,)),
                    np.array((trans.line_profile.nu_min + width_nu / 100,)),
                ]
                for nu in include:
                    intensitycalculator.set_nu(nu=nu)
                    assert len(intensitycalculator.nu_selected_line_indices) == 1
                    assert i in intensitycalculator.nu_selected_line_indices
                not_included = [
                    np.array((1.01 * trans.line_profile.nu_max,)),
                    np.array((0.99 * trans.line_profile.nu_min,)),
                    np.array((trans.line_profile.nu_min - 0.001 * width_nu,)),
                    np.array((trans.line_profile.nu_max + 0.001 * width_nu,)),
                    # broad range
                    np.array(
                        (
                            trans.line_profile.nu_min - 3 * width_nu,
                            trans.line_profile.nu_max + 3 * width_nu,
                        )
                    ),
                ]
                for nu in not_included:
                    intensitycalculator.set_nu(nu=nu)
                    assert len(intensitycalculator.nu_selected_line_indices) == 0
            # test also ranges that cover both transitions
            both = [
                np.array(
                    (
                        transitions[0].nu0,
                        transitions[1].nu0,
                    )
                ),
                np.array(
                    (
                        transitions[0].line_profile.nu_min - 5 * width_nu,
                        transitions[0].nu0,
                        transitions[1].nu0,
                        transitions[1].line_profile.nu_max + 5 * width_nu,
                    )
                ),
                np.array(
                    (
                        transitions[0].nu0,
                        transitions[1].line_profile.nu_max - width_nu / 100,
                    )
                ),
            ]
            for nu in both:
                intensitycalculator.set_nu(nu=nu)
                assert len(intensitycalculator.nu_selected_line_indices) == 2
                for i in trans_indices:
                    assert i in intensitycalculator.nu_selected_line_indices
            # nu that should include all transitions:
            all_inclusive_nu = [
                np.array([t.nu0 for t in f["mol"].rad_transitions]),
                np.array(
                    [
                        t.nu0 + t.line_profile.width_nu / 6
                        for t in f["mol"].rad_transitions
                    ]
                ),
            ]
            for nu in all_inclusive_nu:
                intensitycalculator.set_nu(nu=nu)
                assert len(intensitycalculator.nu_selected_line_indices) == len(
                    f["mol"].rad_transitions
                )
                assert np.all(
                    intensitycalculator.nu_selected_line_indices
                    == np.array(range(len(f["mol"].rad_transitions)))
                )

    @staticmethod
    def get_no_line_nu(trans, width_nu):
        return [
            np.array(
                (
                    trans.line_profile.nu_min - width_nu,
                    trans.line_profile.nu_min - 0.5 * width_nu,
                )
            ),
            np.array(
                (
                    trans.line_profile.nu_max + 0.5 * width_nu,
                    trans.line_profile.nu_max + width_nu,
                )
            ),
        ]

    def check_no_line_spectrum(self, intensitycalculator, trans, expected_spec):
        width_nu = self.width_v / constants.c * trans.nu0
        no_line = self.get_no_line_nu(trans=trans, width_nu=width_nu)
        for nu in no_line:
            intensitycalculator.set_nu(nu=nu)
            assert len(intensitycalculator.nu_selected_lines) == 0
            spec = intensitycalculator.specific_intensity_spectrum()
            assert np.all(spec == expected_spec(nu))

    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_no_line_spectrum(self):
        # if no lines are detected, spectrum should be zero
        def zero_spec(nu):
            return np.zeros_like(nu)

        for geo_name in ("static sphere", "LVG sphere"):
            for f in self.generate_intensity_calculator(
                geo_name=geo_name, tau_dust=zero, S_dust=zero
            ):
                self.check_no_line_spectrum(
                    intensitycalculator=f["intensitycalculator"],
                    trans=f["mol"].rad_transitions[1],
                    expected_spec=zero_spec,
                )

        # test with dust:
        def S_dust(nu):
            return helpers.B_nu(nu=nu, T=123)

        def tau_dust(nu):
            return np.ones_like(nu) * 0.56

        def dust_spec(nu):
            return S_dust(nu) * (1 - np.exp(-tau_dust(nu)))

        for f in self.generate_intensity_calculator(
            geo_name="static slab", tau_dust=tau_dust, S_dust=S_dust
        ):
            self.check_no_line_spectrum(
                intensitycalculator=f["intensitycalculator"],
                trans=f["mol"].rad_transitions[1],
                expected_spec=dust_spec,
            )

    def test_single_nu_spectrum(self):
        for geo_name in ("static sphere", "LVG sphere"):
            for f in self.generate_intensity_calculator(
                geo_name=geo_name, tau_dust=zero, S_dust=zero
            ):
                intensitycalculator = f["intensitycalculator"]
                trans = f["mol"].rad_transitions[5]
                width_nu = self.width_v / constants.c * trans.nu0
                single_nu = np.array((trans.nu0 - width_nu / 6,))
                multi_nu = np.array(
                    (trans.line_profile.nu_min, single_nu[0], trans.line_profile.nu_max)
                )
                intensitycalculator.set_nu(nu=single_nu)
                single_spec = intensitycalculator.specific_intensity_spectrum()
                intensitycalculator.set_nu(nu=multi_nu)
                multi_spec = intensitycalculator.specific_intensity_spectrum()
                assert single_spec[0] == multi_spec[1]


class TestSpecificIntensityNu0NoOverlap:

    Tkin = 100

    def generate_intensity_calculator(
        self, geometry_name, line_profile_type, S_dust, tau_dust, width_v, datafilepath
    ):
        mol = molecule.EmittingMolecule(
            datafilepath=datafilepath,
            line_profile_type=line_profile_type,
            width_v=width_v,
        )
        level_population = mol.Boltzmann_level_population(T=self.Tkin)
        geometry = radiative_transfer.Source.geometries[geometry_name]
        tau_nu0_individual_transitions = (
            np.arange(mol.n_rad_transitions, dtype=float) + 0.1
        )
        tau_nu0_individual_transitions = tau_nu0_individual_transitions[::-1]
        tau_nu0_individual_transitions /= np.max(tau_nu0_individual_transitions)
        intensitycalculator = intensity.IntensityCalculator(
            emitting_molecule=mol,
            level_population=level_population,
            geometry_name=geometry_name,
            V_LVG_sphere=width_v / 2,
            specific_intensity=geometry.specific_intensity,
            tau_nu0_individual_transitions=tau_nu0_individual_transitions,
            tau_dust=tau_dust,
            S_dust=S_dust,
        )
        return intensitycalculator

    def test_specific_intensity_nu0(self):
        for geo, line_profile_type in itertools.product(
            radiative_transfer.Source.geometries.keys(),
            radiative_transfer.Source.line_profiles.keys(),
        ):
            if "LVG" in geo and line_profile_type == "Gaussian":
                continue
            kwargs = {
                "no dust": {
                    "geometry_name": geo,
                    "line_profile_type": line_profile_type,
                    "S_dust": zero,
                    "tau_dust": zero,
                    "width_v": 1 * constants.kilo,
                    "datafilepath": CO_datafilepath,
                }
            }
            if "LVG" not in geo:
                kwargs["dust"] = kwargs["no dust"].copy()
                kwargs["dust"]["S_dust"] = lambda nu: helpers.B_nu(nu=nu, T=200)
                kwargs["dust"]["tau_dust"] = lambda nu: np.ones_like(nu) * 0.5
            for kw in kwargs.values():
                f = self.generate_intensity_calculator(**kw)
                transitions = np.arange(f.emitting_molecule.n_rad_transitions)
                specific_intensity_nu0 = f.specific_intensity_nu0_no_overlap(
                    transitions=transitions
                )
                nu = f.emitting_molecule.nu0
                f.set_nu(nu=nu)
                expected_from_spec = f.specific_intensity_spectrum()
                assert np.allclose(
                    specific_intensity_nu0, expected_from_spec, atol=0, rtol=1e-3
                )
                # now explicit calculation:
                tau_dust = kw["tau_dust"](nu)
                S_dust = kw["S_dust"](nu)
                tau_tot = f.tau_nu0_individual_transitions + tau_dust
                S_lines = helpers.B_nu(nu=nu, T=f.Tex)
                source_function = np.where(
                    tau_tot == 0,
                    0,
                    (f.tau_nu0_individual_transitions * S_lines + tau_dust * S_dust)
                    / tau_tot,
                )
                intensity_kwargs = {
                    "tau": tau_tot,
                    "source_function": source_function,
                }
                if "LVG" in geo:
                    explicit = escape_probability.specific_intensity_nu0_lvg_sphere(
                        **intensity_kwargs
                    )
                else:
                    explicit = f.specific_intensity(**intensity_kwargs)
                assert np.allclose(specific_intensity_nu0, explicit, atol=0, rtol=1e-3)

    def test_thick_intensity(self):
        Tkin = 123
        Tdust = 25
        ext_background = helpers.generate_CMB_background()
        collider_densities = {
            "ortho-H2": 1e9 * constants.centi**-3,
            "para-H2": 1e9 * constants.centi**-3,
        }
        for geo, line_profile_type in itertools.product(
            radiative_transfer.Source.geometries.keys(),
            radiative_transfer.Source.line_profiles.keys(),
        ):
            if "LVG" in geo and line_profile_type == "Gaussian":
                continue
            source = radiative_transfer.Source(
                datafilepath=CO_datafilepath,
                geometry=geo,
                line_profile_type=line_profile_type,
                width_v=1 * constants.kilo,
            )
            # thick line:
            source.update_parameters(
                N=1e20 * constants.centi**-2,
                Tkin=Tkin,
                collider_densities=collider_densities,
                ext_background=ext_background,
                T_dust=0,
                tau_dust=0,
            )
            source.solve_radiative_transfer()
            test_transitions = [0, 1, 2]
            assert np.all(source.tau_nu0_individual_transitions[test_transitions] > 10)
            specific_intensity_nu0 = (
                source.intensity_calculator.specific_intensity_nu0_no_overlap(
                    transitions=test_transitions
                )
            )
            expected = helpers.B_nu(
                T=Tkin, nu=source.emitting_molecule.nu0[test_transitions]
            )
            assert np.allclose(specific_intensity_nu0, expected, atol=0, rtol=1e-3)
            # thick dust:
            if not "LVG" in geo:
                source.update_parameters(
                    N=1e12 * constants.centi**-2,
                    Tkin=Tkin,
                    collider_densities=collider_densities,
                    ext_background=ext_background,
                    T_dust=Tdust,
                    tau_dust=50,
                )
                source.solve_radiative_transfer()
                specific_intensity_nu0 = (
                    source.intensity_calculator.specific_intensity_nu0_no_overlap(
                        transitions=test_transitions
                    )
                )
                expected = helpers.B_nu(
                    T=Tdust, nu=source.emitting_molecule.nu0[test_transitions]
                )
                assert np.allclose(specific_intensity_nu0, expected, atol=0, rtol=1e-3)

    def test_reject_overlap(self):
        intensitycalculator = self.generate_intensity_calculator(
            geometry_name="static sphere",
            line_profile_type="rectangular",
            S_dust=zero,
            tau_dust=zero,
            width_v=10 * constants.kilo,
            datafilepath=HCl_datafilepath,
        )
        assert intensitycalculator.emitting_molecule.has_overlapping_lines
        transitions = [
            1,
        ]
        with pytest.raises(AssertionError):
            intensitycalculator.specific_intensity_nu0_no_overlap(
                transitions=transitions
            )


###### tests using physics ###################


class TestIntensitiesWithPhysics:

    CO_molecules = {
        line_profile_type: molecule.EmittingMolecule(
            datafilepath=CO_datafilepath,
            line_profile_type=line_profile_type,
            width_v=1 * constants.kilo,
        )
        for line_profile_type in ("rectangular", "Gaussian")
    }
    HCl_molecules = {
        line_profile_type: molecule.EmittingMolecule(
            datafilepath=HCl_datafilepath,
            line_profile_type=line_profile_type,
            width_v=30 * constants.kilo,
        )
        for line_profile_type in ("rectangular", "Gaussian")
    }
    HCL_overlapping_transitions = (0, 1, 2)
    distance = 1 * constants.parsec
    sphere_radius = 1 * constants.au
    sphere_volume = 4 / 3 * sphere_radius**3 * np.pi
    sphere_Omega = sphere_radius**2 * np.pi / distance**2
    Tkin = 45
    tau_dust = {"thin": 1e-4, "thick": 50}
    T_dust = 100
    assert Tkin != T_dust, "test needs different Tkin and Tdust"

    def S_dust(self, nu):
        return helpers.B_nu(nu=nu, T=self.T_dust)

    def LTE_intensitycalc_iterator(self, specie, N, tau_dust, S_dust):
        # assume LTE for convenience
        if specie == "CO":
            molecules = self.CO_molecules
        elif specie == "HCl":
            molecules = self.HCl_molecules
        for lp, mol in molecules.items():
            level_pop = mol.Boltzmann_level_population(self.Tkin)
            tau_nu0 = mol.get_tau_nu0_lines(N=N, level_population=level_pop)
            for geo_name, geo in radiative_transfer.Source.geometries.items():
                if not allowed_param_combination(
                    geometry=geo_name, line_profile_type=lp
                ):
                    continue
                intensitycalculator = intensity.IntensityCalculator(
                    emitting_molecule=mol,
                    level_population=level_pop,
                    geometry_name=geo_name,
                    V_LVG_sphere=mol.width_v / 2,
                    specific_intensity=geo.specific_intensity,
                    tau_nu0_individual_transitions=tau_nu0,
                    tau_dust=tau_dust,
                    S_dust=S_dust,
                )
                yield lp, level_pop, tau_nu0, intensitycalculator

    def test_HCl_lines_are_overlapping(self):
        for mol in self.HCl_molecules.values():
            for t in self.HCL_overlapping_transitions:
                assert mol.overlapping_lines[t] == [
                    i for i in self.HCL_overlapping_transitions if i != t
                ]

    def test_thin(self):
        N = 1e12 / constants.centi**2
        for (
            lp,
            level_pop,
            tau_nu0,
            intensitycalculator,
        ) in self.LTE_intensitycalc_iterator(
            specie="CO", N=N, tau_dust=zero, S_dust=zero
        ):
            transitions = np.arange(
                intensitycalculator.emitting_molecule.n_rad_transitions
            )
            intensities = intensitycalculator.intensities_of_individual_transitions(
                transitions=transitions
            )
            expected_intensities = []
            for i, line in enumerate(
                intensitycalculator.emitting_molecule.rad_transitions
            ):
                up_level_pop = level_pop[line.up.index]
                if intensitycalculator.geometry_name in ("static sphere", "LVG sphere"):
                    # in the case of spheres, we can do an elegant test
                    # using physics
                    number_density = N / (2 * self.sphere_radius)
                    total_mol = number_density * self.sphere_volume
                    I = (
                        total_mol
                        * up_level_pop
                        * line.A21
                        * line.Delta_E
                        / (4 * np.pi * self.distance**2)
                        / self.sphere_Omega
                    )
                else:
                    # for slabs, could not come up with elegant test
                    I_nu0 = helpers.B_nu(nu=line.nu0, T=self.Tkin) * tau_nu0[i]
                    if lp == "Gaussian":
                        I = np.sqrt(2 * np.pi) * line.line_profile.sigma_nu * I_nu0
                    elif lp == "rectangular":
                        I = I_nu0 * line.line_profile.width_nu
                expected_intensities.append(I)
            expected_intensities = np.array(expected_intensities)
            assert np.allclose(intensities, expected_intensities, atol=0, rtol=5e-3)
            assert np.allclose(tau_nu0, 0, atol=1e-3, rtol=0)

    def test_thick(self):
        N = 1e19 / constants.centi**2
        for (
            lp,
            level_pop,
            tau_nu0,
            intensitycalculator,
        ) in self.LTE_intensitycalc_iterator(
            specie="CO", N=N, tau_dust=zero, S_dust=zero
        ):
            transitions = np.arange(
                intensitycalculator.emitting_molecule.n_rad_transitions
            )
            intensities = intensitycalculator.intensities_of_individual_transitions(
                transitions=transitions
            )
            thick_lines = tau_nu0 > 10
            assert thick_lines.sum() >= 10
            for i, line in enumerate(
                intensitycalculator.emitting_molecule.rad_transitions
            ):
                if not thick_lines[i]:
                    continue
                bb_intensity_nu0 = helpers.B_nu(nu=line.nu0, T=self.Tkin)
                # can only test rectangular profiles; also, cannot test LVG sphere
                # because it has a different spectral shape than a rectangle
                if (
                    lp == "rectangular"
                    and intensitycalculator.geometry_name != "LVG sphere"
                ):
                    expected_total_intensity = (
                        bb_intensity_nu0 * line.line_profile.width_nu
                    )
                    assert np.isclose(
                        a=expected_total_intensity, b=intensities[i], atol=0, rtol=3e-2
                    )
                else:
                    assert (
                        lp == "Gaussian"
                        or intensitycalculator.geometry_name == "LVG sphere"
                    )

    def get_line_covering_nu(self, lines, width_v):
        nu0s = [line.nu0 for line in lines]
        width_nu = width_v / constants.c * np.mean(nu0s)
        min_nu0 = np.min(nu0s)
        max_nu0 = np.max(nu0s)
        nu = np.linspace(min_nu0 - 3 * width_nu, max_nu0 + 3 * width_nu, 3000)
        return nu

    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_spectrum_single_lines(self):
        N_values = {
            "thin": 1e12 / constants.centi**2,
            "thick": 1e19 / constants.centi**2,
        }
        test_transitions = [0, 2, 5]
        for ID, N in N_values.items():
            iterator = self.LTE_intensitycalc_iterator(
                specie="CO", N=N, tau_dust=zero, S_dust=zero
            )
            for lp, level_pop, tau_nu0, intensitycalculator in iterator:
                width_v = intensitycalculator.emitting_molecule.width_v
                for t in test_transitions:
                    line = intensitycalculator.emitting_molecule.rad_transitions[t]
                    nu = self.get_line_covering_nu(
                        lines=[
                            line,
                        ],
                        width_v=width_v,
                    )
                    intensitycalculator.set_nu(nu=nu)
                    spec = intensitycalculator.specific_intensity_spectrum()
                    assert np.all(spec >= 0)
                    source_func = helpers.B_nu(T=self.Tkin, nu=nu)
                    if intensitycalculator.geometry_name == "LVG sphere":
                        LVG_kwargs = {"nu": nu, "nu0": line.nu0, "V": width_v / 2}
                    else:
                        LVG_kwargs = {}
                    N1 = level_pop[line.low.index] * N
                    N2 = level_pop[line.up.index] * N
                    expected_tau = line.tau(N1, N2, nu)
                    expected_spec = intensitycalculator.specific_intensity(
                        tau=expected_tau,
                        source_function=source_func,
                        **LVG_kwargs,
                    )
                    assert np.allclose(spec, expected_spec, atol=0, rtol=1e-3)
                    if ID == "thick":
                        bb_intensity_nu0 = helpers.B_nu(nu=line.nu0, T=self.Tkin)
                        peak = np.max(spec)
                        assert np.isclose(a=peak, b=bb_intensity_nu0, atol=0, rtol=3e-2)
                if ID == "thin":
                    assert np.allclose(expected_tau, 0, atol=1e-3, rtol=0)

    @pytest.mark.filterwarnings("ignore:LVG sphere geometry")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_tau_and_spectrum_overlapping_lines_without_overlap_treatment(self):
        N = 1e10 / constants.centi**2
        for (
            lp,
            level_pop,
            tau_nu0,
            intensitycalculator,
        ) in self.LTE_intensitycalc_iterator(
            specie="HCl", N=N, tau_dust=zero, S_dust=zero
        ):
            lines = [
                intensitycalculator.emitting_molecule.rad_transitions[t]
                for t in self.HCL_overlapping_transitions
            ]
            width_v = intensitycalculator.emitting_molecule.width_v
            nu = self.get_line_covering_nu(lines=lines, width_v=width_v)
            intensitycalculator.set_nu(nu=nu)
            spec = intensitycalculator.specific_intensity_spectrum()
            expected_spec = np.zeros_like(spec)
            expected_tau = np.zeros_like(spec)
            for line in lines:
                N1 = level_pop[line.low.index] * N
                N2 = level_pop[line.up.index] * N
                tau_line = line.tau(N1, N2, nu)
                expected_tau += tau_line
                source_func = helpers.B_nu(T=self.Tkin, nu=nu)
                if intensitycalculator.geometry_name == "LVG sphere":
                    LVG_kwargs = {"nu": nu, "nu0": line.nu0, "V": width_v / 2}
                else:
                    LVG_kwargs = {}
                expected_spec += intensitycalculator.specific_intensity(
                    tau=tau_line, source_function=source_func, **LVG_kwargs
                )
            assert np.allclose(
                intensitycalculator.tau_tot, expected_tau, atol=1e-20, rtol=1e-3
            )
            assert np.allclose(spec, expected_spec, atol=0, rtol=1e-3)
            assert np.allclose(intensitycalculator.tau_tot, 0, atol=1e-3, rtol=0)

    def test_spectrum_with_dust(self):
        N_values = {
            "CO": {
                "thin": 1e12 / constants.centi**2,
                "thick": 1e19 / constants.centi**2,
            },
            "HCl": {
                "thin": 1e10 / constants.centi**2,
                "thick": 1e17 / constants.centi**2,
            },
        }
        transitions = {
            "CO": [
                2,
            ],
            "HCl": self.HCL_overlapping_transitions,
        }
        for dust_thickness, tau_dust_value in self.tau_dust.items():
            tau_dust = lambda nu: np.ones_like(nu) * tau_dust_value
            for specie in ("CO", "HCl"):
                for line_thickness, N in N_values[specie].items():
                    for (
                        lp,
                        level_pop,
                        tau_nu0,
                        intensitycalculator,
                    ) in self.LTE_intensitycalc_iterator(
                        specie=specie, N=N, tau_dust=tau_dust, S_dust=self.S_dust
                    ):
                        if "LVG" in intensitycalculator.geometry_name:
                            # LVG does not allow dust
                            continue
                        lines = [
                            intensitycalculator.emitting_molecule.rad_transitions[t]
                            for t in transitions[specie]
                        ]
                        width_v = intensitycalculator.emitting_molecule.width_v
                        nu = self.get_line_covering_nu(lines=lines, width_v=width_v)
                        intensitycalculator.set_nu(nu=nu)
                        spec = intensitycalculator.specific_intensity_spectrum()
                        if line_thickness == "thin" and dust_thickness == "thick":
                            # dust should dominate
                            for tau_line in intensitycalculator.tau_lines:
                                assert np.allclose(tau_line, 0, atol=1e-3, rtol=0)
                            expected_spec = self.S_dust(nu=nu)
                            assert np.allclose(spec, expected_spec, atol=0, rtol=1e-3)
                        elif line_thickness == "thick" and dust_thickness == "thick":
                            # at nu0, expect mix of line and dust
                            for line in lines:
                                S_line_nu0 = helpers.B_nu(nu=line.nu0, T=self.Tkin)
                                tau_nu0_alllines = 0
                                for lineline in lines:
                                    tau_nu0_alllines += lineline.tau(
                                        N1=N * level_pop[lineline.low.index],
                                        N2=N * level_pop[lineline.up.index],
                                        nu=line.nu0,
                                    )
                                S_dust_nu0 = helpers.B_nu(nu=line.nu0, T=self.T_dust)
                                tau_dust_nu0 = tau_dust(line.nu0)
                                S_tot_nu0 = (
                                    tau_nu0_alllines * S_line_nu0
                                    + tau_dust_nu0 * S_dust_nu0
                                )
                                S_tot_nu0 /= tau_nu0_alllines + tau_dust_nu0
                                nu0_index = np.argmin(np.abs(nu - line.nu0))
                                expected_spec_nu0 = S_tot_nu0
                                assert np.isclose(
                                    expected_spec_nu0,
                                    spec[nu0_index],
                                    atol=0,
                                    rtol=1e-3,
                                )
                        elif line_thickness == "thin" and dust_thickness == "thin":
                            # expect that I can just add together the line and dust specs
                            dust_S = helpers.B_nu(nu=nu, T=self.T_dust)
                            dust_spec = intensitycalculator.specific_intensity(
                                tau=tau_dust(nu), source_function=dust_S
                            )
                            expected_spec = dust_spec
                            S_line = helpers.B_nu(nu=nu, T=self.Tkin)
                            for tau_line in intensitycalculator.tau_lines:
                                line_spec = intensitycalculator.specific_intensity(
                                    tau=tau_line, source_function=S_line
                                )
                                expected_spec += line_spec
                            assert np.allclose(spec, expected_spec, atol=0, rtol=1e-3)
                        elif line_thickness == "thick" and dust_thickness == "thin":
                            # expect line dominating at nu0
                            for line in lines:
                                S_nu0 = helpers.B_nu(nu=line.nu0, T=self.Tkin)
                                nu0_index = np.argmin(np.abs(nu - line.nu0))
                                assert np.isclose(
                                    spec[nu0_index], S_nu0, atol=0, rtol=1e-3
                                )
                        else:
                            raise RuntimeError
                        # for all cases, at the wavelength grid edges, we should only have
                        # dust emission
                        nu_edge = nu[[0, -1]]
                        expected_dust_spec = intensitycalculator.specific_intensity(
                            tau=tau_dust(nu_edge),
                            source_function=helpers.B_nu(nu=nu_edge, T=self.T_dust),
                        )
                        assert np.allclose(
                            spec[[0, -1]], expected_dust_spec, atol=0, rtol=1e-3
                        )
