# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:35:53 2017

@author: gianni
"""
import os
from pythonradex import molecule, atomic_transition, LAMDA_file, helpers
import numpy as np
import itertools
from scipy import constants
import pytest

here = os.path.dirname(os.path.abspath(__file__))
lamda_folder = os.path.join(here, "LAMDA_files")
lamda_filepaths = {
    "with_freq": os.path.join(lamda_folder, "co.dat"),
    "no_freq": os.path.join(lamda_folder, "co_no_frequencies.dat"),
}
test_data = {
    label: LAMDA_file.read(datafilepath=lamda_filepaths[label], read_frequencies=rf)
    for label, rf in zip(("with_freq", "no_freq"), (True, False))
}

line_profile_type = "rectangular"
width_v = 1 * constants.kilo

molecules = {
    ID: molecule.Molecule(datafilepath=df) for ID, df in lamda_filepaths.items()
}
emitting_molecules = {
    ID: molecule.EmittingMolecule(
        datafilepath=df, line_profile_type=line_profile_type, width_v=width_v
    )
    for ID, df in lamda_filepaths.items()
}

test_molecules_no_freq = [molecules["no_freq"], emitting_molecules["no_freq"]]
test_molecules_with_freq = [molecules["with_freq"], emitting_molecules["with_freq"]]
test_molecules = {
    "no_freq": test_molecules_no_freq,
    "with_freq": test_molecules_with_freq,
}


def test_Molecule_levels():
    for ID, td in test_data.items():
        for i, level in enumerate(td["levels"]):
            for mol in test_molecules[ID]:
                for attribute in ("g", "E", "index"):
                    assert getattr(level, attribute) == getattr(
                        mol.levels[i], attribute
                    )


def test_Molecule_rad_transitions():
    for ID, td in test_data.items():
        for i, rad_trans in enumerate(td["radiative transitions"]):
            for mol in test_molecules[ID]:
                for attribute in ("Delta_E", "A21", "B12", "nu0", "B21"):
                    assert getattr(rad_trans, attribute) == getattr(
                        mol.rad_transitions[i], attribute
                    )


def test_Molecule_coll_transitions():
    for ID, td in test_data.items():
        colliders = sorted(td["collisional transitions"].keys())
        for mol in test_molecules[ID]:
            assert colliders == sorted(mol.coll_transitions.keys())
    for ID, td in test_data.items():
        for collider, coll_transitions in td["collisional transitions"].items():
            for i, coll_trans in enumerate(coll_transitions):
                for mol in test_molecules[ID]:
                    for attribute in ("K21_data", "Tkin_data"):
                        assert np.all(
                            getattr(coll_trans, attribute)
                            == getattr(mol.coll_transitions[collider][i], attribute)
                        )


def test_setting_partition_function():
    for fp in lamda_filepaths.values():
        test_mol = molecule.Molecule(datafilepath=fp, partition_function=lambda x: -10)
        assert test_mol.Z(100) == -10


def test_partition_func():
    T = 50
    for mol in itertools.chain.from_iterable(test_molecules.values()):
        Q = 0
        for level in mol.levels:
            Q += level.g * np.exp(-level.E / (constants.k * T))
        assert np.isclose(Q, mol.Z(T), atol=0, rtol=1e-10)


def test_Boltzmann_level_population():
    T = 30
    for mol in itertools.chain.from_iterable(test_molecules.values()):
        Boltzmann_level_population = mol.Boltzmann_level_population(T=T)
        assert np.isclose(np.sum(Boltzmann_level_population), 1)
        Q = mol.Z(T=T)
        for i, level in enumerate(mol.levels):
            expected_pop = level.g * np.exp(-level.E / (constants.k * T)) / Q
            assert expected_pop == Boltzmann_level_population[i]


def test_get_transition_number():
    for mol in itertools.chain.from_iterable(test_molecules.values()):
        rad_trans_number = mol.get_rad_transition_number("11-10")
        assert rad_trans_number == 10


rng = np.random.default_rng(seed=0)


def test_tau():
    N_values = np.array((1e12, 1e14, 1e16, 1e18)) / constants.centi**2
    for N in N_values:
        for mol in emitting_molecules.values():
            level_population = rng.random(mol.n_levels)
            level_population /= np.sum(level_population)
            tau_nu0 = mol.get_tau_nu0_lines(N=N, level_population=level_population)
            expected_tau_nu0 = []
            for i, rad_trans in enumerate(mol.rad_transitions):
                i_up = rad_trans.up.index
                N2 = N * level_population[i_up]
                i_low = rad_trans.low.index
                N1 = N * level_population[i_low]
                t = atomic_transition.tau_nu(
                    A21=rad_trans.A21,
                    phi_nu=rad_trans.line_profile.phi_nu(rad_trans.nu0),
                    g_low=rad_trans.low.g,
                    g_up=rad_trans.up.g,
                    N1=N1,
                    N2=N2,
                    nu=rad_trans.nu0,
                )
                expected_tau_nu0.append(t)
            assert np.all(expected_tau_nu0 == tau_nu0)


def test_tau_LTE():
    N_values = np.array((1e12, 1e14, 1e16, 1e18)) / constants.centi**2
    T = 123
    for N in N_values:
        for mol in emitting_molecules.values():
            level_population = mol.Boltzmann_level_population(T=T)
            tau_nu0_LTE = mol.get_tau_nu0_lines_LTE(N=N, T=T)
            expected_tau_nu0 = []
            for i, rad_trans in enumerate(mol.rad_transitions):
                i_up = rad_trans.up.index
                N2 = N * level_population[i_up]
                i_low = rad_trans.low.index
                N1 = N * level_population[i_low]
                t = atomic_transition.tau_nu(
                    A21=rad_trans.A21,
                    phi_nu=rad_trans.line_profile.phi_nu(rad_trans.nu0),
                    g_low=rad_trans.low.g,
                    g_up=rad_trans.up.g,
                    N1=N1,
                    N2=N2,
                    nu=rad_trans.nu0,
                )
                expected_tau_nu0.append(t)
            assert np.all(expected_tau_nu0 == tau_nu0_LTE)


def test_LTE_Tex():
    T = 123
    for mol in emitting_molecules.values():
        Boltzmann_level_population = mol.Boltzmann_level_population(T=T)
        Tex = mol.get_Tex(level_population=Boltzmann_level_population)
        assert np.allclose(a=T, b=Tex, atol=0, rtol=1e-10)


def test_get_tau_line_nu():
    N = 1e15 / constants.centi**2
    for mol in emitting_molecules.values():
        level_pop = mol.Boltzmann_level_population(T=214)
        tau_line_funcs = [
            mol.get_tau_line_nu(line_index=i, level_population=level_pop, N=N)
            for i in range(mol.n_rad_transitions)
        ]
        for i, line in enumerate(mol.rad_transitions):
            width_nu = width_v / constants.c * line.nu0
            nu = np.linspace(line.nu0 - width_nu, line.nu0 + width_nu, 100)
            expected_tau = line.tau_nu(
                N1=N * level_pop[line.low.index], N2=N * level_pop[line.up.index], nu=nu
            )
            assert np.all(expected_tau == tau_line_funcs[i](nu))


def test_Tex():
    for mol in list(emitting_molecules.values()) + list(molecules.values()):
        level_population = rng.random(mol.n_levels)
        level_population /= np.sum(level_population)
        Tex = mol.get_Tex(level_population=level_population)
        expected_Tex = []
        for i, rad_trans in enumerate(mol.rad_transitions):
            i_up = rad_trans.up.index
            x2 = level_population[i_up]
            i_low = rad_trans.low.index
            x1 = level_population[i_low]
            tex = atomic_transition.Tex(
                Delta_E=rad_trans.Delta_E,
                g_low=rad_trans.low.g,
                g_up=rad_trans.up.g,
                x1=x1,
                x2=x2,
            )
            expected_Tex.append(tex)
        assert np.all(Tex == expected_Tex)


def test_coll_Tkin_data():
    for mol in emitting_molecules.values():
        for collider, coll_transitions in mol.coll_transitions.items():
            for coll_trans in coll_transitions:
                assert np.all(coll_trans.Tkin_data == mol.Tkin_data[collider])
            assert mol.Tkin_data_limits[collider] == (
                np.min(mol.Tkin_data[collider]),
                np.max(mol.Tkin_data[collider]),
            )


def test_K21_matrix():
    for mol in emitting_molecules.values():
        for collider, coll_transitions in mol.coll_transitions.items():
            for i, coll_trans in enumerate(coll_transitions):
                assert np.all(mol.K21_matrix[collider][i, :] == coll_trans.K21_data)


def test_K_matrix():
    for mol in emitting_molecules.values():
        for collider, coll_transitions in mol.coll_transitions.items():
            T_index = 1
            K21 = mol.K21_matrix[collider][:, T_index]
            K12 = atomic_transition.compute_K12(
                K21=K21,
                g_up=mol.coll_gups[collider],
                g_low=mol.coll_glows[collider],
                Delta_E=mol.coll_DeltaEs[collider],
                Tkin=mol.Tkin_data[collider][T_index],
            )
            K_contribution = mol.construct_K_matrix(
                n_levels=mol.n_levels,
                K12=K12,
                K21=K21,
                ilow=mol.coll_ilow[collider],
                iup=mol.coll_iup[collider],
            )
            expected_K = np.zeros((mol.n_levels,) * 2)
            for i in range(len(K12)):
                il = mol.coll_ilow[collider][i]
                iu = mol.coll_iup[collider][i]
                expected_K[iu, il] += K12[i]
                expected_K[il, il] += -K12[i]
                expected_K[il, iu] += K21[i]
                expected_K[iu, iu] += -K21[i]
            assert np.all(K_contribution == expected_K)


def test_K_interpolation():
    for mol in emitting_molecules.values():
        for collider, coll_transitions in mol.coll_transitions.items():
            Tkin_data = mol.Tkin_data[collider]
            # include edge cases:
            test_Tkin = list(np.linspace(Tkin_data[0], Tkin_data[-1], 9))
            test_Tkin.append(Tkin_data[-1])
            for Tkin in test_Tkin:
                Ks = mol.interpolate_K(Tkin=Tkin, collider=collider)
                expected_K21 = np.empty(len(coll_transitions))
                expected_K12 = expected_K21.copy()
                for i, coll_trans in enumerate(coll_transitions):
                    expected_K21[i] = np.interp(
                        Tkin, coll_trans.Tkin_data, coll_trans.K21_data
                    )
                    expected_K12[i] = atomic_transition.compute_K12(
                        K21=expected_K21[i],
                        g_up=coll_trans.up.g,
                        g_low=coll_trans.low.g,
                        Delta_E=coll_trans.Delta_E,
                        Tkin=Tkin,
                    )
                rtol = 1e-15
                atol = 0
                assert np.allclose(Ks["K21"], expected_K21, atol=atol, rtol=rtol)
                assert np.allclose(Ks["K12"], expected_K12, atol=atol, rtol=rtol)


def test_K_interpolation_invalid_Tkin():
    for mol in emitting_molecules.values():
        for collider, coll_transitions in mol.coll_transitions.items():
            Tkin_data = mol.Tkin_data[collider]
            test_Tkin = [Tkin_data[0] - 1, Tkin_data[-1] + 1]
            for Tkin in test_Tkin:
                with pytest.raises(AssertionError):
                    mol.interpolate_K(Tkin=Tkin, collider=collider)


def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def test_GammaC():
    # could not find an elegant test...
    for mol in emitting_molecules.values():
        Tkin = np.mean([np.mean(Tkin_data) for Tkin_data in mol.Tkin_data.values()])
        # check all possible combinations of colliders:
        for test_collider_sets in powerset(mol.coll_transitions.keys()):
            collider_densities = {
                coll: i**2 for i, coll in enumerate(test_collider_sets)
            }
            expected_GammaC = np.zeros((mol.n_levels,) * 2)
            for collider, coll_dens in collider_densities.items():
                interpK = mol.interpolate_K(Tkin=Tkin, collider=collider)
                K = mol.construct_K_matrix(
                    n_levels=mol.n_levels,
                    K12=interpK["K12"],
                    K21=interpK["K21"],
                    ilow=mol.coll_ilow[collider],
                    iup=mol.coll_iup[collider],
                )
                expected_GammaC += K * coll_dens
            GammaC = mol.get_GammaC(Tkin=Tkin, collider_densities=collider_densities)
            assert np.all(expected_GammaC == GammaC)


def get_molecule(line_profile_type, width_v, datafilename):
    return molecule.EmittingMolecule(
        datafilepath=os.path.join(here, "LAMDA_files", datafilename),
        line_profile_type=line_profile_type,
        width_v=width_v,
    )


class TestOverlappingLines:

    line_profile_types = ("rectangular", "Gaussian")

    def get_HCl_molecule(self, line_profile_type, width_v):
        return get_molecule(
            line_profile_type=line_profile_type, width_v=width_v, datafilename="hcl.dat"
        )

    def get_CO_molecule(self, line_profile_type, width_v):
        return get_molecule(
            line_profile_type=line_profile_type, width_v=width_v, datafilename="co.dat"
        )

    def test_line_has_overlap(self):
        HCl_mol = self.get_HCl_molecule(
            line_profile_type="rectangular", width_v=50 * constants.kilo
        )
        assert np.any(HCl_mol.line_has_overlap)
        for i in range(HCl_mol.n_rad_transitions):
            n_overlap = len(HCl_mol.overlapping_lines[i])
            if HCl_mol.line_has_overlap[i]:
                assert n_overlap > 0
            else:
                assert n_overlap == 0

    def test_overlapping_lines(self):
        # first three transitions of HCl are separated by ~8 km/s and 6 km/s respectively
        overlapping_3lines = [
            self.get_HCl_molecule(
                line_profile_type="rectangular", width_v=16 * constants.kilo
            ),
            self.get_HCl_molecule(
                line_profile_type="Gaussian", width_v=10 * constants.kilo
            ),
        ]
        for ol in overlapping_3lines:
            assert ol.overlapping_lines[0] == [1, 2]
            assert ol.overlapping_lines[1] == [0, 2]
            assert ol.overlapping_lines[2] == [0, 1]
            assert np.all(ol.line_has_overlap[0:3])
        overlapping_2lines = [
            self.get_HCl_molecule(
                line_profile_type="rectangular", width_v=8.5 * constants.kilo
            ),
            self.get_HCl_molecule(
                line_profile_type="Gaussian", width_v=3.5 * constants.kilo
            ),
        ]
        for ol in overlapping_2lines:
            assert ol.overlapping_lines[0] == [
                1,
            ]
            assert ol.overlapping_lines[1] == [0, 2]
            assert ol.overlapping_lines[2] == [
                1,
            ]
            assert np.all(ol.line_has_overlap[0:3])
        # transitions 4-11 are separated by ~11.2 km/s
        overlapping_8lines = [
            self.get_HCl_molecule(
                line_profile_type="rectangular", width_v=11.5 * constants.kilo
            ),
            self.get_HCl_molecule(
                line_profile_type="Gaussian", width_v=4 * constants.kilo
            ),
        ]
        for ol in overlapping_8lines:
            for i in range(3, 11):
                assert ol.overlapping_lines[i] == [
                    index for index in range(3, 11) if index != i
                ]
            assert np.all(ol.line_has_overlap[3:11])
        for line_profile_type in self.line_profile_types:
            CO_molecule = self.get_CO_molecule(
                line_profile_type=line_profile_type, width_v=1 * constants.kilo
            )
            for overlap_lines in CO_molecule.overlapping_lines:
                assert overlap_lines == []
            assert not np.any(CO_molecule.line_has_overlap)
            HCl_molecule = self.get_HCl_molecule(
                line_profile_type=line_profile_type, width_v=0.01 * constants.kilo
            )
            assert HCl_molecule.overlapping_lines[0] == []
            assert HCl_molecule.overlapping_lines[11] == []
            assert not np.any(HCl_molecule.line_has_overlap[[0, 11]])

    def test_any_overlapping(self):
        for line_profile_type in self.line_profile_types:
            HCl_molecule = self.get_HCl_molecule(
                line_profile_type=line_profile_type, width_v=10 * constants.kilo
            )
            assert HCl_molecule.any_line_has_overlap(line_indices=[0, 1, 2, 3, 4])
            assert HCl_molecule.any_line_has_overlap(
                line_indices=[
                    0,
                ]
            )
            assert HCl_molecule.any_line_has_overlap(
                line_indices=list(range(len(HCl_molecule.rad_transitions)))
            )
            HCl_molecule = self.get_HCl_molecule(
                line_profile_type=line_profile_type, width_v=1 * constants.kilo
            )
            assert not HCl_molecule.any_line_has_overlap(line_indices=[0, 1, 2])
            CO_molecule = self.get_CO_molecule(
                line_profile_type=line_profile_type, width_v=1 * constants.kilo
            )
            assert not CO_molecule.any_line_has_overlap(
                line_indices=list(range(len(CO_molecule.rad_transitions)))
            )


class TestTotalQuantities:

    CO_molecule = get_molecule(
        line_profile_type="Gaussian", width_v=1 * constants.kilo, datafilename="co.dat"
    )
    HCl_molecule = get_molecule(
        line_profile_type="Gaussian",
        width_v=10 * constants.kilo,
        datafilename="hcl.dat",
    )
    N_CO = 1e15 / constants.centi**2
    N_HCl = 1e14 / constants.centi**2
    test_tau_dust = 1
    test_S = helpers.B_nu(nu=230 * constants.giga, T=100)

    @staticmethod
    def tau_dust_zero(nu):
        return np.zeros_like(nu)

    def tau_dust_nonzero(self, nu):
        return np.ones_like(nu) * self.test_tau_dust

    def tau_dust_iterator(self):
        for tau_d, tau_dust in zip(
            (0, self.test_tau_dust), (self.tau_dust_zero, self.tau_dust_nonzero)
        ):
            yield tau_d, tau_dust

    def test_no_overlap(self):
        line_index = 3
        line = self.CO_molecule.rad_transitions[line_index]
        width_nu = line.line_profile.width_nu
        nu = np.linspace(line.nu0 - 3 * width_nu, line.nu0 + 3 * width_nu, 200)
        level_population = self.CO_molecule.Boltzmann_level_population(T=23)
        x1 = level_population[line.low.index]
        x2 = level_population[line.up.index]
        tau_line = line.tau_nu(nu=nu, N1=x1 * self.N_CO, N2=x2 * self.N_CO)
        for tau_d, tau_dust in self.tau_dust_iterator():
            tau_tot = self.CO_molecule.get_tau_tot_nu(
                line_index=line_index,
                level_population=level_population,
                N=self.N_CO,
                tau_dust=tau_dust,
            )(nu)
            x1 = level_population[line.low.index]
            x2 = level_population[line.up.index]
            expected_tau_tot = tau_line + tau_d
            assert np.all(tau_tot == expected_tau_tot)

    def test_with_overlap(self):
        line_index = 1
        line = self.HCl_molecule.rad_transitions[line_index]
        assert self.HCl_molecule.overlapping_lines[line_index] == [0, 2]
        width_nu = line.line_profile.width_nu
        nu_start = self.HCl_molecule.rad_transitions[0].nu0 - 3 * width_nu
        nu_end = self.HCl_molecule.rad_transitions[2].nu0 + 3 * width_nu
        nu = np.linspace(nu_start, nu_end, 500)
        level_population = self.CO_molecule.Boltzmann_level_population(T=23)

        for tau_d, tau_dust in self.tau_dust_iterator():
            tau_tot = self.HCl_molecule.get_tau_tot_nu(
                line_index=line_index,
                level_population=level_population,
                N=self.N_HCl,
                tau_dust=tau_dust,
            )(nu)
            expected_tau_tot = np.zeros_like(nu)
            for i in (0, 1, 2):
                line_i = self.HCl_molecule.rad_transitions[i]
                x1 = level_population[line_i.low.index]
                x2 = level_population[line_i.up.index]
                expected_tau_tot += line_i.tau_nu(
                    N1=self.N_HCl * x1, N2=self.N_HCl * x2, nu=nu
                )
            expected_tau_tot += tau_d
            assert np.all(tau_tot == expected_tau_tot)
