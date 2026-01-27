#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:41:14 2025

@author: gianni
"""

from pythonradex import molecule, atomic_transition
from scipy import constants
import numpy as np


def construct_old_GammaC(mol, Tkin, collider_densities):
    GammaC = np.zeros((mol.n_levels,) * 2)
    elements = GammaC.copy()
    for collider, coll_dens in collider_densities.items():
        for coll_trans in mol.coll_transitions[collider]:
            K21 = np.interp(Tkin, coll_trans.Tkin_data, coll_trans.K21_data)

            # here is fundamental problem with the K cube:
            # 1) interpolation of K21, and then compute K12, or
            # 2) interpolation of both K21 and K12
            # with option 2), the interpolated values of K21 and K12 do not satisfy
            # anymore the equation relating them, so I think option 1 is preferable,
            # which makes the K cube approach difficult

            # using this will produce differences:
            K12 = atomic_transition.compute_K12(
                K21=K21,
                g_up=coll_trans.up.g,
                g_low=coll_trans.low.g,
                Delta_E=coll_trans.Delta_E,
                Tkin=Tkin,
            )

            # using this will remove differences:
            # K12_data = atomic_transition.compute_K12(
            #           K21=coll_trans.K21_data,g_up=coll_trans.up.g,
            #           g_low=coll_trans.low.g,
            #           Delta_E=coll_trans.Delta_E,Tkin=coll_trans.Tkin_data)
            # K12 = np.interp(Tkin,coll_trans.Tkin_data,K12_data)

            # K12 and K21 are 1D arrays because Tkin is a 1D array
            i_low = coll_trans.low.index
            i_up = coll_trans.up.index
            GammaC[i_up, i_low] += K12 * coll_dens
            GammaC[i_low, i_low] -= K12 * coll_dens
            GammaC[i_low, i_up] += K21 * coll_dens
            GammaC[i_up, i_up] -= K21 * coll_dens
            elements[i_up, i_low] += 1
            elements[i_low, i_low] += 1
            elements[i_low, i_up] += 1
            elements[i_up, i_up] += 1
    return GammaC, elements


def compute_K_cube(mol):
    K_cube = {
        collider: np.zeros((mol.n_levels, mol.n_levels, mol.Tkin_data[collider].size))
        for collider in mol.coll_transitions.keys()
    }
    for collider, coll_transitions in mol.coll_transitions.items():
        for i, Tkin in enumerate(mol.Tkin_data[collider]):
            for coll_trans in coll_transitions:
                K21 = coll_trans.K21_data[i]
                K12 = atomic_transition.compute_K12(
                    K21=K21,
                    g_up=coll_trans.up.g,
                    g_low=coll_trans.low.g,
                    Delta_E=coll_trans.Delta_E,
                    Tkin=Tkin,
                )
                i_low = coll_trans.low.index
                i_up = coll_trans.up.index
                # production of upper level from lower level:
                K_cube[collider][i_up, i_low, i] += K12
                # destruction of lower level by transitions to upper level:
                K_cube[collider][i_low, i_low, i] += -K12
                # production lower level from upper level:
                K_cube[collider][i_low, i_up, i] += K21
                # destruction of upper level by transition to lower level:
                K_cube[collider][i_up, i_up, i] += -K21
        assert np.all(np.isfinite(K_cube[collider]))
    return K_cube


def get_GammaC_with_interpolation(mol, Tkin, collider_densities):
    GammaC = np.zeros((mol.n_levels,) * 2)
    K_cube = compute_K_cube(mol=mol)
    for collider, coll_dens in collider_densities.items():
        Tlimits = mol.Tkin_data_limits[collider]
        assert Tlimits[0] <= Tkin <= Tlimits[1]
        Tkin_data = mol.Tkin_data[collider]
        j = np.searchsorted(Tkin_data, Tkin, side="left")
        if j == 0:
            GammaC += K_cube[collider][:, :, 0] * coll_dens
            continue
        i = j - 1
        x0 = Tkin_data[i]
        y0 = K_cube[collider][:, :, i]
        x1 = Tkin_data[j]
        y1 = K_cube[collider][:, :, j]
        x = Tkin
        # linear interpolation:
        interp_K = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)
        GammaC += coll_dens * interp_K
        return GammaC


# filename = 'co.dat'
# collider_densities = {'para-H2':1}
# filename = 'ocs@xpol.dat'
# collider_densities = {'H2':1}
filename = "c.dat"
collider_densities = {"e": 1}
Tkin = 15


mol = molecule.EmittingMolecule(
    datafilepath=f"../../tests/LAMDA_files/{filename}",
    line_profile_type="Gaussian",
    width_v=1 * constants.kilo,
)
old_GammaC, elements = construct_old_GammaC(
    mol=mol, Tkin=Tkin, collider_densities=collider_densities
)
GammaC_interp = get_GammaC_with_interpolation(
    mol=mol, Tkin=Tkin, collider_densities=collider_densities
)
diff = old_GammaC - GammaC_interp
print(diff.diagonal())
print(elements)
relative_diff = np.abs(diff / old_GammaC)
print(np.max(relative_diff))
print(np.unravel_index(np.argmax(relative_diff), old_GammaC.shape))
