# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:35:53 2017

@author: gianni
"""
import os
from pythonradex import molecule,atomic_transition,LAMDA_file
import numpy as np
from scipy import constants

here = os.path.dirname(os.path.abspath(__file__))
lamda_filepath = os.path.join(here,'LAMDA_files/co.dat')

test_data = LAMDA_file.read(datafilepath=lamda_filepath,read_frequencies=False)
line_profile_type = 'rectangular'
width_v=1*constants.kilo

# molecule_std = molecule.Molecule(
#                     levels=test_data['levels'],
#                     rad_transitions=test_data['radiative transitions'],
#                     coll_transitions=test_data['collisional transitions'])
molecule_lamda = molecule.Molecule(
                                datafilepath=lamda_filepath,read_frequencies=False)
molecule_lamda_frequencies = molecule.Molecule(
                                datafilepath=lamda_filepath,read_frequencies=True)
# emitting_molecule_std = molecule.EmittingMolecule(
#                          levels=test_data['levels'],
#                          rad_transitions=test_data['radiative transitions'],
#                          coll_transitions=test_data['collisional transitions'],
#                          line_profile_type=line_profile_type,width_v=width_v)
emitting_molecule_lamda = molecule.EmittingMolecule(
                           datafilepath=lamda_filepath,
                           line_profile_type=line_profile_type,width_v=width_v,
                           read_frequencies=False)
emitting_molecule_lamda_frequencies = molecule.EmittingMolecule(
                                        datafilepath=lamda_filepath,
                                        line_profile_type=line_profile_type,
                                        width_v=width_v,read_frequencies=True)

#reference_mol = molecule_std
test_molecules_no_freq = [molecule_lamda,emitting_molecule_lamda]
test_molecules_freq = [molecule_lamda_frequencies,emitting_molecule_lamda_frequencies]
test_molecules = test_molecules_no_freq+test_molecules_freq
emitting_molecules = [emitting_molecule_lamda,
                      emitting_molecule_lamda_frequencies]

def test_Molecule_levels():
    for i,level in enumerate(test_data['levels']):
        for mol in test_molecules:
            for attribute in ('g','E','number'):
                assert getattr(level,attribute)\
                                  == getattr(mol.levels[i],attribute)

def test_Molecule_rad_transitions():
    for i,rad_trans in enumerate(test_data['radiative transitions']):
        for mol in test_molecules:
            for attribute in ('Delta_E','A21'):
                assert getattr(rad_trans,attribute)\
                                  == getattr(mol.rad_transitions[i],attribute)
        #following attributes depend on nu0, so change if nu0 is taken from the LAMDA file
        for attribute in ('B12','nu0','B21'):
            for mol in test_molecules_no_freq:
                assert getattr(rad_trans,attribute)\
                             == getattr(mol.rad_transitions[i],attribute)
            for mol in test_molecules_freq:
                assert np.isclose(a=getattr(rad_trans,attribute),
                                  b=getattr(mol.rad_transitions[i],attribute),
                                  atol=0,rtol=1e-4)

def test_Molecule_coll_transitions():
    colliders = sorted(test_data['collisional transitions'].keys())
    for mol in test_molecules:
        assert colliders == sorted(mol.coll_transitions.keys())
    for collider,coll_transitions in test_data['collisional transitions'].items():
        for i,coll_trans in enumerate(coll_transitions):
            for mol in test_molecules:
                for attribute in ('K21_data','Tkin_data'):
                    assert np.all(getattr(coll_trans,attribute)\
                                      == getattr(mol.coll_transitions[collider][i],
                                                 attribute))

def test_setting_partition_function():
    test_mol = molecule.Molecule(
                        datafilepath=lamda_filepath,
                        partition_function=lambda x: -10)
    assert test_mol.Z(100) == -10

def test_partition_func():
    T = 50
    for mol in test_molecules:
        Q = 0
        for level in mol.levels:
            Q += level.g*np.exp(-level.E/(constants.k*T))
        assert np.isclose(Q,mol.Z(T),atol=0,rtol=1e-10)

def test_LTE_level_pop():
    T = 30
    for mol in test_molecules:
        LTE_level_pop = mol.LTE_level_pop(T=T)
        assert np.isclose(np.sum(LTE_level_pop),1)
        Q = mol.Z(T=T)
        for i,level in enumerate(mol.levels):
            expected_pop = level.g*np.exp(-level.E/(constants.k*T)) / Q
            assert expected_pop == LTE_level_pop[i]

def test_get_transition_number():
    for mol in test_molecules:
        rad_trans_number = mol.get_rad_transition_number('11-10')
        assert rad_trans_number == 10

rng = np.random.default_rng(seed=0)

def test_tau():
    N_values = np.array((1e12,1e14,1e16,1e18))/constants.centi**2
    for N in N_values:
        for mol in emitting_molecules:
            level_population = rng.random(mol.n_levels)
            level_population /= np.sum(level_population)
            tau_nu0 = mol.get_tau_nu0(N=N,level_population=level_population)
            expected_tau_nu0 = []
            for i,rad_trans in enumerate(mol.rad_transitions):
                n_up = rad_trans.up.number
                N2 = N*level_population[n_up]
                n_low = rad_trans.low.number
                N1 = N*level_population[n_low]
                t = atomic_transition.fast_tau_nu(
                         A21=rad_trans.A21,phi_nu=rad_trans.line_profile.phi_nu(rad_trans.nu0),
                         g_low=rad_trans.low.g,g_up=rad_trans.up.g,N1=N1,N2=N2,nu=rad_trans.nu0)
                expected_tau_nu0.append(t)
            assert np.all(expected_tau_nu0==tau_nu0)

def test_tau_LTE():
    N_values = np.array((1e12,1e14,1e16,1e18))/constants.centi**2
    T = 123
    for N in N_values:
        for mol in emitting_molecules:
            level_population = mol.LTE_level_pop(T=T)
            tau_nu0_LTE = mol.get_tau_nu0_LTE(N=N,T=T)
            expected_tau_nu0 = []
            for i,rad_trans in enumerate(mol.rad_transitions):
                n_up = rad_trans.up.number
                N2 = N*level_population[n_up]
                n_low = rad_trans.low.number
                N1 = N*level_population[n_low]
                t = atomic_transition.fast_tau_nu(
                         A21=rad_trans.A21,phi_nu=rad_trans.line_profile.phi_nu(rad_trans.nu0),
                         g_low=rad_trans.low.g,g_up=rad_trans.up.g,N1=N1,N2=N2,nu=rad_trans.nu0)
                expected_tau_nu0.append(t)
            assert np.all(expected_tau_nu0==tau_nu0_LTE)
    
def test_LTE_Tex():
    T = 123
    for mol in emitting_molecules:
        LTE_level_pop = mol.LTE_level_pop(T=T)
        Tex = mol.get_Tex(level_population=LTE_level_pop)
        assert np.allclose(a=T,b=Tex,atol=0,rtol=1e-10)

def test_Tex():
    for mol in emitting_molecules:
        level_population = rng.random(mol.n_levels)
        level_population /= np.sum(level_population)
        Tex = mol.get_Tex(level_population=level_population)
        expected_Tex = []
        for i,rad_trans in enumerate(mol.rad_transitions):
            n_up = rad_trans.up.number
            x2 = level_population[n_up]
            n_low = rad_trans.low.number
            x1 = level_population[n_low]
            tex = atomic_transition.fast_Tex(
                        Delta_E=rad_trans.Delta_E,g_low=rad_trans.low.g,
                        g_up=rad_trans.up.g,x1=x1,x2=x2)
            expected_Tex.append(tex)
        assert np.all(Tex==expected_Tex)