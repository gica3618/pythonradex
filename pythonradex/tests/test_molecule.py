# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:35:53 2017

@author: gianni
"""
import os
from pythonradex import molecule,atomic_transition
import numpy as np
from scipy import constants
import itertools

here = os.path.dirname(os.path.abspath(__file__))
lamda_filepath = os.path.join(here,'co.dat')

test_molecule = molecule.Molecule.from_LAMDA_datafile(data_filepath=lamda_filepath)
line_profile_cls=atomic_transition.SquareLineProfile
width_v=1*constants.kilo

emitting_molecule_std = molecule.EmittingMolecule(
                         levels=test_molecule.levels,
                         rad_transitions=test_molecule.rad_transitions,
                         coll_transitions=test_molecule.coll_transitions,
                         line_profile_cls=line_profile_cls,width_v=width_v)

emitting_molecule_lamda = molecule.EmittingMolecule.from_LAMDA_datafile(
                           data_filepath=lamda_filepath,
                           line_profile_cls=line_profile_cls,width_v=width_v)

Tex = 101
LTE_level_pop = emitting_molecule_lamda.LTE_level_pop(Tex)

def test_LTE_level_pop_normalisation():
    assert np.isclose(np.sum(LTE_level_pop),1)

def test_LTE_level_pop():
    all_transitions = list(emitting_molecule_lamda.coll_transitions.values())
    all_transitions.append(emitting_molecule_lamda.rad_transitions)
    for trans in itertools.chain.from_iterable(all_transitions):
        trans_Tex = trans.Tex(x1=LTE_level_pop[trans.low.number],
                              x2=LTE_level_pop[trans.up.number])
        assert np.isclose(Tex,trans_Tex,rtol=1e-3)

def test_get_transition_number():
    rad_trans_number = test_molecule.get_transition_number('11-10')
    assert rad_trans_number == 10

def test_emitting_molecule_constructor():
    assert isinstance(emitting_molecule_std.rad_transitions[0],
                      atomic_transition.EmissionLine)
    assert emitting_molecule_std.rad_transitions[-1].up.number\
            == test_molecule.rad_transitions[-1].up.number
    assert emitting_molecule_std.rad_transitions[-1].low.number\
            == test_molecule.rad_transitions[-1].low.number

def test_emitting_molecule_constructor_from_LAMDA():
    assert emitting_molecule_lamda.n_rad_transitions == 40
    assert len(emitting_molecule_lamda.coll_transitions) == 2
    assert len(emitting_molecule_lamda.levels) == 41
    test_level = emitting_molecule_lamda.levels[1]
    assert test_level.g == 3
    assert test_level.number == 1
    assert test_level.E == constants.h*constants.c*3.845033413/constants.centi
    test_rad_trans = emitting_molecule_lamda.rad_transitions[2]
    assert test_rad_trans.A21 == 2.497e-06
    assert test_rad_trans.up.number == 3
    assert test_rad_trans.low.number == 2
    assert np.isclose(test_rad_trans.nu0,345.7959899*constants.giga)
    assert np.isclose(test_rad_trans.Delta_E,test_rad_trans.nu0*constants.h,atol=0)
    assert len(emitting_molecule_lamda.coll_transitions['para-H2']) == 820
    test_coll_trans_pH2 = emitting_molecule_lamda.coll_transitions['para-H2'][3]
    assert test_coll_trans_pH2.up.number == 3
    assert test_coll_trans_pH2.low.number == 0
    assert np.isclose(test_coll_trans_pH2.coeffs(5)['K21'],4.820E-12*constants.centi**3,
                      atol=0)
    test_coll_trans_oH2 = emitting_molecule_lamda.coll_transitions['ortho-H2'][2]
    assert test_coll_trans_oH2.up.number == 2
    assert test_coll_trans_oH2.low.number == 1
    assert np.isclose(test_coll_trans_oH2.coeffs(10)['K21'],6.276E-11*constants.centi**3,
                      atol=0)
    
def test_Tex():
    assert np.allclose(Tex,emitting_molecule_lamda.get_Tex(LTE_level_pop),atol=0)

def test_get_tau_nu0():
    tau_nu0 = emitting_molecule_lamda.get_tau_nu0(
                          N=1e10,level_population=LTE_level_pop)
    assert len(tau_nu0) == emitting_molecule_lamda.n_rad_transitions
    
