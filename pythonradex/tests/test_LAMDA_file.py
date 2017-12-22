# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:47:19 2017

@author: gianni
"""

from pythonradex import LAMDA_file
import os
from scipy import constants
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

filepath = os.path.join(here,'co.dat')
lamda_data = LAMDA_file.read(filepath)
levels = lamda_data['levels']
rad_transitions = lamda_data['radiative transitions']
coll_transitions = lamda_data['collisional transitions']

def test_levels():
    assert len(levels) == 41
    test_level_index = 10
    test_level = levels[test_level_index]
    assert test_level.g == 21
    assert np.isclose(test_level.E/(constants.h*constants.c),
                      211.404098498/constants.centi,atol=0)
    assert test_level.number == test_level_index

def test_rad_transitions():
    assert len(rad_transitions) == 40
    test_trans_index = 20
    test_trans = rad_transitions[test_trans_index]
    assert test_trans.up.number == 21
    assert test_trans.low.number == 20
    assert test_trans.A21 == 8.833e-4
    nu0_in_file = 2413.9171130*constants.giga
    assert np.isclose(test_trans.nu0,nu0_in_file,atol=0)
    assert np.isclose(test_trans.Delta_E,nu0_in_file*constants.h,atol=0)

def test_coll_transitions():
    assert len(coll_transitions) == 2
    assert 'para-H2' in coll_transitions
    assert 'ortho-H2' in coll_transitions
    test_coll_trans_index_H2 = 100
    test_coll_trans_pH2 = coll_transitions['para-H2'][test_coll_trans_index_H2]
    Tkin_data_pH2 = np.array((2,5,10,20,30,40,50,60,70,80,90,100,150,200,300,400,
                              500,600,700,750,800,900,1000,2000,3000))
    assert np.all(test_coll_trans_pH2.Tkin_data == Tkin_data_pH2)
    assert test_coll_trans_pH2.up.number == 14
    assert test_coll_trans_pH2.low.number == 9
    coeffs_pH2 = np.array((1.484e-11,1.507e-11,1.548e-11,1.579e-11,1.616e-11,
                           1.677e-11,1.751e-11,1.831e-11,1.910e-11,1.987e-11,
                           2.060e-11,2.129e-11,2.420e-11,2.643e-11,2.975e-11,
                           3.224e-11,3.429e-11,3.605e-11,3.758e-11,3.827e-11,
                           3.893e-11,4.013e-11,4.121e-11,4.756e-11,4.901e-11))\
                *constants.centi**3
    assert np.allclose(coeffs_pH2,test_coll_trans_pH2.coeffs(Tkin_data_pH2)['K21'],
                       atol=0)
    test_coll_trans_index_oH2 = 200
    test_coll_trans_oH2 = coll_transitions['ortho-H2'][test_coll_trans_index_oH2]
    Tkin_data_oH2 = Tkin_data_pH2
    assert np.all(test_coll_trans_oH2.Tkin_data == Tkin_data_oH2)
    assert test_coll_trans_oH2.up.number == 20
    assert test_coll_trans_oH2.low.number == 10
    coeffs_oH2 = np.array((4.949E-13,5.627E-13,6.206E-13,6.794E-13,7.309E-13,7.881E-13,
                         8.533E-13,9.263E-13,1.006E-12,1.093E-12,1.184E-12,1.280E-12,
                         1.807E-12,2.369E-12,3.471E-12,4.461E-12,5.324E-12,6.076E-12,
                         6.740E-12,7.046E-12,7.337E-12,7.883E-12,8.389E-12,1.229E-11,
                         1.465E-11)) * constants.centi**3
    assert np.allclose(coeffs_oH2,test_coll_trans_oH2.coeffs(Tkin_data_oH2)['K21'],
                       atol=0)
