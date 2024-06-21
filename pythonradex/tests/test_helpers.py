# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:42:40 2017

@author: gianni
"""

from pythonradex import helpers
import numpy as np
from scipy import constants


def test_relative_difference_arrays():
    x = np.array((0,4,2,10,0,2, -1,-1))
    y = np.array((0,0,4,10,2,2.1,1,-1))
    relative_difference = helpers.relative_difference(x,y)
    expected_relative_difference = np.array((0,1,1,0,1,0.05,2,0))
    assert np.allclose(relative_difference,expected_relative_difference,atol=0,
                       rtol=1e-10)

def test_zero_background():
    assert helpers.zero_background(10) == 0
    assert np.all(helpers.zero_background(np.random.rand(10)) == 0)

def test_CMB_background():
    test_nu = np.logspace(np.log10(1),np.log10(1000),20)*constants.giga
    test_z = (0,2)
    for z in test_z:
        CMB = helpers.generate_CMB_background(z=z)
        assert np.all(CMB(test_nu) == helpers.B_nu(nu=test_nu,T=2.73*(1+z)))