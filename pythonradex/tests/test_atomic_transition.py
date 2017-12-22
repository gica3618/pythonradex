# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:07:35 2017

@author: gianni
"""

from pythonradex import atomic_transition
from scipy import constants
import pytest
import numpy as np


class TestLineProfile():
    nu0 = 400*constants.giga
    width_v = 10*constants.kilo
    gauss_line_profile = atomic_transition.GaussianLineProfile(nu0=nu0,width_v=width_v)
    square_line_profile = atomic_transition.SquareLineProfile(nu0=nu0,width_v=width_v)
    profiles = (gauss_line_profile,square_line_profile)

    def test_abstract_line_profile(self):
        with pytest.raises(NotImplementedError):
            atomic_transition.LineProfile(nu0=self.nu0,width_v=self.width_v)
    
    def test_constant_average_over_nu(self):
        for profile in self.profiles:
            const_array = np.ones_like(profile.nu_array)
            const_average = profile.average_over_nu_array(const_array)
            assert np.isclose(const_average,1,rtol=1e-2,atol=0)
    
    def test_asymmetric_average_over_nu(self):
        for profile in self.profiles:
            left_value,right_value = 0,1
            asymmetric_array = np.ones_like(profile.nu_array)*left_value
            asymmetric_array[:asymmetric_array.size//2] = right_value
            asymmetric_average = profile.average_over_nu_array(asymmetric_array)
            assert np.isclose(asymmetric_average,np.mean((left_value,right_value)),
                              rtol=1e-2,atol=0)
    
    def test_square_profile_average_over_nu(self):
        np.random.seed(0)
        random_values = np.random.rand(self.square_line_profile.nu_array.size)
        profile_window = np.where(self.square_line_profile.phi_nu==0,0,1)
        expected_average = np.mean(profile_window*random_values)
        average = self.square_line_profile.average_over_nu_array(random_values)
        assert np.isclose(expected_average,average,rtol=5e-2,atol=0)
    
    def test_normalisation(self):
        for profile in self.profiles:
            integrated_line_profile = np.trapz(profile.phi_nu_array,profile.nu_array)
            assert np.isclose(integrated_line_profile,1,rtol=1e-2,atol=0)

    def test_profile_shape(self):
        square_phi_nu = self.square_line_profile.phi_nu_array
        assert square_phi_nu[0] ==  square_phi_nu[-1] == 0
        assert square_phi_nu[square_phi_nu.size//2] > 0
        square_indices = np.where(square_phi_nu>0)[0]
        square_window_size = self.square_line_profile.nu_array[square_indices[-1]]\
                             - self.square_line_profile.nu_array[square_indices[0]]
        assert np.isclose(square_window_size,self.square_line_profile.width_nu,
                          rtol=5e-2,atol=0)
        gauss_phi_nu = self.gauss_line_profile.phi_nu_array
        assert np.all(np.array((gauss_phi_nu[0],gauss_phi_nu[-1]))
                      <gauss_phi_nu[gauss_phi_nu.size//2])


class TestTransition():

    up = atomic_transition.Level(g=1,E=1,number=1)
    low = atomic_transition.Level(g=1,E=0,number=0)
    line_profile_cls = atomic_transition.SquareLineProfile
    A21 = 1
    radiative_transition = atomic_transition.RadiativeTransition(
                             up=up,low=low,A21=A21)
    width_v = 1*constants.kilo
    K21_data=np.array((2,1,4,6,3))
    Tkin_data=np.array((1,2,3,4,5))
    coll_transition = atomic_transition.CollisionalTransition(
                            up=up,low=low,K21_data=K21_data,Tkin_data=Tkin_data)

    def test_transition(self):
        with pytest.raises(AssertionError):
            atomic_transition.Transition(up=self.low,low=self.up)

    def test_emission_line_constructor(self):
        emission_line = atomic_transition.EmissionLine(
                              up=self.up,low=self.low,A21=self.A21,
                              line_profile_cls=self.line_profile_cls,
                              width_v=self.width_v)
        assert emission_line.nu0 == emission_line.line_profile.nu0

    def test_constructor_from_radiative_transition(self):
        emission_line = atomic_transition.EmissionLine.from_radiative_transition(
                               radiative_transition=self.radiative_transition,
                               line_profile_cls=self.line_profile_cls,
                               width_v=self.width_v)
        assert emission_line.nu0 == self.radiative_transition.nu0
        assert emission_line.B12 == self.radiative_transition.B12

    def test_coll_coeffs(self):
        Tkin_interp = np.array((self.Tkin_data[0],self.Tkin_data[-1]))
        coeff = self.coll_transition.coeffs(Tkin_interp)['K21']
        expected_coeff = np.array((self.K21_data[0],self.K21_data[-1]))
        assert np.allclose(coeff,expected_coeff,atol=0)
        intermediate_temp = np.mean(self.Tkin_data[:2])
        intermediate_coeff = self.coll_transition.coeffs(intermediate_temp)['K21']
        boundaries = np.sort(self.K21_data[:2])
        assert boundaries[0] <= intermediate_coeff <= boundaries[1]

    @pytest.mark.filterwarnings('ignore:invalid value','ignore:divide by zero')
    def test_Tex(self):
        assert self.radiative_transition.Tex(x1=0,x2=0) == 0
        assert self.radiative_transition.Tex(x1=1,x2=0) == 0
        assert self.radiative_transition.Tex(x1=0.5,x2=0) == 0
