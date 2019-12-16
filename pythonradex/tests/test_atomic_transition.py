# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:07:35 2017

@author: gianni
"""

from pythonradex import atomic_transition
from scipy import constants
import pytest
import numpy as np


class TestLevel():

    g = 2
    E = 3
    level = atomic_transition.Level(g=g,E=E,number=1)
    
    def test_LTE_level_pop(self):
        T = 50
        Z = 3
        lte_level_pop = self.level.LTE_level_pop(Z=Z,T=T)
        assert lte_level_pop == self.g*np.exp(-self.E/(constants.k*T))/Z
        shape = (5,5)
        T_array = np.ones(shape)*T
        Z_array = np.ones(shape)*Z
        lte_level_pop_array = self.level.LTE_level_pop(Z=Z_array,T=T_array)
        assert lte_level_pop_array.shape == shape
        assert np.all(lte_level_pop==lte_level_pop_array)


class TestLineProfile():
    nu0 = 400*constants.giga
    width_v = 10*constants.kilo
    gauss_line_profile = atomic_transition.GaussianLineProfile(nu0=nu0,width_v=width_v)
    square_line_profile = atomic_transition.SquareLineProfile(nu0=nu0,width_v=width_v)
    profiles = (gauss_line_profile,square_line_profile)
    test_v = np.linspace(-3*width_v,3*width_v,600)

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
        nu_array = self.square_line_profile.nu_array
        random_values = np.random.rand(nu_array.size)
        profile_window = np.where(self.square_line_profile.phi_nu(nu_array)==0,0,1)
        expected_average = np.sum(profile_window*random_values)/np.count_nonzero(profile_window)
        average = self.square_line_profile.average_over_nu_array(random_values)
        assert np.isclose(expected_average,average,rtol=5e-2,atol=0)
    
    def test_normalisation(self):
        for profile in self.profiles:
            integrated_line_profile = np.trapz(profile.phi_nu_array,profile.nu_array)
            integrated_line_profile_v = np.trapz(profile.phi_v(self.test_v),self.test_v)
            for intg_prof in (integrated_line_profile,integrated_line_profile_v):
                 assert np.isclose(intg_prof,1,rtol=1e-2,atol=0)

    def test_profile_shape(self):
        square_phi_nu = self.square_line_profile.phi_nu_array
        square_phi_v = self.square_line_profile.phi_v(self.test_v)
        for square_phi,x_axis,width in zip((square_phi_nu,square_phi_v),
                                     (self.square_line_profile.nu_array,self.test_v),
                                     (self.square_line_profile.width_nu,self.width_v)):
            assert square_phi[0] ==  square_phi[-1] == 0
            assert square_phi[square_phi.size//2] > 0
            square_indices = np.where(square_phi>0)[0]
            square_window_size = x_axis[square_indices[-1]] - x_axis[square_indices[0]]
            assert np.isclose(square_window_size,width,rtol=5e-2,atol=0)
        gauss_phi_nu = self.gauss_line_profile.phi_nu_array
        gauss_phi_v = self.gauss_line_profile.phi_v(self.test_v)
        for gauss_phi,x_axis,width in zip((gauss_phi_nu,gauss_phi_v),
                                          (self.square_line_profile.nu_array,self.test_v),
                                          (self.square_line_profile.width_nu,self.width_v)):
            assert np.all(np.array((gauss_phi[0],gauss_phi[-1]))
                          <gauss_phi[gauss_phi.size//2])
            max_index = np.argmax(gauss_phi)
            half_max_index = np.argmin(np.abs(gauss_phi-np.max(gauss_phi)/2))
            assert np.isclose(2*np.abs(x_axis[max_index]-x_axis[half_max_index]),
                              width,rtol=3e-2,atol=0)


class TestTransition():

    up = atomic_transition.Level(g=1,E=1,number=1)
    low = atomic_transition.Level(g=1,E=0,number=0)
    line_profile_cls = atomic_transition.SquareLineProfile
    A21 = 1
    radiative_transition = atomic_transition.RadiativeTransition(
                             up=up,low=low,A21=A21)
    width_v = 1*constants.kilo
    Tkin_data=np.array((1,2,3,4,5))
    test_emission_line = atomic_transition.EmissionLine(
                              up=up,low=low,A21=A21,
                              line_profile_cls=line_profile_cls,
                              width_v=width_v)

    def test_radiative_transition_negative_DeltaE(self):
        with pytest.raises(AssertionError):
            atomic_transition.RadiativeTransition(up=self.low,low=self.up,A21=1)

    def test_emission_line_constructor(self):
        assert self.test_emission_line.nu0 == self.test_emission_line.line_profile.nu0

    def test_constructor_from_radiative_transition(self):
        emission_line = atomic_transition.EmissionLine.from_radiative_transition(
                               radiative_transition=self.radiative_transition,
                               line_profile_cls=self.line_profile_cls,
                               width_v=self.width_v)
        assert emission_line.nu0 == self.radiative_transition.nu0
        assert emission_line.B12 == self.radiative_transition.B12

    def test_coll_coeffs(self):
        K21_data_sets = [np.array((2,1,4,6,3)),np.array((1,0,0,6,3))]
        for K21_data in K21_data_sets:
            coll_transition = atomic_transition.CollisionalTransition(
                                up=self.up,low=self.low,K21_data=K21_data,
                                Tkin_data=self.Tkin_data)
            Tkin_interp = np.array((self.Tkin_data[0],self.Tkin_data[-1]))
            coeff = coll_transition.coeffs(Tkin_interp)['K21']
            expected_coeff = np.array((K21_data[0],K21_data[-1]))
            assert np.allclose(coeff,expected_coeff,atol=0)
            intermediate_temp = np.mean(self.Tkin_data[:2])
            intermediate_coeff = coll_transition.coeffs(intermediate_temp)['K21']
            boundaries = np.sort(K21_data[:2])
            assert boundaries[0] <= intermediate_coeff <= boundaries[1]

    @pytest.mark.filterwarnings('ignore:invalid value','ignore:divide by zero')
    def test_Tex(self):
        assert self.radiative_transition.Tex(x1=0,x2=0) == 0
        assert self.radiative_transition.Tex(x1=1,x2=0) == 0
        assert self.radiative_transition.Tex(x1=0.5,x2=0) == 0

    def tau_nu(self):
        nu = np.ones((2,4,7))*50
        N1 = 1
        N2 = 3
        tau_nu = self.test_emission_line.tau_nu(N1=N1,N2=N2,nu=nu)
        assert tau_nu.shape == nu.shape
        tau_nu_array = self.test_emission_line.tau_nu_array(N1=N1,N2=N2)
        tau_nu_array_explicit = self.test_emission_line.tau_nu(
                                   N1=N1,N2=N2,
                                   nu=self.test_emission_line.line_profile.nu_array)
        assert np.all(tau_nu_array==tau_nu_array_explicit)