# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:07:35 2017

@author: gianni
"""

from pythonradex import atomic_transition
from scipy import constants
import pytest
import numpy as np


def test_fast_Tex():
    kwargs = {'Delta_E':1,'g_low':1,'g_up':1}
    assert atomic_transition.fast_Tex(x1=np.zeros(1),x2=np.zeros(1),**kwargs)[0] == 0
    assert atomic_transition.fast_Tex(x1=np.zeros(1),x2=np.array((0.3,)),**kwargs)[0] == 0
    assert np.all(atomic_transition.fast_Tex(x1=np.array((0.2,0.1)),x2=np.zeros(2),**kwargs)[0]
                  == 0)
    x1,x2 = np.array((0.2,)),np.array((0.1,))
    test_Tex = atomic_transition.fast_Tex(x1=x1,x2=x2,**kwargs)[0]
    explicit_Tex = -kwargs['Delta_E']/constants.k\
                       * (np.log((x2*kwargs['g_low'])/(x1*kwargs['g_up'])))**-1
    assert np.isclose(test_Tex,explicit_Tex[0],rtol=1e-10,atol=0)
    #test that floats also work as arguments:
    atomic_transition.fast_Tex(**kwargs,x1=0.2,x2=0.2)

def test_fast_tau():
    A21 = 1e-4
    phi_nu = 0.2
    g_low = 2
    g_up = 4
    N1 = 1e14
    N2 = 1e13
    nu = 200*constants.giga
    test_tau = atomic_transition.fast_tau_nu(A21=A21,phi_nu=phi_nu,g_low=g_low,
                                             g_up=g_up,N1=N1,N2=N2,nu=nu)
    explicit_tau = constants.c**2/(8*np.pi*nu**2)*A21*phi_nu*(g_up/g_low*N1-N2)
    assert test_tau == explicit_tau
    #check that numpy arrays also work:
    nu0 = 100*constants.giga
    width_v = 4*constants.kilo
    width_nu = width_v/constants.c*nu0
    nu_array = np.linspace(nu0-width_nu,nu0+width_nu,100)
    sigma_nu = width_nu/np.sqrt(8*np.log(2))
    phi_nu_array = 1/(np.sqrt(2*np.pi)*sigma_nu)*np.exp(-(nu_array-nu0)**2/(2*sigma_nu**2))
    atomic_transition.fast_tau_nu(A21=A21,phi_nu=phi_nu_array,g_low=g_low,
                                             g_up=g_up,N1=N1,N2=N2,nu=nu_array)

# def test_fast_coll_coeffs():
#     Tkin_data = np.array((20,40,100,200,300))
#     K21_data = np.array((1e-3,1e-4,2e-3,2e-4,5e-5))
#     K21_data_with_0 = K21_data.copy()
#     K21_data_with_0[1] = 0
#     gup = 2
#     glow = 3
#     Delta_E = 1e-10
#     for invalid_Tkin in (10,500):
#         with pytest.raises(AssertionError):
#             atomic_transition.fast_coll_coeffs(
#                      Tkin=np.array((invalid_Tkin,)),Tkin_data=Tkin_data,
#                      K21_data=K21_data,gup=gup,glow=glow,Delta_E=Delta_E)
#     log_Tkin_data = np.log(Tkin_data)
#     log_K21_data = np.log(K21_data)
#     test_Tkin = np.array((20,90,150,220.1))
#     log_interp_K21 = np.interp(np.log(test_Tkin),log_Tkin_data,log_K21_data)
#     log_interp_K21 = np.exp(log_interp_K21)
#     def get_K12(K21):
#         return K21*gup/glow*np.exp(-Delta_E/(constants.k*test_Tkin))
#     log_interp_K12 = get_K12(log_interp_K21)
#     K12_to_test,K21_to_test = atomic_transition.fast_coll_coeffs(
#                                  Tkin=test_Tkin,Tkin_data=Tkin_data,K21_data=K21_data,
#                                  gup=gup,glow=glow,Delta_E=Delta_E)
#     assert np.all(K12_to_test==log_interp_K12)
#     assert np.all(K21_to_test==log_interp_K21)
#     interp_K21 = np.interp(np.log(test_Tkin),log_Tkin_data,K21_data_with_0)
#     interp_K12 = get_K12(interp_K21)
#     K12_to_test_0,K21_to_test_0 = atomic_transition.fast_coll_coeffs(
#                                  Tkin=test_Tkin,Tkin_data=Tkin_data,K21_data=K21_data_with_0,
#                                  gup=gup,glow=glow,Delta_E=Delta_E)
#     assert np.all(K12_to_test_0==interp_K12)
#     assert np.all(K21_to_test_0==interp_K21)


class TestLineProfile():
    nu0 = 400*constants.giga
    width_v = 10*constants.kilo
    gauss_line_profile = atomic_transition.GaussianLineProfile(nu0=nu0,width_v=width_v)
    rect_line_profile = atomic_transition.RectangularLineProfile(nu0=nu0,width_v=width_v)
    profiles = {'gauss':gauss_line_profile,'rect':rect_line_profile}
    test_v = np.linspace(-3*width_v,3*width_v,600)
    width_nu = width_v/constants.c*nu0
    test_nu = np.linspace(nu0-3*width_nu,nu0+3*width_nu,600)

    def test_abstract_line_profile(self):
        with pytest.raises(NotImplementedError):
            atomic_transition.LineProfile(nu0=self.nu0,width_v=self.width_v)
    
    def test_normalisation(self):
        for profile in self.profiles.values():
            integrated_line_profile = np.trapezoid(profile.phi_nu(self.test_nu),self.test_nu)
            integrated_line_profile_v = np.trapezoid(profile.phi_v(self.test_v),self.test_v)
            for intg_prof in (integrated_line_profile,integrated_line_profile_v):
                 assert np.isclose(intg_prof,1,rtol=3e-2,atol=0)

    def test_profile_shape_rect(self):
        phi_nu = self.rect_line_profile.phi_nu(self.test_nu)
        phi_v = self.rect_line_profile.phi_v(self.test_v)
        for phi,x_axis,width in zip((phi_nu,phi_v),
                                     (self.test_nu,self.test_v),
                                     (self.rect_line_profile.width_nu,self.width_v)):
            assert np.all(phi>=0)
            larger_0 = phi > 0
            larger_0_indices = np.where(larger_0)[0]
            assert larger_0_indices[0] > 0
            assert larger_0_indices[-1] < len(phi)-1
            assert np.all(np.diff(larger_0_indices)==1)
            window_size = x_axis[larger_0_indices[-1]] - x_axis[larger_0_indices[0]]
            assert np.isclose(window_size,width,rtol=5e-2,atol=0)

    def test_profile_shape_gauss(self):
        phi_nu = self.gauss_line_profile.phi_nu(self.test_nu)
        phi_v = self.gauss_line_profile.phi_v(self.test_v)
        for phi,x_axis,width in zip((phi_nu,phi_v),
                                    (self.test_nu,self.test_v),
                                    (self.gauss_line_profile.width_nu,self.width_v)):
            max_index = np.argmax(phi)
            assert np.all(np.diff(phi[:max_index+1]) > 0)
            #because nu array is created with linspace, it can be symmetric
            #around the peak, so need to take care:
            assert phi[max_index+1]-phi[max_index] <= 0
            assert np.all(np.diff(phi[max_index+1:]) < 0)
            half_max_index = np.argmin(np.abs(phi-np.max(phi)/2))
            assert np.isclose(2*np.abs(x_axis[max_index]-x_axis[half_max_index]),
                              width,rtol=4e-2,atol=0)

    def test_phi_nu0(self):
        for profile in self.profiles.values():
            assert profile.phi_nu0 == profile.phi_nu(nu=profile.nu0)

    def test_phi_nu_averaging(self):
        def unity(nu):
            return 1
        for profile in self.profiles.values():
            avg = profile.average_over_phi_nu(func=unity)
            assert np.isclose(avg,1,rtol=1,atol=1e-2)

        def test_func(func,grid_width,n_grid_elements,rtol):
            fine_nu_grid = np.linspace(self.nu0-grid_width*self.width_nu,
                                       self.nu0+grid_width*self.width_nu,n_grid_elements)
            for profile in self.profiles.values():
                expected_average = np.trapezoid(func(fine_nu_grid)*profile.phi_nu(fine_nu_grid),
                                            fine_nu_grid)
                expected_average /= np.trapezoid(profile.phi_nu(fine_nu_grid),fine_nu_grid)
                average = profile.average_over_phi_nu(func)
                assert np.isclose(a=expected_average,b=average,atol=0,rtol=rtol)
        
        def quadratic(nu):
            return np.where(np.abs(nu-self.nu0)<2*self.width_nu,nu**2,0)
        def Gaussian(nu):
            return np.exp(-(nu-self.nu0)**2/(2*self.width_nu**2))
        def nasty(nu):
            return np.where((np.abs(nu-self.nu0)<6*self.width_nu)
                            & (np.abs(nu-self.nu0)>0.8*self.width_nu),
                            nu/self.nu0,0)
        test_func(func=quadratic,grid_width=4,n_grid_elements=100,rtol=1e-2)
        test_func(func=Gaussian,grid_width=4,n_grid_elements=100,rtol=1e-2)
        test_func(func=nasty,grid_width=7,n_grid_elements=600,rtol=7e-2)


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


class TestTransition():

    up = atomic_transition.Level(g=1,E=1,number=1)
    low = atomic_transition.Level(g=1,E=0,number=0)
    line_profile_type = 'rectangular'
    A21 = 1
    general_transition = atomic_transition.Transition(up=up,low=low)
    radiative_transition = atomic_transition.RadiativeTransition(
                                                          up=up,low=low,A21=A21)
    width_v = 1*constants.kilo
    Tkin_data = np.array((1,2,3,4,5))
    log_Tkin_data = np.log(Tkin_data)
    test_emission_line = atomic_transition.EmissionLine(
                              up=up,low=low,A21=A21,line_profile_type=line_profile_type,
                              width_v=width_v)

    @pytest.mark.filterwarnings('ignore:invalid value','ignore:divide by zero')
    def test_Tex(self):
        assert self.general_transition.Tex(x1=np.zeros(1),x2=np.zeros(1))[0] == 0
        assert self.general_transition.Tex(x1=np.array((1,)),x2=np.zeros(1))[0] == 0
        assert self.general_transition.Tex(x1=np.zeros(1),x2=np.array((0.5,)))[0] == 0

    def test_radiative_transition_negative_DeltaE(self):
        with pytest.raises(AssertionError):
            atomic_transition.RadiativeTransition(up=self.low,low=self.up,A21=self.A21)

    def test_radiative_transition_wrong_nu0(self):
        wrong_nu0 = (self.up.E-self.low.E)/constants.h*1.01
        with pytest.raises(AssertionError):
            atomic_transition.RadiativeTransition(up=self.up,low=self.low,A21=self.A21,
                                                  nu0=wrong_nu0)
        with pytest.raises(AssertionError):
            atomic_transition.EmissionLine(
                              up=self.up,low=self.low,A21=1,
                              line_profile_type=self.line_profile_type,
                              width_v=self.width_v,nu0=wrong_nu0)

    def test_emission_line_constructor(self):
        assert self.test_emission_line.nu0 == self.test_emission_line.line_profile.nu0
        assert self.test_emission_line.tau_kwargs['A21'] == self.test_emission_line.A21
        assert self.test_emission_line.tau_kwargs['g_up'] == self.test_emission_line.up.g
        assert self.test_emission_line.tau_kwargs['g_low'] == self.test_emission_line.low.g

    def test_constructor_from_radiative_transition(self):
        emission_line = atomic_transition.EmissionLine.from_radiative_transition(
                               radiative_transition=self.radiative_transition,
                               line_profile_type=self.line_profile_type,
                               width_v=self.width_v)
        assert emission_line.nu0 == self.radiative_transition.nu0
        assert emission_line.A21 == self.radiative_transition.A21
        assert emission_line.B12 == self.radiative_transition.B12
        assert emission_line.B21 == self.radiative_transition.B21
        assert emission_line.up.E == self.radiative_transition.up.E
        assert emission_line.low.E == self.radiative_transition.low.E
        wrong_nu0_rad_trans = atomic_transition.RadiativeTransition(
                                                       up=self.up,low=self.low,
                                                       A21=self.A21)
        wrong_nu0_rad_trans.nu0 = wrong_nu0_rad_trans.nu0*1.01
        with pytest.raises(AssertionError):
            atomic_transition.EmissionLine.from_radiative_transition(
                               radiative_transition=wrong_nu0_rad_trans,
                               line_profile_type=self.line_profile_type,
                               width_v=self.width_v)

    def test_coll_transition_constructor(self):
        negative_21 = np.array((1,2,-2))
        with pytest.raises(AssertionError):
            atomic_transition.CollisionalTransition(
                      up=self.up,low=self.low,K21_data=negative_21,
                      Tkin_data=self.Tkin_data)

    def test_tau_nu_shape(self):
        nu = np.ones((2,4,7))*50
        N1 = 1
        N2 = 3
        tau_nu = self.test_emission_line.tau_nu(N1=N1,N2=N2,nu=nu)
        assert tau_nu.shape == nu.shape

    def test_tau_nu0(self):
        N1 = 1
        N2 = 3
        tau_nu0 = self.test_emission_line.tau_nu0(N1=N1,N2=N2)
        assert tau_nu0 == self.test_emission_line.tau_nu(
                                   N1=N1,N2=N2,nu=self.test_emission_line.nu0)

    def test_coll_coeffs(self):
        K21_data_sets = [np.array((2,1,4,6,3)),np.array((2,1,0,0,3))]
        for K21_data in K21_data_sets:
            coll_trans = atomic_transition.CollisionalTransition(
                               up=self.up,low=self.low,K21_data=K21_data,
                               Tkin_data=self.Tkin_data)
            def get_K12(K21,Tkin):
                return K21*coll_trans.up.g/coll_trans.low.g\
                          *np.exp(-coll_trans.Delta_E/(constants.k*Tkin))
            test_Tkin = [3.56,np.array((1,4.56))]
            for Tkin in test_Tkin:
                expected_K21 = np.interp(np.log(Tkin),self.log_Tkin_data,K21_data)
                expected_K12 = get_K12(K21=expected_K21,Tkin=Tkin)
                K12,K21 = coll_trans.coeffs(Tkin=Tkin)
                assert np.all(K12 == expected_K12)
                assert np.all(K21 == expected_K21)
            #test also with absolute values:
            Tkin = np.array([1,4])
            K12,K21 = coll_trans.coeffs(Tkin=Tkin)
            assert np.all(K21 == K21_data[[0,3]])
            assert np.all(K12 == get_K12(K21=K21,Tkin=Tkin))

    def test_coll_coeff_invalid_T(self):
        K21_data = np.array((2,1,0,0,3))
        coll_trans = atomic_transition.CollisionalTransition(
                           up=self.up,low=self.low,K21_data=K21_data,
                           Tkin_data=self.Tkin_data)
        invalid_Tkin = [0.5,300,self.Tkin_data+2]
        for Tkin in invalid_Tkin:
            with pytest.raises(AssertionError):
                coll_trans.coeffs(Tkin)