# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:03:55 2017

@author: gianni
"""

from pythonradex import helpers
import numpy as np
from scipy import constants
import numba as nb


@nb.jit(nopython=True,cache=True)
def fast_tau_nu(A21,phi_nu,g_low,g_up,N1,N2,nu):
    return constants.c**2/(8*np.pi*nu**2)*A21*phi_nu*(g_up/g_low*N1-N2)

@nb.jit(nopython=True,cache=True,error_model='numpy')
def fast_Tex(Delta_E,g_low,g_up,x1,x2):
    return np.where((x1==0) & (x2==0),0,
                    -Delta_E/(constants.k*np.log(g_low*x2/(g_up*x1))))

@nb.jit(nopython=True,cache=True)
def fast_coll_coeffs(Tkin,Tkin_data,K21_data,gup,glow,Delta_E):
    #the following is not working because of a bug:
    # assert np.all((Tmin <= Tkin) & (Tkin <= Tmax)),\
    #           'Requested Tkin out of interpolation range. '\
    #                 +f'Tkin must be within {Tmin}-{Tmax} K'
    #instead I do a for loop:
    Tmin,Tmax = np.min(Tkin_data),np.max(Tkin_data)
    for T in Tkin:
        assert Tmin <= T,f'requested T={T} is below Tmin={Tmin},'\
                              +'i.e. outside the interpolation range'
        assert T <= Tmax, f'requested T={T} is above Tmax={Tmax},'\
                              +'i.e. outside the interpolation range'
    logTkin = np.log(Tkin)
    log_Tkin_data = np.log(Tkin_data)
    #interpolate in log space if possible
    if np.all(K21_data>0):
        log_K21_data = np.log(K21_data)
        logK21 = np.interp(logTkin,log_Tkin_data,log_K21_data)
        K21 = np.exp(logK21)
    else:
        K21 = np.interp(logTkin,log_Tkin_data,K21_data)
    #see RADEX manual for following formula
    K12 = (gup/glow*K21*np.exp(-Delta_E/(constants.k*Tkin)))
    return [K12,K21]


class LineProfile():
    
    '''Abstract class representing a general, normalised line profile.'''

    n_nu_elements_for_coarse_array = 10

    def __init__(self,nu0,width_v):
        self.nu0 = nu0
        self.width_v = width_v
        self.width_nu = helpers.Delta_nu(Delta_v=self.width_v,nu0=self.nu0)
        self.initialise_phi_nu_params()
        self.initialise_coarse_nu_array()
        self.coarse_phi_nu_array = self.phi_nu(nu=self.coarse_nu_array)
        self.phi_nu0 = self.phi_nu(nu=self.nu0)

    def initialise_phi_nu_params(self):
        raise NotImplementedError
    
    def initialise_coarse_nu_array(self):
        raise NotImplementedError

    def phi_nu(self,nu):
        r'''The value of the line profile in frequency space.
        
        The line profile is normalised such that its integral over frequency equals 1.

        Args:
            nu (float or numpy.ndarray): frequency in [Hz]
        
        Returns:
            float or numpy.ndarray: The line profile in [Hz\ :sup:`-1`]
        '''
        raise NotImplementedError

    def phi_v(self,v):
        r'''The value of the line profile in velocity space.
        
        The line profile is normalised such that its integral over velocity equals 1.

        Args:
            v (float or numpy.ndarray): velocity in [m/s]
        
        Returns:
            float or numpy.ndarra: The line profile in [(m/s)\ :sup:`-1`]
        '''
        nu = self.nu0*(1-v/constants.c)
        return self.phi_nu(nu)*self.nu0/constants.c


class GaussianLineProfile(LineProfile):

    """Represents a Gaussian line profile
    
    Attributes:
        nu0 (:obj:`float`): rest frequency in [Hz]
        width_nu (:obj:`float`): FWHM in [Hz]
        width_v (:obj:`float`): FWHM in [m/s]
    """

    def initialise_phi_nu_params(self):
        self.sigma_nu = helpers.FWHM2sigma(self.width_nu)
        self.normalisation = 1/(self.sigma_nu*np.sqrt(2*np.pi))

    def initialise_coarse_nu_array(self):
        #choose +- 1.57 FWHM, where the Gaussian is at ~0.1% of the peak
        self.coarse_nu_array = np.linspace(self.nu0-1.57*self.width_nu,
                                           self.nu0+1.57*self.width_nu,
                                           self.n_nu_elements_for_coarse_array)

    def phi_nu(self,nu):
        return self.normalisation*np.exp(-(nu-self.nu0)**2/(2*self.sigma_nu**2))


class RectangularLineProfile(LineProfile):

    '''Represents a rectangular line profile, i.e. constant over width_v, 0 outside'''

    def initialise_phi_nu_params(self):
        self.normalisation = 1/self.width_nu

    def initialise_coarse_nu_array(self):
        self.coarse_nu_array = np.linspace(self.nu0-self.width_nu*0.55,
                                           self.nu0+self.width_nu*0.55,
                                           self.n_nu_elements_for_coarse_array)

    def phi_nu(self,nu):
        inside_line = (nu>self.nu0-self.width_nu/2) & (nu<self.nu0+self.width_nu/2)
        return np.where(inside_line,self.normalisation,0)


line_profiles = {'Gaussian':GaussianLineProfile,'rectangular':RectangularLineProfile}


class Level():
    '''Represents an atomic / molecular level.

    Attributes:
        g (:obj:`float`): the statistical weight of the level
        E (:obj:`float`): the energy of the level in [J]
        number (:obj:`int`): the number of the level
    '''

    def __init__(self,g,E,number):
        self.g = g
        self.E = E
        self.number = number

    def LTE_level_pop(self,Z,T):
        '''Calculates the fractional population of the level in LTE.

        Args:
            Z (:obj:`float`): The partiion function.
            T (:obj:`float`): The temperature in [K].
        
        Returns:
            float: The fractional population of the level in LTE.
        '''
        #this is a convenience function that is not used when solving the non-LTE
        #radiative transfer
        return self.g*np.exp(-self.E/(constants.k*T))/Z


class Transition():

    def __init__(self,up,low):
        self.up = up
        self.low = low
        self.Delta_E = self.up.E-self.low.E
        self.name = f'{self.up.number}-{self.low.number}'

    def Tex(self,x1,x2):
        '''Computes the excitation temperature.
        
        Args:
            x1: (numpy.ndarray): fractional population of the lower level
            x2: (numpy.ndarray): fractional population of the upper level

        Returns:
            numpy.ndarray: excitation temperature in [K]
        '''
        return fast_Tex(Delta_E=self.Delta_E,g_up=self.up.g,g_low=self.low.g,
                        x1=x1,x2=x2)


class RadiativeTransition(Transition):

    r'''Represents the radiative transition between two energy levels.

    Attributes:
        up (pythonradex.atomic_transition.Level): the upper level of the transition
        low (pythonradex.atomic_transition.Level): the lower level of the transition
        Delta_E (:obj:`float`): the energy difference between the upper and lower level
        name (:obj:`str`): name of the transition
        nu0 (:obj:`float`): rest frequency in [Hz]
        A21 (:obj:`float`): Einstein A21 coefficient in [s\ :sup:`-1`]
        B21 (:obj:`float`): Einstein B21 coefficient in [sr m\ :sup:`2` Hz / Jy]
        B12 (:obj:`float`): Einstein B12 coefficient in [sr m\ :sup:`2` Hz / Jy]
    '''

    def __init__(self,up,low,A21,nu0=None):
        Transition.__init__(self,up=up,low=low)
        assert self.Delta_E > 0, 'negative Delta_E for radiative transition'
        self.A21 = A21
        nu0_from_Delta_E = self.Delta_E/constants.h
        if nu0 is None:
            self.nu0 = nu0_from_Delta_E
        else:
            #when reading LAMDA files, it is useful to specify nu0 directly, since
            #it is sometimes given with more significant digits
            assert np.isclose(nu0_from_Delta_E,nu0,atol=0,rtol=1e-3)
            self.nu0 = nu0
        self.B21 = helpers.B21(A21=self.A21,nu=self.nu0)
        self.B12 = helpers.B12(A21=self.A21,nu=self.nu0,g1=self.low.g,g2=self.up.g)


class EmissionLine(RadiativeTransition):

    r'''Represents an emission line arising from the radiative transition between
    two levels
    
    Attributes:
        up (pythonradex.atomic_transition.Level): the upper level of the transition
        low (pythonradex.atomic_transition.Level): the lower level of the transition
        Delta_E (:obj:`float`): the energy difference between the upper and lower level
            in [J]
        name (:obj:`str`): name of the transition
        nu0 (:obj:`float`): rest frequency in [Hz]
        A21 (:obj:`float`): Einstein A21 coefficient in [s\ :sup:`-1`]
        B21 (:obj:`float`): Einstein B21 coefficient in [sr m\ :sup:`2` Hz / Jy]
        B12 (:obj:`float`): Einstein B12 coefficient in [sr m\ :sup:`2` Hz / Jy]
        line_profile (pythonradex.atomic_transition.LineProfile): object representing
            the shape of the line profile
    '''
    def __init__(self,up,low,A21,line_profile_type,width_v,nu0=None):
        RadiativeTransition.__init__(self,up=up,low=low,A21=A21,nu0=nu0)
        self.line_profile = line_profiles[line_profile_type](nu0=self.nu0,width_v=width_v)
        self.tau_kwargs = {'A21':self.A21,'g_up':self.up.g,'g_low':self.low.g}

    @classmethod
    def from_radiative_transition(cls,radiative_transition,line_profile_type,
                                  width_v):
        '''Alternative constructor, taking an instance of RadiativeTransition,
        a line profile class and the width of the line'''
        return cls(up=radiative_transition.up,low=radiative_transition.low,
                   A21=radiative_transition.A21,line_profile_type=line_profile_type,
                   width_v=width_v,nu0=radiative_transition.nu0)

    def tau_nu(self,N1,N2,nu):
        r'''Computes the optical depth
        
        Args:
            N1: (:obj:`float`): column density of molecules in the lower level
                in [m\ :sup:`-2`]
            N2: (:obj:`float`): column density of molecules in the upper level
                in [m\ :sup:`-2`]
            nu: (numpy.ndarray): frequencies in [Hz]

        Returns:
            numpy.ndarray: the optical depth at the requested frequencies
        
        '''
        return fast_tau_nu(phi_nu=self.line_profile.phi_nu(nu),N1=N1,N2=N2,nu=nu,
                           **self.tau_kwargs)

    def tau_nu0(self,N1,N2):
        r'''Computes the optical depth at the rest frequency
        
        Args:
            N1: (:obj:`float` or numpy.ndarray): column density of molecules
                in the lower level in [m\ :sup:`-2`]
            N2: (:obj:`float` or numpy.ndarray): column density of molecules
                in the upper level in [m\ :sup:`-2`]

        Returns:
            float or numpy.ndarray: the optical depth at the rest frequency
        
        '''
        return fast_tau_nu(phi_nu=self.line_profile.phi_nu0,N1=N1,N2=N2,nu=self.nu0,
                           **self.tau_kwargs)

            
class CollisionalTransition(Transition):

    '''Represent the collisional transition between two energy levels
    
    Attributes:
        up (pythonradex.atomic_transition.Level): the upper level of the transition
        low (pythonradex.atomic_transition.Level): the lower level of the transition
        Delta_E (:obj:`float`): the energy difference between the upper and lower level
            in [J]
        name (:obj:`str`): name of the transition
    ''' 

    def __init__(self,up,low,K21_data,Tkin_data):
        Transition.__init__(self,up=up,low=low)
        assert np.all(K21_data >= 0)
        self.K21_data = K21_data
        self.Tkin_data = Tkin_data

    def coeffs(self,Tkin):
        r'''
        Computes the collisional coefficients

        Args:
            Tkin (float or numpy.ndarray): kinetic temperature in [K]
        
        Returns:
            tuple: The collision coefficients K12 and K21 in [m\ :sup:`3`/s]
            at the requested temperature(s)
        
        Raises:
            AssertionError: If Tkin is outside the available temperature range.
        '''
        Tkin = np.atleast_1d(Tkin)
        K12,K21 = fast_coll_coeffs(
                         Tkin=Tkin,Tkin_data=self.Tkin_data,K21_data=self.K21_data,
                         gup=self.up.g,glow=self.low.g,Delta_E=self.Delta_E)
        if Tkin.size == 1:
            return K12[0],K21[0]
        else:
            return K12,K21