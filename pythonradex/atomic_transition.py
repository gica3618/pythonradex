# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:03:55 2017

@author: gianni
"""

from pythonradex import helpers
import numpy as np
from scipy import constants
import warnings


class LineProfile():
    
    '''Abstract class representing a general, normalised line profile.'''    
    
    n_nu_elements = 700
    #width of the frequency window we consider, in units of width of the line:
    window_width = 6
    
    def __init__(self,nu0,width_v):
        '''nu0 is the central frequency, width_v is the width of the line in velocity
        space. The exact meaning of width_v depends on the type of the line profile'''
        self.nu0 = nu0
        self.width_v = width_v
        #transfer width to frequency:
        self.width_nu = helpers.Delta_nu(Delta_v=self.width_v,nu0=self.nu0)
        self.initialise_phi_nu_params()
        #array of nu values covering the line
        self.nu_array = np.linspace(self.nu0-self.window_width/2*self.width_nu,
                                    self.nu0+self.window_width/2*self.width_nu,
                                    self.n_nu_elements)
        self.phi_nu_array = self.phi_nu(self.nu_array)

    def initialise_phi_nu_params(self):
        raise NotImplementedError

    def phi_nu(self,nu):
        raise NotImplementedError

    def phi_v(self,v):
        nu = self.nu0*(1-v/constants.c)
        return self.phi_nu(nu)*self.nu0/constants.c

    def average_over_nu_array(self,x_nu_array):
        """Computes the average over the line profile of some quantity
        (i.e. line profile weighted integral), given
        as an array x_nu_array which has to be defined over the frequencies of
        the nu_array of the LineProfile instance."""
        return np.trapz(x_nu_array*self.phi_nu_array,self.nu_array)


class GaussianLineProfile(LineProfile):

    """Represents a Gaussian line profile; the width_v parameter is interpreted
    as FWHM"""

    def initialise_phi_nu_params(self):
        '''compute the standard deviation and the normalisation of the line profile
        and set them as attributes'''
        self.sigma_nu = helpers.FWHM2sigma(self.width_nu)
        self.normalisation = 1/(self.sigma_nu*np.sqrt(2*np.pi))

    def phi_nu(self,nu):
        '''compute the normalised line profile for the frequency nu'''
        return self.normalisation*np.exp(-(nu-self.nu0)**2/(2*self.sigma_nu**2))


class SquareLineProfile(LineProfile):

    '''Represents a square line profile, i.e. constant over width_v, 0 outside'''

    def initialise_phi_nu_params(self):
        '''compute the normalisation of the line profile'''
        self.normalisation = 1/self.width_nu
        
    def phi_nu(self,nu):
        '''compute the normalised line profile for the frequency nu'''
        inside_line = (nu>self.nu0-self.width_nu/2) & (nu<self.nu0+self.width_nu/2)
        return np.where(inside_line,self.normalisation,0)


class Level():

    '''Represents an atomic/molecular energy level

    Attributes:
    ------------

    - g: float
        statistical weight

    - E: float
        energy in [J]

    - number: int
        the level number (0 for the lowest level)'''

    def __init__(self,g,E,number):
        '''statistical weight g, energy E and number of the level'''
        self.g = g
        self.E = E
        self.number = number

    def LTE_level_pop(self,Z,T):
        '''LTE level population for partition function Z and temperature T'''
        return self.g*np.exp(-self.E/(constants.k*T))/Z


class Transition():

    '''Represents the transition between two energy levels'''

    def __init__(self,up,low):
        '''up and low are instances of the Level class, representing the upper
        and lower level of the transition'''
        self.up = up
        self.low = low
        self.Delta_E = self.up.E-self.low.E
        self.name = '{:d}-{:d}'.format(self.up.number,self.low.number)

    def Tex(self,x1,x2):
        '''
        Excitation temperature
        
        Computes the excitation temperature from the fractional population
        
        Parameters
        ------------
        x1: array_like
            fractional population of the lower level
        x2: array_like
            fractional population of the upper level

        Returns
        ---------
        numpy.ndarray
            excitation temperature in K
        '''
        x1,x2 = np.array(x1),np.array(x2)
        return np.where((x1==0) & (x2==0),0,
                        -self.Delta_E/(constants.k*np.log(self.low.g*x2/(self.up.g*x1))))


class RadiativeTransition(Transition):

    '''Represents the radiative transition between two energy levels

    Attributes:
    ------------
    
    - up: Level
        upper level

    - low: Level
        lower level
        
    - Delta_E: float
        energy difference between upper and lower level

    - name: str
        transition name, for example '3-2' for the transition between the fourth and the
        third level

    - A21: float 
        Einstein A21 coefficient

    - nu0: float
        central frequency of the transition

    - B21: float
        Einstein B21 coefficient

    - B12: float
        Einstein B12 coefficient'''

    def __init__(self,up,low,A21):
        '''up and low are instances of the Level class, representing the upper
        and lower level of the transition. A21 is the Einstein coefficient for
        spontaneous emission'''
        Transition.__init__(self,up=up,low=low)
        assert self.Delta_E > 0, 'negative Delta_E for radiative transition'
        self.A21 = A21
        self.nu0 = self.Delta_E/constants.h
        self.B21 = helpers.B21(A21=self.A21,nu=self.nu0)
        self.B12 = helpers.B12(A21=self.A21,nu=self.nu0,g1=self.low.g,g2=self.up.g)


class EmissionLine(RadiativeTransition):

    '''Represents an emission line arising from the radiative transition between
    two levels'''

    def __init__(self,up,low,A21,line_profile_cls,width_v):
        '''up and low are instances of the Level class, representing the upper
        and lower level of the transition. A21 is the Einstein coefficient,
        line_profile_cls is the line profile class to be used,
        and width_v the witdht of the line'''
        RadiativeTransition.__init__(self,up=up,low=low,A21=A21)
        self.line_profile = line_profile_cls(nu0=self.nu0,width_v=width_v)

    @classmethod
    def from_radiative_transition(cls,radiative_transition,line_profile_cls,
                                  width_v):
        '''Alternative constructor, taking an instance of RadiativeTransition,
        a line profile class and the width of the line'''
        return cls(up=radiative_transition.up,low=radiative_transition.low,
                   A21=radiative_transition.A21,line_profile_cls=line_profile_cls,
                   width_v=width_v)

    def tau_nu(self,N1,N2,nu):
        '''Compute the optical depth from the column densities N1 and N2 in the lower
        and upper level respectively.'''
        return (constants.c**2/(8*np.pi*nu**2)
                *self.A21*self.line_profile.phi_nu(nu)
                *(self.up.g/self.low.g*N1-N2))

    def tau_nu_array(self,N1,N2):
        '''Compute the optical depth from the column densities N1 and N2 in the lower
        and upper level respectively. Returns an array corresponding to the
        frequencies defined in the line profile'''
        return self.tau_nu(N1=N1,N2=N2,nu=self.line_profile.nu_array)

    def tau_nu0(self,N1,N2):
        '''Computes the optical depth at the line center from the column densities
        N1 and N2 in the lower and upper level respectively.'''
        tau_nu0 = np.interp(self.nu0,self.line_profile.nu_array,
                            self.tau_nu_array(N1=N1,N2=N2))
        return tau_nu0


class CollisionalTransition(Transition):

    '''Represent the collisional transtion between two energy levels
    
    Attributes:
    -------------    

    - up: Level
        upper level

    - low: Level
        lower level

    - Delta_E: float
        energy difference between upper and lower level

    - name: str
        transition name, for example '3-2' for the transition between the fourth and the
        third level

    - K21_data: numpy.ndarray
        value of the collision rate coefficient K21 at different temperatures

    - log_K21_data: numpy.ndarray
         the logarithm of K21_data

    - Tkin_data: numpy.ndarray
        the temperature values corresponding to the K21 values

    - log_Tkin_data: numpy.ndarray
        the logarithm of the temperature values

    - Tmax: float
        the maximum temperature value

    - Tmin: float
        the minimum temperature value
    ''' 

    def __init__(self,up,low,K21_data,Tkin_data):
        '''up and low are instances of the Level class, representing the upper
        and lower level of the transition. K21_data is an array of rate coefficients
        over the temperatures defined in the array Tkin_data'''
        Transition.__init__(self,up=up,low=low)
        #set of coeffs for different temperatures
        assert np.all(K21_data >= 0)
        self.K21_data = K21_data
        if np.all(self.K21_data>0):
            self.K21_data_all_larger_than_0 = True
            self.log_K21_data = np.log(self.K21_data)
        else:
            self.K21_data_all_larger_than_0 = False
        #array of temperatures for which rate coeffs are available:
        self.Tkin_data = Tkin_data
        self.log_Tkin_data = np.log(self.Tkin_data)
        self.Tmax = np.max(self.Tkin_data)
        self.Tmin = np.min(self.Tkin_data)

    def coeffs(self,Tkin):
        '''
        collisional transition rates
        
        computes the collisional transition rate coefficients by interpolation.

        Parameters
        -----------
        Tkin: array_like
            kinetic temperature in K. Must be within the interpolation range.
        
        Returns
        ---------
        dict
            The keys "K12" and "K21" of the dict correspond to the collision coefficients at
            the requested temperature(s)'''
        #interpolate in log space
        assert np.all((self.Tmin <= Tkin) & (Tkin <= self.Tmax)),\
            'Requested Tkin out of interpolation range. Tkin must be within {:g}-{:g} K'.format(
                                                                self.Tmin,self.Tmax)
        logTkin = np.log(Tkin)
        if self.K21_data_all_larger_than_0:
            logK21 = np.interp(logTkin,self.log_Tkin_data,self.log_K21_data)
            K21 = np.exp(logK21)
        else:
            K21 = np.interp(logTkin,self.log_Tkin_data,self.K21_data)
        #see RADEX manual for following formula
        K12 = (self.up.g/self.low.g*K21
               *np.exp(-self.Delta_E/(constants.k*Tkin)))
        return {'K12':K12,'K21':K21}