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

@nb.jit(nopython=True,cache=True)
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

    n_nu_elements_for_dense_array = 200
    window_width_for_dense_nu_array = 6 #in units of the line width
    n_nu_elements_for_coarse_array = 10

    def __init__(self,nu0,width_v):
        '''nu0 is the central frequency, width_v is the width of the line in velocity
        space. The exact meaning of width_v depends on the type of the line profile'''
        self.nu0 = nu0
        self.width_v = width_v
        self.width_nu = helpers.Delta_nu(Delta_v=self.width_v,nu0=self.nu0)
        self.initialise_phi_nu_params()
        self.initialise_coarse_nu_array()
        self.coarse_phi_nu_array = self.phi_nu(nu=self.coarse_nu_array)
        w = self.window_width_for_dense_nu_array
        self.dense_nu_array = np.linspace(self.nu0-w*self.width_nu/2,
                                          self.nu0+w*self.width_nu/2,
                                          self.n_nu_elements_for_dense_array)
        self.dense_phi_nu_array = self.phi_nu(nu=self.dense_nu_array)
        self.phi_nu0 = self.phi_nu(nu=self.nu0)

    def initialise_phi_nu_params(self):
        raise NotImplementedError
    
    def initialise_coarse_nu_array(self):
        raise NotImplementedError

    def phi_nu(self,nu):
        raise NotImplementedError

    def phi_v(self,v):
        nu = self.nu0*(1-v/constants.c)
        return self.phi_nu(nu)*self.nu0/constants.c


class GaussianLineProfile(LineProfile):

    """Represents a Gaussian line profile; the width_v parameter is interpreted
    as FWHM"""

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
        self.g = g
        self.E = E
        self.number = number

    def LTE_level_pop(self,Z,T):
        '''LTE level population for partition function Z and temperature T'''
        #this is a convenience function that is not used when solving the non-LTE
        #radiative transfer
        return self.g*np.exp(-self.E/(constants.k*T))/Z


class Transition():

    '''Represents a transition between two energy levels'''

    def __init__(self,up,low):
        '''up and low are instances of the Level class, representing the upper
        and lower level of the transition'''
        self.up = up
        self.low = low
        self.Delta_E = self.up.E-self.low.E
        self.name = f'{self.up.number}-{self.low.number}'

    def Tex(self,x1,x2):
        '''
        Computes the excitation temperature from the fractional level population
        
        Parameters
        ------------
        x1: float or numpy.ndarray
            fractional population of the lower level
        x2: float or numpy.ndarray
            fractional population of the upper level

        Returns
        ---------
        numpy.ndarray
            excitation temperature in K
        '''
        return fast_Tex(Delta_E=self.Delta_E,g_up=self.up.g,g_low=self.low.g,
                        x1=x1,x2=x2)


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

    def __init__(self,up,low,A21,nu0=None):
        '''up and low are instances of the Level class, representing the upper
        and lower level of the transition. A21 is the Einstein coefficient for
        spontaneous emission. The optinal argument nu0 is the line frequency; if not
        given, nu0 will be calculated from the level energies.'''
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

    '''Represents an emission line arising from the radiative transition between
    two levels'''

    def __init__(self,up,low,A21,line_profile_cls,width_v,nu0=None):
        '''up and low are instances of the Level class, representing the upper
        and lower level of the transition. A21 is the Einstein coefficient,
        line_profile_cls is the line profile class to be used,
        and width_v the width of the line in [m/s]. The optinal argument nu0 is the
        line frequency in [Hz]; if not given, nu0 will be calculated from
        the level energies.'''
        RadiativeTransition.__init__(self,up=up,low=low,A21=A21,nu0=nu0)
        self.line_profile = line_profile_cls(nu0=self.nu0,width_v=width_v)
        self.tau_kwargs = {'A21':self.A21,'g_up':self.up.g,'g_low':self.low.g}

    @classmethod
    def from_radiative_transition(cls,radiative_transition,line_profile_cls,
                                  width_v):
        '''Alternative constructor, taking an instance of RadiativeTransition,
        a line profile class and the width of the line'''
        return cls(up=radiative_transition.up,low=radiative_transition.low,
                   A21=radiative_transition.A21,line_profile_cls=line_profile_cls,
                   width_v=width_v,nu0=radiative_transition.nu0)

    def tau_nu(self,N1,N2,nu):
        '''Compute the optical depth from the column densities N1 and N2 (in m-2)
        in the lower and upper level respectively, and the frequency nu (in Hz).'''
        return fast_tau_nu(phi_nu=self.line_profile.phi_nu(nu),N1=N1,N2=N2,nu=nu,
                           **self.tau_kwargs)

    def dense_tau_nu_array(self,N1,N2):
        '''Compute the optical depth from the column densities N1 and N2 (in m-2)
        in the lower and upper level respectively. Returns an array corresponding to the
        dense frequencies grid defined in the line profile'''
        return fast_tau_nu(phi_nu=self.line_profile.dense_phi_nu_array,N1=N1,N2=N2,
                           nu=self.line_profile.dense_nu_array,**self.tau_kwargs)

    def tau_nu0(self,N1,N2):
        '''Computes the optical depth at the line center from the column densities
        N1 and N2 (in m-2) in the lower and upper level respectively.'''
        return fast_tau_nu(phi_nu=self.line_profile.phi_nu0,N1=N1,N2=N2,nu=self.nu0,
                           **self.tau_kwargs)

            
class CollisionalTransition(Transition):

    '''Represent the collisional transtion between two energy levels
    
    Attributes:
    -------------    

    - up: Level
        upper level

    - low: Level
        lower level

    - Delta_E: float
        energy difference [Jy] between upper and lower level

    - name: str
        transition name, for example '3-2' for the transition between the fourth and the
        third level

    - K21_data: numpy.ndarray
        value of the collision rate coefficient K21 [m3/s] at different temperatures

    - Tkin_data: numpy.ndarray
        the temperature values corresponding to the K21 values
    ''' 

    def __init__(self,up,low,K21_data,Tkin_data):
        '''up and low are instances of the Level class, representing the upper
        and lower level of the transition. K21_data is an array of rate coefficients
        in [m3/s] over the temperatures defined in the array Tkin_data'''
        Transition.__init__(self,up=up,low=low)
        assert np.all(K21_data >= 0)
        self.K21_data = K21_data
        self.Tkin_data = Tkin_data

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
        tuple
            The collision coefficients K12 and K21 in [m3/s] at the requested
            temperature(s)'''
        Tkin = np.atleast_1d(Tkin)
        K12,K21 = fast_coll_coeffs(
                         Tkin=Tkin,Tkin_data=self.Tkin_data,K21_data=self.K21_data,
                         gup=self.up.g,glow=self.low.g,Delta_E=self.Delta_E)
        if Tkin.size == 1:
            return K12[0],K21[0]
        else:
            return K12,K21