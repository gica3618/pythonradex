# -*- coding: utf-8 -*-

import numpy as np
from scipy import constants
import numba as nb

#@nb.jit(nopython=True,cache=True) #doesn't help
def B21(A21,nu):
    '''Returns the Einstein B21 coefficient for stimulated emission, computed
    from the Einstein A21 coefficient and the frequency nu.'''
    return constants.c**2/(2*constants.h*nu**3)*A21

#@nb.jit(nopython=True,cache=True) #doesn't help
def B12(A21,nu,g1,g2):
    '''Einstein B12 coefficient for absorption, computed from the Einstein A21
    coefficient, the frequency nu, statistical weights g2 and g1 of upper and 
    lower level respectively.'''
    return g2/g1*B21(A21=A21,nu=nu)

@nb.jit(nopython=True,cache=True)
def B_nu(nu,T):
    """Planck function
    
    Return the value of the Planck function (black body) in [W/m2/Hz/sr].
    
    Parameters
    ----------
    nu : float or numpy.ndarray
        frequency in Hz

    T : float or numpy.ndarray
        temperature in K

    Returns
    -------
    float or numpy.ndarray
        Value of Planck function in [W/m2/Hz/sr]
    """
    return (2*constants.h*nu**3/constants.c**2
           *(np.exp(constants.h*nu/(constants.k*T))-1)**-1)

def generate_CMB_background(z=0):
    '''generates a function that gives the CMB background at redshift z
    
    Parameters
    -----------
    z: float
        redshift

    Returns
    --------
    function
        function giving CMB background in [W/m2/Hz/sr] for an input frequency in [Hz]
        '''
    T_CMB = 2.73*(1+z)
    def CMB_background(nu):
        return B_nu(nu=nu,T=T_CMB)
    return CMB_background

@nb.jit(nopython=True,cache=True)
def zero_background(nu):
    '''Zero intensity radiation field
    
    Returns zero intensity for any frequency
    
    Parameters
    ------------
    nu: array_like
        frequency in Hz
    
    Returns
    ---------------
    numpy.ndarray
        Zero at all requested frequencies'''
    return np.zeros_like(nu)

@nb.jit(nopython=True,cache=True)
def FWHM2sigma(FWHM):
    """Convert FWHM of a Gaussian to standard deviation.
    
    Parameters
    -----------
    FWHM: float or numpy.ndarray
        FWHM of the Gaussian
    
    Returns
    ------------
    float or numpy.ndarray
        the standard deviation of the Gaussian"""
    return FWHM/(2*np.sqrt(2*np.log(2)))

@nb.jit(nopython=True,cache=True)
def relative_difference(a,b):
    """Computes the elementwise relative difference between a and b.
    In general, return |a-b|/a.
    Special cases:
    a=0 and b=0: return 0
    a=0 and b!=0: return 1"""
    abs_diff = np.abs(a-b)
    rel_diff = np.where((a==0) & (b==0),0,np.where(a==0,1,abs_diff/a))
    assert not np.any(np.isnan(rel_diff))
    return np.abs(rel_diff)

#@nb.jit(nopython=True,cache=True) #doesn't help
def Delta_nu(Delta_v,nu0):
    return nu0 * Delta_v / constants.c

@nb.jit(nopython=True,cache=True)
def fast_trapz(y,x):
    return np.trapz(y,x)

@nb.jit(nopython=True,cache=True)
def assert_all_finite(x):
    assert np.all(np.isfinite(x))