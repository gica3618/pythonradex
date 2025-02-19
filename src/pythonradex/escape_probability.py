# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:31:02 2017

@author: gianni
"""
import numpy as np
import numba as nb
from pythonradex import escape_probability_functions
from scipy import constants


class EscapeProbabilityUniformSphere():
    
    '''Represents the escape probability from a uniform spherical medium.'''

    def __init__(self):
        self.beta = escape_probability_functions.beta_uniform_sphere


class UniformSphere(EscapeProbabilityUniformSphere):

    '''Represents the escape probability and emerging flux from a uniform
    spherical medium'''    
    
    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def compute_flux_nu(tau_nu,source_function,solid_angle):
        '''Computes the observed flux in [W/m2/Hz], given the optical depth
        tau_nu, the source function in [W/m2/Hz/sr] and the solid angle in [sr].'''
        #see old Osterbrock book for this formula, appendix 2
        #this is the flux per surface of the emitting region
        #for lower tau_nu, the Osterbrock formula becomes numerically unstable,
        #so we use a Taylor expansion
        min_tau_nu = 1e-2
        stable_region = tau_nu > min_tau_nu
        flux_nu = 2*np.pi*source_function/tau_nu**2\
                   *(tau_nu**2/2-1+(tau_nu+1)*np.exp(-tau_nu))
        flux_nu_Taylor = 2*np.pi*source_function*(tau_nu/3-tau_nu**2/8+tau_nu**3/30
                          -tau_nu**4/144) #from Wolfram Alpha
        flux_nu = np.where(stable_region,flux_nu,flux_nu_Taylor)
        assert np.all(np.isfinite(flux_nu))
        #the observed flux is (flux at sphere surface)*4*pi*r**2/(4*pi*d**2)
        #=F_surface*Omega*4/(4*pi)
        return flux_nu*solid_angle/np.pi

@nb.jit(nopython=True,cache=True)
def exp_tau_factor(tau_nu):
    #for very small tau, the exp can introduce numerical errors, so we
    #take care of that when computing (1-exp(-tau))
    return np.where(tau_nu<1e-5,tau_nu,1-np.exp(-tau_nu))


class Flux1D():

    '''Represents the computation of the flux when considering only a single direction'''

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def compute_flux_nu(tau_nu,source_function,solid_angle):
        '''Computes the observed flux in [W/m2/Hz], given the optical depth
        tau_nu, the source function in [W/m2/Hz/sr] and the solid angle of
        the source (in [sr]) seen by the observer.'''
        tau_factor = exp_tau_factor(tau_nu=tau_nu)
        return source_function*tau_factor*solid_angle


class UniformSphereRADEX(EscapeProbabilityUniformSphere,Flux1D):
    """Represents the escape probability from a uniform sphere, but uses the
    single direction assumption to compute the emerging flux. This is what is
    done in the original RADEX code."""
    #see line 288 and following in io.f of RADEX

    pass


class UniformSlab(Flux1D):
    #Since I assume the source is in the far field, it is ok to calculate the flux
    #with the 1D formula
    '''Represents the escape probability and emerging flux from a uniform
    slab'''

    def __init__(self):
        self.beta = escape_probability_functions.beta_uniform_slab


class UniformLVGSlab(Flux1D):
    """The escape probability and flux for a uniform large velocity gradient (LVG)
    slab"""

    def __init__(self):
        self.beta = escape_probability_functions.beta_LVG_slab


class UniformLVGSphere():
    """The escape probability and flux for a uniform large velocity gradient (LVG)
    sphere"""

    def __init__(self):
        self.beta = escape_probability_functions.beta_LVG_sphere

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def compute_flux_nu(tau_nu,source_function,solid_angle,nu,nu0,V):
        #V is the velocity at the surface of the sphere
        #this formula can be derived by using an approach similar to
        #de Jong et al. (1975, Fig. 3)
        tau_factor = exp_tau_factor(tau_nu=tau_nu)
        v = constants.c*(1-nu/nu0)
        return source_function*tau_factor*solid_angle*(1-(v/V)**2)


class LVGSphereRADEX(Flux1D):
    """The escape probability and flux for a large velocity gradient (LVG)
    sphere (Expanding sphere in RADEX terminology), using the same
    formula for the escape probability as the RADEX code"""

    def __init__(self):
        self.beta = escape_probability_functions.beta_LVG_sphere_RADEX