# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:31:02 2017

@author: gianni
"""
import numpy as np


class Flux0D():

    '''Represents the computation of the flux when considering only a single direction'''

    def compute_flux_nu(self,tau_nu,source_function):
        '''Computes the emerging flux in [W/m2/Hz], given the optical depth
        tau_nu and the source function.'''
        return np.pi*source_function*(1-np.exp(-tau_nu))


class FluxUniformSlab():

    '''Represents the computation of the flux from a uniform slab'''
    theta = np.linspace(0,np.pi/2,200)
    tau_grid = np.logspace(-3,2,1000)

    def __init__(self):
        #the expression for the flux contains an integral term;
        #here I pre-compute this term so it can be interpolated to speed up the code
        self.integral_term_grid = np.array([self.integral_term(tau) for tau in
                                            self.tau_grid])

    def integral_term(self,tau):
        return np.trapz((1-np.exp(-tau/np.cos(self.theta)))*np.cos(self.theta)
                        *np.sin(self.theta),self.theta)

    def interpolated_integral_term(self,tau):
        interp = np.interp(x=tau,xp=self.tau_grid,fp=self.integral_term_grid,
                          left=0,right=0.5)
        return np.where(tau<np.min(self.tau_grid),tau,interp)

    def compute_flux_nu(self,tau_nu,source_function):
        '''Computes the emerging flux in [W/m2/Hz], given the optical depth
        tau_nu and the source function.'''
        return 2*np.pi*source_function*self.interpolated_integral_term(tau_nu)


class FluxUniformSphere():

    '''Represents the computation of the flux from a uniform sphere'''

    min_tau_nu = 1e-2

    def compute_flux_nu(self,tau_nu,source_function):
        '''Computes the emerging flux in [W/m2/Hz], given the optical depth
        tau_nu and the source function.'''
        #see old Osterbrock book for this formula, appendix 2
        #this is the flux per surface of the emitting region
        #convert to numpy array to avoid ZeroDivisionError:
        tau_nu = np.array(tau_nu)
        #for lower tau_nu, the Osterbrock formula becomes numerically unstable:
        stable_region = tau_nu > self.min_tau_nu
        with np.errstate(divide='ignore',invalid='ignore'):
            flux_nu = 2*np.pi*source_function/tau_nu**2\
                       *(tau_nu**2/2-1+(tau_nu+1)*np.exp(-tau_nu))
        flux_nu_Taylor = 2*np.pi*source_function*(tau_nu/3-tau_nu**2/8+tau_nu**3/30
                          -tau_nu**4/144) #from Wolfram Alpha
        flux = np.where(stable_region,flux_nu,flux_nu_Taylor)
        assert np.all(np.isfinite(flux))
        return flux


class TaylorEscapeProbability():

    # use Taylor expansion if tau is below epsilon; this is to avoid numerical
    #problems with the anlytical formula to compute the escape probability
    tau_epsilon = 0.00005

    def beta_analytical(self,tau_nu):
        raise NotImplementedError

    def beta_Taylor(self,tau_nu):
        raise NotImplementedError

    def beta(self,tau_nu):
        '''Computes the escape probability from the optical depth tau_nu'''
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            prob = np.where(tau_nu < self.tau_epsilon,self.beta_Taylor(tau_nu),
                            self.beta_analytical(tau_nu))
        assert np.all(np.isfinite(prob))
        return prob
    

class EscapeProbabilityUniformSphere(TaylorEscapeProbability):
    
    '''Represents the escape probability from a uniform spherical medium.'''    

    def beta_analytical(self,tau_nu):
        '''Computes the escape probability analytically, given the optical
        depth tau_nu'''
        #see the RADEX manual for this formula; derivation is found in the old
        #Osterbrock (1974) book, appendix 2. Note that Osterbrock uses tau for
        #radius, while I use it for diameter
        #convert to numpy array to avoid ZeroDivisionError (numpy converts to inf
        #instead of raising an error)
        tau_nu = np.array(tau_nu)
        return 1.5/tau_nu*(1-2/tau_nu**2+(2/tau_nu+2/tau_nu**2)*np.exp(-tau_nu))

    def beta_Taylor(self,tau_nu):
        '''Computes the escape probability using a Taylor expansion, given the
        optical depth tau_nu'''
        #Taylor expansion of beta for uniform sphere, easier to evaluate numerically
        #(for small tau_nu)
        #Series calculated using Wolfram Alpha; not so easy analytically, to
        #calculate the limit as tau->0, use rule of L'Hopital
        return (1 - 0.375*tau_nu + 0.1*tau_nu**2 - 0.0208333*tau_nu**3
                + 0.00357143*tau_nu**4)


class UniformSphere(EscapeProbabilityUniformSphere,FluxUniformSphere):

    '''Represents the escape probability and emerging flux from a uniform spherical medium'''    
    
    pass


class UniformSphereRADEX(EscapeProbabilityUniformSphere,Flux0D):
    """Represents the escape probability from a uniform sphere, but uses the
    single direction assumption to compute the emerging flux. This is what is done in the
    original RADEX code."""

    pass


class UniformSlab(FluxUniformSlab):
    """The escape probability and emerging flux from a uniform slab"""

    min_tau_nu = np.min(FluxUniformSlab.tau_grid)

    def beta(self,tau_nu):
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            prob = FluxUniformSlab.interpolated_integral_term(self,tau_nu)/tau_nu
        prob = np.where(tau_nu<self.min_tau_nu,1,prob)
        assert np.all(np.isfinite(prob))
        return prob


class UniformSlabRADEX(TaylorEscapeProbability,Flux0D):
    """The escape probability for a uniform slab as in RADEX"""

    def beta_analytical(self,tau_nu):
        return (1-np.exp(-3*tau_nu))/(3*tau_nu)

    def beta_Taylor(self,tau_nu):
        return 1 - (3*tau_nu)/2 + (3*tau_nu**2)/2 - (9*tau_nu**3)/8 +\
               (27*tau_nu**4)/40 - (27*tau_nu**5)/80