# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:31:02 2017

@author: gianni
"""
import numpy as np


class Flux0D():

    '''Represents the computation of the flux when considering only a single direction'''

    def compute_flux_nu(self,tau_nu,source_function,solid_angle):
        '''Computes the observed flux in [W/m2/Hz], given the optical depth
        tau_nu, the source function and the solid angle of the source seen by
        the observer.'''
        return source_function*(1-np.exp(-tau_nu))*solid_angle


class FluxUniformSphere():

    '''Represents the computation of the flux from a uniform sphere'''

    min_tau_nu = 1e-2

    def compute_flux_nu(self,tau_nu,source_function,solid_angle):
        '''Computes the observed flux in [W/m2/Hz], given the optical depth
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
        flux_nu = np.where(stable_region,flux_nu,flux_nu_Taylor)
        assert np.all(np.isfinite(flux_nu))
        #observed flux = (flux at sphere surface)*4*pi*r**2/(4*pi*d**2)
        #=F_surface*Omega*4/(4*pi)
        return flux_nu*solid_angle/np.pi


class TaylorEscapeProbability():

    # use Taylor expansion if tau is below epsilon; this is to avoid numerical
    #problems with the anlytical formula to compute the escape probability
    tau_epsilon = 0.00005
    min_tau = -1

    def beta_analytical(self,tau_nu):
        raise NotImplementedError

    def beta_Taylor(self,tau_nu):
        raise NotImplementedError

    def beta(self,tau_nu):
        '''Computes the escape probability from the optical depth tau_nu.'''
        #with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        #the RADEX paper advises that negativ tau (inverted population) cannot be
        #treated correctly by a non-local code like RADEX, and that results for
        #lines with tau<~1 should be ignored
        #Moreover, negative tau can make the code crash: very negative tau
        #leads to very large transition rates, which makes the matrix of the
        #rate equations ill-conditioned.
        #thus, for tau < -1, I just take abs(tau). Note that this is different
        #from RADEX: they make something quite strange: the have different
        #approximations for positive large, normal and small tau. But then they
        #use abs(tau) to decide which function to use, but then use tau in that
        #function
        tau_nu = np.atleast_1d(np.array(tau_nu))
        prob = np.ones_like(tau_nu)*np.inf
        normal_tau_region = tau_nu > self.tau_epsilon
        prob[normal_tau_region] = self.beta_analytical(tau_nu[normal_tau_region])
        small_tau_region = np.abs(tau_nu) <= self.tau_epsilon
        #here I use tau even if tau < 0:
        prob[small_tau_region] = self.beta_Taylor(tau_nu[small_tau_region])
        negative_tau_region = (tau_nu >= self.min_tau) & (tau_nu<-self.tau_epsilon)
        prob[negative_tau_region] = self.beta_analytical(tau_nu[negative_tau_region])
        unreliable_negative_region = tau_nu < self.min_tau
        #here I just use abs(tau) to stabilize the code
        prob[unreliable_negative_region] = self.beta_analytical(
                                  np.abs(tau_nu[unreliable_negative_region]))
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


class UniformFaceOnSlab(Flux0D):
    #Since I assume the source is in the far field, it is ok to calculate the flux
    #with the 0D formula
    """Represents the computation of the flux from a uniform slab that is seen
    face-on (think of a face-on disk, i.e. x-y-size of much larger than the z-size,
    where z is along the line of sight)"""

    theta = np.linspace(0,np.pi/2,200)
    tau_grid = np.logspace(-3,2,1000)
    min_tau_nu = np.min(tau_grid)

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

    def beta(self,tau_nu):
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            prob = self.interpolated_integral_term(tau_nu)/tau_nu
        #for negative tau, prob will be 1, which is fine if tau is close to 0, but
        #not correct for very negative tau, so results are not valid in that case
        prob = np.where(tau_nu<self.min_tau_nu,1,prob)
        assert np.all(np.isfinite(prob))
        return prob


class UniformShockSlabRADEX(TaylorEscapeProbability,Flux0D):
    """The escape probability for a uniform slab (shock) as in RADEX"""

    def beta_analytical(self,tau_nu):
        return (1-np.exp(-3*tau_nu))/(3*tau_nu)

    def beta_Taylor(self,tau_nu):
        return 1 - (3*tau_nu)/2 + (3*tau_nu**2)/2 - (9*tau_nu**3)/8 +\
               (27*tau_nu**4)/40 - (27*tau_nu**5)/80