#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:10:38 2024

@author: gianni
"""

import numba as nb
import numpy as np
from pythonradex import helpers

#It would be much more elegant to put these functions directly into the
#classes defined in escape_probability.py. But I want to compile the beta functions
#with numba, so I have to put these functions outside the class definitions
#(defining them as staticmethod does not work here)

min_reliable_tau = -1

@nb.jit(nopython=True,cache=True)
def clip_prob(prob):
    return np.where(prob>1,1,np.where(prob<0,0,prob))

####### functions for escape probability using Taylor expansion ######

@nb.jit(nopython=True,cache=True)
def identify_tau_regions(tau_nu):
    tau_epsilon = 0.05 #important that this is not too small, to avoid the unstable region
    normal_tau_region = tau_nu > tau_epsilon
    small_tau_region = np.abs(tau_nu) <= tau_epsilon
    negative_tau_region = (tau_nu >= min_reliable_tau) & (tau_nu<-tau_epsilon)
    unreliable_negative_region = tau_nu < min_reliable_tau
    total_region_points = normal_tau_region.sum()+small_tau_region.sum()\
                          +negative_tau_region.sum()+unreliable_negative_region.sum()
    assert total_region_points==len(tau_nu),\
                          'selection needs to cover all points of tau_nu'
    return normal_tau_region,small_tau_region,negative_tau_region,\
                unreliable_negative_region

@nb.jit(nopython=True,cache=True)
def beta_analytical_uniform_sphere(tau_nu):
    '''Computes the escape probability for a uniform sphere analytically,
    given the optical depth tau_nu'''
    #see the RADEX manual for this formula; derivation is found in the old
    #Osterbrock (1974) book, appendix 2. Note that Osterbrock uses tau for
    #radius, while I use it for diameter
    return 1.5/tau_nu*(1-2/tau_nu**2+(2/tau_nu+2/tau_nu**2)*np.exp(-tau_nu))

@nb.jit(nopython=True,cache=True)
def beta_Taylor_uniform_sphere(tau_nu):
    '''Computes the escape probability of a uniform sphere using a Taylor expansion,
    given the optical depth tau_nu'''
    #Taylor expansion of beta for uniform sphere is easier to evaluate numerically
    #(for small tau_nu)
    #Series calculated using Wolfram Alpha; not so easy analytically, to
    #calculate the limit as tau->0, use rule of L'Hopital
    return 1 - 0.375*tau_nu + 0.1*tau_nu**2 - 0.0208333*tau_nu**3

@nb.jit(nopython=True,cache=True)
def beta_analytical_LVG_slab(tau_nu):
    #see line 357 in matrix.f and RADEX paper, or Scoville & Solomon (1974)
    return (1-np.exp(-3*tau_nu))/(3*tau_nu)

@nb.jit(nopython=True,cache=True)
def beta_Taylor_LVG_slab(tau_nu):
    return 1 - (3*tau_nu)/2 + (3*tau_nu**2)/2 - (9*tau_nu**3)/8

#about the LVG sphere
#we need to assume that the LVG sphere is homogeneous, because from formula 2.6.36
#in Elitzur92, it becomes clear that the optical depth in a velocity-
#coherent element depends on the local number density. But since pythonradex assumes
#uniform excitation temperature and a single optical depth characerising the source,
#the local number density must be the same everywhere
#Thus, we need to assume that the LVG sphere is uniform, otherwise we cannot handle it
#in pythonradex.
#We are indeed considering the model by Goldreich & Kwan (1974) which has uniform density.
#in fact, same argument also implies that LVG slab needs to be uniform...

@nb.jit(nopython=True,cache=True)
def beta_analytical_LVG_sphere(tau_nu):
    #e.g Elitzur p. 44, or Ramos & Elitzur 2018, eq. 14
    return (1-np.exp(-tau_nu))/tau_nu

@nb.jit(nopython=True,cache=True)
def beta_Taylor_LVG_sphere(tau_nu):
    #from Wolfram Alpha
    return 1 - tau_nu/2 + tau_nu**2/6 - tau_nu**3/24

def generate_Taylor_beta(beta_ana,beta_Taylor):
    @nb.jit(nopython=True,cache=True)
    def beta(tau_nu):
        #use Taylor expansion if tau is below epsilon; this is to avoid numerical
        #problems with the anlytical formula to compute the escape probability.
        #Concerning negative optical depth, the RADEX paper advises that
        #negativ tau (inverted population) cannot be treated correctly by a
        #non-local code like RADEX, and that results for lines with tau <~ -1
        #should be ignored. Moreover, negative tau can make the code crash:
        #very negative tau leads to very large transition rates, which makes the
        #matrix of the rate equations ill-conditioned.
        #thus, for tau < -1, I just take abs(tau). Note that this is different
        #from RADEX: they make something quite strange: they have different
        #approximations for positive large, normal and small tau. But then they
        #use abs(tau) to decide which function to use, but then use tau in that
        #function (line 333 in matrix.f)
        normal,small,negative,unreliable = identify_tau_regions(tau_nu=tau_nu)
        prob = np.empty_like(tau_nu)
        prob[normal] = beta_ana(tau_nu[normal])
        #here I use tau even if tau < 0:
        prob[small] = beta_Taylor(tau_nu[small])
        prob[negative] = beta_ana(tau_nu[negative])
        #here I just use abs(tau) to stabilize the code
        prob[unreliable] = beta_ana(np.abs(tau_nu[unreliable]))
        helpers.assert_all_finite(prob)
        return clip_prob(prob)
    return beta

beta_uniform_sphere = generate_Taylor_beta(beta_ana=beta_analytical_uniform_sphere,
                                           beta_Taylor=beta_Taylor_uniform_sphere)
beta_LVG_slab = generate_Taylor_beta(beta_ana=beta_analytical_LVG_slab,
                                     beta_Taylor=beta_Taylor_LVG_slab)
beta_LVG_sphere = generate_Taylor_beta(beta_ana=beta_analytical_LVG_sphere,
                                       beta_Taylor=beta_Taylor_LVG_sphere)

######### functions for uniform slab  ##################
#see Elitzur92 (https://ui.adsabs.harvard.edu/abs/1992ASSL..170.....E/abstract,
#Problem 2.12) for confirmation of the formulas used here

def integral_term_for_uniform_slab(tau):
    mu = np.linspace(1e-5,1,200)
    return np.trapezoid((1-np.exp(-tau/mu))*mu,mu)

tau_grid_for_UniformSlab = np.logspace(-5,2,1000)
min_grid_tau = np.min(tau_grid_for_UniformSlab)
max_grid_tau = np.max(tau_grid_for_UniformSlab)
#the expression for the flux contains an integral term;
#here I pre-compute this term so it can be interpolated to speed up the code
integral_term_grid = np.array([integral_term_for_uniform_slab(tau)
                               for tau in tau_grid_for_UniformSlab])

@nb.jit(nopython=True,cache=True)
def interpolated_integral_term(tau):
    interp = np.interp(x=tau,xp=tau_grid_for_UniformSlab,fp=integral_term_grid)
    #limiting values:
    # -very large tau: integral term goes to 0.5
    # -very small tau: integral term goes to tau
    #note that I cannot use the left and right keywords of np.interp because
    #I want to compile this function; so I use np.where instead
    return np.where(tau>max_grid_tau,0.5,np.where(tau<min_grid_tau,tau,interp))

@nb.jit(nopython=True,cache=True)
def beta_uniform_slab(tau_nu):
    #for negative tau, this function will also return 1, which is fine if tau
    #is close to 0, but not correct for very negative tau; in general, results
    #should be ignored for tau < -1
    int_term = interpolated_integral_term(tau=tau_nu)
    prob = np.where(tau_nu<min_grid_tau,1,int_term/tau_nu)
    helpers.assert_all_finite(prob)
    return clip_prob(prob)

######### LVG sphere as implemented by RADEX ##################

@nb.jit(nopython=True,cache=True)
def beta_LVG_sphere_RADEX_gtr7(tau_nu):
    tau_nu_r = tau_nu/2
    return 2/(tau_nu_r*4*(np.sqrt(np.log(tau_nu_r/np.sqrt(np.pi)))))

@nb.jit(nopython=True,cache=True)
def beta_LVG_sphere_RADEX_less7(tau_nu):
    tau_nu_r = tau_nu/2
    return 2*(1-np.exp(-2.34*tau_nu_r))/(4.68*tau_nu_r)

@nb.jit(nopython=True,cache=True)
def beta_LVG_sphere_RADEX(tau_nu):
    #RADEX paper, eq. 18, claims that they use beta_analytical_LVG_sphere
    #(beta= (1-exp(-tau))/tau
    #but in fact they use a different formula from de Jong+80
    #see line 346 in matrix.f
    #I use the same formulae as RADEX, except that I take care of negative tau
    #handling negative tau turns out to be important, e.g. for CO, Tkin=200,
    #width_v=3 km/s, Ntot=1e20 cm-2, RADEX gives an invalid solution
    assert -7 < min_reliable_tau < -0.01
    gtr7 = 7 <= tau_nu
    less7 = (0.001 <= tau_nu) & (tau_nu < 7)
    small = (-0.001 <= tau_nu) & (tau_nu < 0.001)
    negative = (min_reliable_tau <= tau_nu) & (tau_nu < -0.001)
    unreliable_less7 = (-7 <= tau_nu) & (tau_nu < min_reliable_tau)
    unreliable_gtr7 = tau_nu < -7
    assert gtr7.sum()+less7.sum()+small.sum()+negative.sum()+unreliable_less7.sum()\
              +unreliable_gtr7.sum() == tau_nu.size
    beta = np.empty_like(tau_nu)
    beta[gtr7] = beta_LVG_sphere_RADEX_gtr7(tau_nu[gtr7])
    beta[less7] = beta_LVG_sphere_RADEX_less7(tau_nu[less7])
    beta[small] = 1.0
    beta[negative] = beta_LVG_sphere_RADEX_less7(tau_nu[negative])
    beta[unreliable_less7] = beta_LVG_sphere_RADEX_less7(np.abs(tau_nu[unreliable_less7]))
    beta[unreliable_gtr7] = beta_LVG_sphere_RADEX_gtr7(np.abs(tau_nu[unreliable_gtr7]))
    return clip_prob(beta)