#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:33:35 2024

@author: gianni
"""

from pythonradex import escape_probability_functions as epf
import numpy as np


Taylor_geometries = [{'analytical':epf.beta_analytical_static_sphere,
                      'Taylor':epf.beta_Taylor_static_sphere,
                      'beta':epf.beta_static_sphere},
                     {'analytical':epf.beta_analytical_LVG_slab,
                      'Taylor':epf.beta_Taylor_LVG_slab,
                      'beta':epf.beta_LVG_slab},
                     {'analytical':epf.beta_analytical_LVG_sphere,
                      'Taylor':epf.beta_Taylor_LVG_sphere,
                      'beta':epf.beta_LVG_sphere}]

def test_clipping():
    prob = np.array(((1.0001,1,0.9999,0.5,-0.001,0)))
    expected_clipped_prop = np.array((1,1,0.9999,0.5,0,0))
    clipped_prob = epf.clip_prob(prob=prob)
    assert np.all(clipped_prob==expected_clipped_prop)

def test_identify_tau_regions():
    test_tau = np.array((-2,-1.2,-0.5,0,0.04,0.06,1,10))
    normal_tau_region,small_tau_region,negative_tau_region,\
                unreliable_negative_region = epf.identify_tau_regions(tau_nu=test_tau)
    assert np.all(np.where(normal_tau_region)[0]==np.array((5,6,7)))
    assert np.all(np.where(small_tau_region)[0]==np.array((3,4)))
    assert np.all(np.where(negative_tau_region)[0]==np.array((2)))
    assert np.all(np.where(unreliable_negative_region)[0]==np.array((0,1)))

def test_Taylor_approx():
    #for very small tau, Taylor is needed because analytical formula gives
    #numerical problems
    #for small tau, they should give consistent results
    small_tau = np.array((0.003,0.03,-0.03))
    for geo in Taylor_geometries:
        ana = geo['analytical'](small_tau)
        Tay = geo['Taylor'](small_tau)
        assert np.allclose(a=ana,b=Tay,atol=0,rtol=1e-4)

zero_tau = np.zeros(1)
small_tau = np.array((0.001,))
medium_tau = np.array((1.,))
large_tau = np.array((100.,))
unreliable_tau = np.array((-5,))
Taylor_taus = np.array((-0.04,0.04))
analytical_taus = np.array((-0.2,0.2))
#boundary points between analytical / Taylor / unreliable:
boundary_taus = [-1,-0.05,0.05]
def test_Taylor_betas():
    for geo in Taylor_geometries:
        beta = geo['beta']
        assert beta(zero_tau) == 1
        assert beta(small_tau) >= 0.99
        assert beta(medium_tau) > 0 and beta(medium_tau) < 1
        assert beta(large_tau) < 0.02
        assert beta(unreliable_tau) == beta(np.abs(unreliable_tau))
        assert np.all(beta(Taylor_taus)==epf.clip_prob(geo['Taylor'](Taylor_taus)))
        assert np.all(beta(analytical_taus)
                      ==epf.clip_prob(geo['analytical'](analytical_taus)))
        #test that everything goes well at the boundaries:
        for boundary_value in boundary_taus:
            beta(np.array((boundary_value,)))


def test_integral_term_interpolation():
    below_min_tau = np.array((epf.min_grid_tau/2,))
    above_max_tau = np.array((epf.max_grid_tau*2,))
    assert epf.interpolated_integral_term(below_min_tau) == below_min_tau
    assert epf.interpolated_integral_term(above_max_tau) == 0.5
    min_log_tau = -5.3
    max_log_tau = 4
    tau_values = np.logspace(min_log_tau,max_log_tau,100)
    #make sure the test covers the space outside the grid as well:
    assert np.min(tau_values) < epf.min_grid_tau
    assert np.max(tau_values) > epf.max_grid_tau
    interp_values = epf.interpolated_integral_term(tau_values)
    computed_values = [epf.integral_term_for_static_slab(t) for t in tau_values]
    assert np.allclose(interp_values,computed_values,rtol=1e-2,atol=0)

def test_static_slab_beta():
    assert epf.beta_static_slab(small_tau) >= 0.99
    assert epf.beta_static_slab(medium_tau) > 0 and epf.beta_static_slab(medium_tau) < 1
    assert epf.beta_static_slab(large_tau) < 0.02
    assert epf.beta_static_slab(unreliable_tau) == 1

def test_LVG_sphere_RADEX_beta():
    assert epf.beta_LVG_sphere_RADEX(small_tau) >= 0.99
    assert epf.beta_LVG_sphere_RADEX(medium_tau) > 0\
                     and epf.beta_LVG_sphere_RADEX(medium_tau) < 1
    assert epf.beta_LVG_sphere_RADEX(large_tau) < 0.02
    test_tau_covering_all = np.linspace(-8,8,1000)
    assert np.any(np.abs(test_tau_covering_all)<0.01)
    #test if the function does not throw an error, i.e. the selection of the
    #different ranges works
    epf.beta_LVG_sphere_RADEX(test_tau_covering_all)
    #test the different range selections explicitly:
    grt7 = np.array((7,7.1,10))
    assert np.all(epf.beta_LVG_sphere_RADEX(grt7) == epf.beta_LVG_sphere_RADEX_gtr7(grt7))
    positive_less7 = np.array((0.01,6.5))
    assert np.all(epf.beta_LVG_sphere_RADEX(positive_less7)
                               == epf.beta_LVG_sphere_RADEX_less7(positive_less7))
    small = np.array((-0.01,-0.009,0,0.0008))
    assert np.all(epf.beta_LVG_sphere_RADEX(small)==1)
    negative = np.array((-1,-0.5,-0.11))
    assert np.all(epf.beta_LVG_sphere_RADEX(negative)
                  ==epf.clip_prob(epf.beta_LVG_sphere_RADEX_less7(negative)))
    unreliable_less7 = np.array((-7,-5.5,-1.001))
    assert np.all(epf.beta_LVG_sphere_RADEX(unreliable_less7)
                  ==epf.beta_LVG_sphere_RADEX_less7(np.abs(unreliable_less7)))
    unreliable_gtr7 = np.array((-10,-7.001))
    assert np.all(epf.beta_LVG_sphere_RADEX(unreliable_gtr7)
                  ==epf.beta_LVG_sphere_RADEX_gtr7(np.abs(unreliable_gtr7)))