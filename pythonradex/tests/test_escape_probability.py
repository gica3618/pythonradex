# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:22:37 2017

@author: gianni
"""

from pythonradex import escape_probability
import numpy as np
from scipy import constants


flux0D = escape_probability.Flux0D()
flux_uniform_sphere = escape_probability.FluxUniformSphere()
all_fluxes = [flux0D,flux_uniform_sphere]
large_tau_nu = 5e2
solid_angle = (100*constants.au)**2/(1*constants.parsec)**2

def test_fluxes():
    for flux in all_fluxes:
        for zero in (0,np.zeros(5)):
            assert np.all(flux.compute_flux_nu(tau_nu=zero,source_function=1,
                                               solid_angle=solid_angle) == 0)
            assert np.all(flux.compute_flux_nu(tau_nu=1,source_function=zero,
                                               solid_angle=solid_angle) == 0)
        for large in (large_tau_nu,np.ones(5)*large_tau_nu):
            source_function = 1
            f = flux.compute_flux_nu(tau_nu=large,source_function=source_function,
                                     solid_angle=solid_angle)
            assert np.allclose(f,source_function*solid_angle,rtol=1e-3,atol=0)

def test_flux_uniform_sphere():
    limit_tau_nu = flux_uniform_sphere.min_tau_nu
    epsilon_tau_nu = 0.01*limit_tau_nu
    source_function = 1
    flux_Taylor = flux_uniform_sphere.compute_flux_nu(tau_nu=limit_tau_nu-epsilon_tau_nu,
                                                      source_function=source_function,
                                                      solid_angle=solid_angle)
    flux_analytical = flux_uniform_sphere.compute_flux_nu(tau_nu=limit_tau_nu+epsilon_tau_nu,
                                                          source_function=source_function,
                                                          solid_angle=solid_angle)
    assert np.isclose(flux_Taylor,flux_analytical,rtol=0.05,atol=0)

def test_esc_prob_uniform_sphere():
    esc_prob = escape_probability.EscapeProbabilityUniformSphere()
    assert esc_prob.beta(0) == 1
    assert np.all(esc_prob.beta(np.zeros(4)) == np.ones(4))
    assert large_tau_nu > esc_prob.large_tau
    assert np.isclose(esc_prob.beta(large_tau_nu),0,atol=1e-2,rtol=0)
    assert np.allclose(esc_prob.beta(np.ones(4)*large_tau_nu),np.zeros(4),
                       atol=1e-2,rtol=0)
    assert np.isclose(esc_prob.beta_analytical(esc_prob.tau_epsilon),
                      esc_prob.beta_Taylor(esc_prob.tau_epsilon),rtol=1e-2,atol=0)
    assert np.isclose(esc_prob.beta_analytical(esc_prob.large_tau),
                      esc_prob.beta_large_tau(esc_prob.large_tau),rtol=0,atol=0.05)
    assert np.isclose(esc_prob.beta(-1e-2),1,atol=0,rtol=1e-2)
    assert np.isclose(esc_prob.beta(-large_tau_nu),1.5/-large_tau_nu,atol=0,rtol=1e-6)

uniform_sphere = escape_probability.UniformSphere()
radex_uniform_sphere = escape_probability.UniformSphereRADEX()
uniform_slab = escape_probability.UniformFaceOnSlab()
radex_uniform_slab = escape_probability.UniformShockSlabRADEX()

def test_uniform_slab_interpolation():
    min_log_tau = -4
    max_log_tau = 4
    tau_values = np.logspace(min_log_tau,max_log_tau,100)
    #make sure the test covers the space outside the grid as well:
    assert np.min(tau_values) < np.min(uniform_slab.tau_grid)
    assert np.max(tau_values) > np.max(uniform_slab.tau_grid)
    interp_values = uniform_slab.interpolated_integral_term(tau_values)
    computed_values = [uniform_slab.integral_term(t) for t in tau_values]
    assert np.allclose(interp_values,computed_values,rtol=1e-2,atol=0)

taylor_gemoetries = [uniform_sphere,radex_uniform_sphere,radex_uniform_slab]

def test_taylor_geometries():
    limit_tau_nu = escape_probability.TaylorEscapeProbability.tau_epsilon
    epsilon_tau_nu = 0.01*limit_tau_nu
    for geo in taylor_gemoetries:
        prob_Taylor = geo.beta_Taylor(limit_tau_nu-epsilon_tau_nu)
        prob_analytical = geo.beta_analytical(limit_tau_nu+epsilon_tau_nu)
        prob = geo.beta(limit_tau_nu)
        assert np.allclose([prob,prob_Taylor,prob_analytical],prob,rtol=1e-2,atol=0)
        prob_large_tau = geo.beta_large_tau(geo.large_tau+epsilon_tau_nu)
        prob_analytical_large_tau = geo.beta_analytical(geo.large_tau-epsilon_tau_nu)
        output_prob_large_tau = geo.beta(geo.large_tau)
        assert np.allclose([prob_large_tau,prob_analytical_large_tau,output_prob_large_tau],
                           output_prob_large_tau,rtol=0,atol=0.05)
        assert geo.beta(-limit_tau_nu+epsilon_tau_nu)\
                     == geo.beta_Taylor(-limit_tau_nu+epsilon_tau_nu)
        assert geo.beta(-large_tau_nu) == geo.beta_large_tau(-large_tau_nu)
        assert geo.beta(-1) == geo.beta_analytical(-1)

all_geometries = [uniform_sphere,radex_uniform_sphere,uniform_slab,radex_uniform_slab]

def test_all_geometries():
    array_size = 10
    tau_nus_0 = [0,np.zeros(array_size)]
    tau_nus_large = [large_tau_nu,np.ones(array_size)*large_tau_nu]
    small_tau_nus = [1e-3,np.ones(array_size)*1e-3]
    small_neg_tau_nus = [-st for st in small_tau_nus]
    for geo in all_geometries:
        for zero in tau_nus_0:
            assert np.all(geo.beta(zero) == 1)
        for large in tau_nus_large:
            assert np.allclose(geo.beta(large),0,atol=1e-2,rtol=0)
        for small in small_tau_nus+small_neg_tau_nus:
            assert np.allclose(geo.beta(small),1,atol=0,rtol=1e-2)