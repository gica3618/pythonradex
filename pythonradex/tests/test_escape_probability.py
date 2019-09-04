# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:22:37 2017

@author: gianni
"""

from pythonradex import escape_probability
import numpy as np
import itertools


flux0D = escape_probability.Flux0D()
flux_uniform_slab = escape_probability.FluxUniformSlab()
flux_uniform_sphere = escape_probability.FluxUniformSphere()
all_fluxes = [flux0D,flux_uniform_slab,flux_uniform_sphere]
large_tau_nu = 1e4

def test_fluxes():
    for flux in all_fluxes:
        for zero in (0,np.zeros(5)):
            assert np.all(flux.compute_flux_nu(tau_nu=zero,source_function=1) == 0)
            assert np.all(flux.compute_flux_nu(tau_nu=1,source_function=zero) == 0)
        for large in (large_tau_nu,np.ones(5)*large_tau_nu):
            source_function = 1
            f = flux.compute_flux_nu(tau_nu=large,source_function=source_function)
            assert np.allclose(f,np.pi*source_function,rtol=1e-4,atol=0)

def test_flux_uniform_slab():
    tau_values = np.logspace(-1,1,10)
    interp_values = flux_uniform_slab.interpolated_integral_term(tau_values)
    computed_values = [flux_uniform_slab.integral_term(t) for t in tau_values]
    assert np.allclose(interp_values,computed_values,rtol=1e-2,atol=0)

def test_flux_uniform_sphere():
    limit_tau_nu = flux_uniform_sphere.min_tau_nu
    epsilon_tau_nu = 0.01*limit_tau_nu
    source_function = 1
    flux_Taylor = flux_uniform_sphere.compute_flux_nu(tau_nu=limit_tau_nu-epsilon_tau_nu,
                                                      source_function=source_function)
    flux_analytical = flux_uniform_sphere.compute_flux_nu(tau_nu=limit_tau_nu+epsilon_tau_nu,
                                                          source_function=source_function)
    print(flux_Taylor)
    print(flux_analytical)
    assert np.isclose(flux_Taylor,flux_analytical,rtol=0.05,atol=0)

esc_prob_uniform_sphere = escape_probability.EscapeProbabilityUniformSphere()

def test_esc_prob_uniform_sphere():
    assert esc_prob_uniform_sphere.beta(0) == 1
    assert np.all(esc_prob_uniform_sphere.beta(np.zeros(4)) == np.ones(4))
    assert np.isclose(esc_prob_uniform_sphere.beta(large_tau_nu),0,atol=1e-2,rtol=0)
    assert np.allclose(esc_prob_uniform_sphere.beta(np.ones(4)*large_tau_nu),np.zeros(4),
                       atol=1e-2)
    assert np.isclose(esc_prob_uniform_sphere.beta_analytical(esc_prob_uniform_sphere.tau_epsilon),
                      esc_prob_uniform_sphere.beta_Taylor(esc_prob_uniform_sphere.tau_epsilon),
                      rtol=1e-2,atol=0)


uniform_sphere = escape_probability.UniformSphere()
radex_uniform_sphere = escape_probability.UniformSphereRADEX()
uniform_slab = escape_probability.UniformSlab()
radex_uniform_slab = escape_probability.UniformSlabRADEX()

taylor_gemoetries = [uniform_sphere,radex_uniform_sphere,radex_uniform_slab]

def test_taylor_geometries():
    limit_tau_nu = escape_probability.TaylorEscapeProbability.tau_epsilon
    epsilon_tau_nu = 0.01*limit_tau_nu
    for geo in taylor_gemoetries:
        prob_Taylor = geo.beta_Taylor(limit_tau_nu-epsilon_tau_nu)
        prob_analytical = geo.beta_analytical(limit_tau_nu+epsilon_tau_nu)
        prob = geo.beta(limit_tau_nu)
        assert np.allclose([prob,prob_Taylor,prob_analytical],prob,rtol=1e-2,atol=0)


all_geometries = [uniform_sphere,radex_uniform_sphere,uniform_slab,radex_uniform_slab]

def test_geometries():
    array_size = 10
    tau_nus_0 = [0,np.zeros(array_size)]
    tau_nus_large = [large_tau_nu,np.ones(array_size)*large_tau_nu]
    for geo in all_geometries:
        for zero in tau_nus_0:
            assert np.all(geo.beta(zero) == 1)
        for large in tau_nus_large:
            assert np.allclose(geo.beta(large),0,atol=1e-3,rtol=1e-3)