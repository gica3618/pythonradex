# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:22:37 2017

@author: gianni
"""

from pythonradex import escape_probability
import numpy as np
import itertools

esc_prob_uniform_sphere = escape_probability.EscapeProbabilityUniformSphere()
osterbrock_esc_prob_uniform_sphere = escape_probability.UniformSphere()
radex_esc_prob_uniform_sphere = escape_probability.UniformSphereRADEX()
large_tau_nu = 1e4

def test_esc_prob_uniform_sphere():
    assert esc_prob_uniform_sphere.beta(0) == 1
    assert np.all(esc_prob_uniform_sphere.beta(np.zeros(4)) == np.ones(4))
    assert np.isclose(esc_prob_uniform_sphere.beta(large_tau_nu),0,atol=1e-2,rtol=0)
    assert np.allclose(esc_prob_uniform_sphere.beta(np.ones(4)*large_tau_nu),np.zeros(4),
                       atol=1e-2)
    assert np.isclose(esc_prob_uniform_sphere.beta_analytical(esc_prob_uniform_sphere.tau_epsilon),
                      esc_prob_uniform_sphere.beta_Taylor(esc_prob_uniform_sphere.tau_epsilon),
                      rtol=1e-2,atol=0)

def test_osterbrock_and_radex_esc_prob_uniform_sphere():
    array_size = 10
    source_func_value = 20
    tau_nus_0 = [0,np.zeros(array_size)]
    tau_nus_large = [large_tau_nu,np.ones(array_size)*large_tau_nu]
    source_functions_0 = [0,np.zeros(array_size)]
    source_functions_non0 = [source_func_value,np.ones(array_size)*source_func_value]
    all_source_functions = source_functions_0 + source_functions_non0
    for prob in (osterbrock_esc_prob_uniform_sphere,radex_esc_prob_uniform_sphere):
        for tau_nu_0, source_function_0 in itertools.product(tau_nus_0,source_functions_0):
            flux = prob.compute_flux_nu(tau_nu=tau_nu_0,source_function=source_function_0)
            assert np.all(flux == 0)
        for tau_nu_large,source_function in\
                          itertools.product(tau_nus_large,all_source_functions):
            flux_large_tau_nu = prob.compute_flux_nu(
                                   tau_nu=tau_nu_large,source_function=source_function)
            assert np.allclose(flux_large_tau_nu,np.pi*source_function,atol=0)
