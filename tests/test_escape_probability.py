# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:22:37 2017

@author: gianni
"""

from pythonradex import escape_probability
import numpy as np
from scipy import constants


flux1D = escape_probability.Flux1D()
all_fluxes = [flux1D.compute_flux_nu,escape_probability.UniformSphere.compute_flux_nu]
large_tau_nu = np.array((5e2,))
solid_angle = (100*constants.au)**2/(1*constants.parsec)**2

def test_fluxes():
    for flux in all_fluxes:
        assert np.all(flux(tau_nu=np.zeros(5),source_function=1,
                           solid_angle=solid_angle) == 0)
        assert np.all(flux(tau_nu=1,source_function=np.zeros(5),
                           solid_angle=solid_angle) == 0)
        test_source_func = 1
        thick_flux = flux(tau_nu=large_tau_nu,source_function=test_source_func,
                          solid_angle=solid_angle)
        assert np.allclose(thick_flux,test_source_func*solid_angle,rtol=1e-3,atol=0)

def test_flux_uniform_sphere():
    limit_tau_nu = 1e-2
    epsilon_tau_nu = 0.01*limit_tau_nu
    source_function = 1
    flux_Taylor = escape_probability.UniformSphere.compute_flux_nu(
                       tau_nu=np.array((limit_tau_nu-epsilon_tau_nu,)),
                       source_function=source_function,solid_angle=solid_angle)
    flux_analytical = escape_probability.UniformSphere.compute_flux_nu(
                          tau_nu=np.array((limit_tau_nu+epsilon_tau_nu,)),
                          source_function=source_function,solid_angle=solid_angle)
    assert np.isclose(flux_Taylor,flux_analytical,rtol=0.05,atol=0)

def test_flux_LVG_sphere():
    V = 1
    v = np.linspace(-2*V,2*V,100)
    nu0 = 100*constants.giga
    nu = nu0*(1-v/constants.c)
    flux = escape_probability.UniformLVGSphere.compute_flux_nu(
               tau_nu=1,source_function=1,solid_angle=1,nu=nu,nu0=nu0,V=V)
    zero_region = np.abs(v) > V
    assert np.any(zero_region)
    assert np.any(~zero_region)
    assert np.all(flux[~zero_region]>0)
    assert np.all(flux[zero_region]==0)