# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:22:37 2017

@author: gianni
"""

from pythonradex import escape_probability
import numpy as np
from scipy import constants


flux1D = escape_probability.Flux1D()
all_intensities = {"1D":flux1D.specific_intensity,
                   "static sphere":escape_probability.StaticSphere.specific_intensity,
                   "LVG sphere":escape_probability.LVGSphere.specific_intensity}
large_tau_nu = np.array((5e2,))

def test_intensities():
    for ID,intensity in all_intensities.items():
        if ID == "LVG sphere":
            kwargs = {"nu":100*constants.giga,"V":10*constants.kilo}
            kwargs["nu0"] = kwargs["nu"]
        else:
            kwargs = {}
        assert np.all(intensity(tau_nu=np.zeros(5),source_function=1,**kwargs) == 0)
        assert np.all(intensity(tau_nu=1,source_function=np.zeros(5),**kwargs) == 0)
        test_source_func = 1
        thick_I = intensity(tau_nu=large_tau_nu,source_function=test_source_func,
                            **kwargs)
        assert np.allclose(thick_I,test_source_func,rtol=1e-3,atol=0)

def test_flux_static_sphere():
    limit_tau_nu = 1e-2
    epsilon_tau_nu = 0.01*limit_tau_nu
    source_function = 1
    I_Taylor = escape_probability.StaticSphere.specific_intensity(
                       tau_nu=np.array((limit_tau_nu-epsilon_tau_nu,)),
                       source_function=source_function)
    I_analytical = escape_probability.StaticSphere.specific_intensity(
                          tau_nu=np.array((limit_tau_nu+epsilon_tau_nu,)),
                          source_function=source_function)
    assert np.isclose(I_Taylor,I_analytical,rtol=0.05,atol=0)

def test_flux_LVG_sphere():
    V = 1
    v = np.linspace(-2*V,2*V,100)
    nu0 = 100*constants.giga
    nu = nu0*(1-v/constants.c)
    intensity = escape_probability.LVGSphere.specific_intensity(
               tau_nu=1,source_function=1,nu=nu,nu0=nu0,V=V)
    zero_region = np.abs(v) > V
    assert np.any(zero_region)
    assert np.any(~zero_region)
    assert np.all(intensity[~zero_region]>0)
    assert np.all(intensity[zero_region]==0)