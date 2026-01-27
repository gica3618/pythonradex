# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:31:02 2017

@author: gianni
"""
import numpy as np
import numba as nb
from pythonradex import escape_probability_functions
from scipy import constants


class EscapeProbabilityStaticSphere:
    """Represents the escape probability from a static sphere."""

    def __init__(self):
        self.beta = escape_probability_functions.beta_static_sphere


class StaticSphere(EscapeProbabilityStaticSphere):
    """Represents the escape probability and emerging flux from a static
    spherical medium"""

    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def specific_intensity(tau_nu, source_function):
        """Computes the observed specific intensity in [W/m2/Hz/sr], given
        the optical depth tau_nu and the source function in [W/m2/Hz/sr]."""
        # see old Osterbrock book for this formula, appendix 2
        # the observed flux in [W/m2/Hz] is given by
        # (flux at sphere surface)*4*pi*r**2/(4*pi*d**2) = F_surface*Omega*4/(4*pi)
        # = F_surface*Omega/pi
        # with Omega the solid angle of the sphere (Omega=r^2pi/d^2)
        # to calculate the intensity, I simply divide by Omega. This gives the
        # intensity one would measure if the source is unresolved. Of course,
        # if the source is resolved, then for different locations on the sphere
        # different intensities will be measured. so the intensity calculate here
        # is some kind of mean intensity. In fact, it can be understood as the intensity
        # that would produce the same flux if one would assume that it was independent
        # of solid angle, i.e. flux at sphere surface = 2*pi*integral(I_nu*cos(theta)*sin(theta))dtheta,
        # where cos(theta) is necessary to take into account surface inclination.
        # This evaluates to pi*I_nu if I_nu is assumed to be constant

        # for small tau_nu, the Osterbrock formula becomes numerically unstable,
        # so we use a Taylor expansion
        min_tau_nu = 1e-2
        stable_region = tau_nu > min_tau_nu
        flux_nu = (
            2
            * np.pi
            * source_function
            / tau_nu**2
            * (tau_nu**2 / 2 - 1 + (tau_nu + 1) * np.exp(-tau_nu))
        )
        flux_nu_Taylor = (
            2
            * np.pi
            * source_function
            * (tau_nu / 3 - tau_nu**2 / 8 + tau_nu**3 / 30 - tau_nu**4 / 144)
        )  # from Wolfram Alpha
        flux_nu = np.where(stable_region, flux_nu, flux_nu_Taylor)
        assert np.all(np.isfinite(flux_nu))
        return flux_nu / np.pi


@nb.jit(nopython=True, cache=True)
def exp_tau_factor(tau_nu):
    # for very small tau, the exp can introduce numerical errors, so we
    # take care of that when computing (1-exp(-tau))
    return np.where(tau_nu < 1e-5, tau_nu, 1 - np.exp(-tau_nu))


class Flux1D:
    """Represents the computation of the flux when considering only a single direction"""

    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def specific_intensity(tau_nu, source_function):
        """Computes the observed specific intensity in [W/m2/Hz/sr], given
        the optical depth tau_nu and the source function in [W/m2/Hz/sr]."""
        tau_factor = exp_tau_factor(tau_nu=tau_nu)
        return source_function * tau_factor


class StaticSphereRADEX(EscapeProbabilityStaticSphere, Flux1D):
    """Represents the escape probability from a static sphere, but uses the
    single direction assumption to compute the emerging flux. This is what is
    done in the original RADEX code."""

    # see line 288 and following in io.f of RADEX

    pass


class StaticSlab(Flux1D):
    # Since I assume the source is in the far field, it is ok to calculate the flux
    # with the 1D formula
    """Represents the escape probability and emerging flux from a static
    slab"""

    def __init__(self):
        self.beta = escape_probability_functions.beta_static_slab


class LVGSlab(Flux1D):
    """The escape probability and flux for a large velocity gradient (LVG)
    slab"""

    def __init__(self):
        self.beta = escape_probability_functions.beta_LVG_slab


@nb.jit(nopython=True, cache=True)
def specific_intensity_nu0_lvg_sphere(tau_nu, source_function):
    # convenience function useful to compute brightness temperature for LVG sphere
    # in flux.py
    tau_factor = exp_tau_factor(tau_nu=tau_nu)
    return source_function * tau_factor


class LVGSphere:
    """The escape probability and flux for a large velocity gradient (LVG)
    sphere"""

    def __init__(self):
        self.beta = escape_probability_functions.beta_LVG_sphere

    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def specific_intensity(tau_nu, source_function, nu, nu0, V):
        # V is the velocity at the surface of the sphere
        # this formula can be derived by using an approach similar to
        # de Jong et al. (1975, Fig. 3)
        intensity_nu0 = specific_intensity_nu0_lvg_sphere(
            tau_nu=tau_nu, source_function=source_function
        )
        v = constants.c * (1 - nu / nu0)
        # actually, since the line profile is always rectangular for LVG, in principle
        # there is no need to do the np.where, but it's cleaner
        return np.where(np.abs(v) > V, 0, intensity_nu0 * (1 - (v / V) ** 2))


class LVGSphereRADEX(Flux1D):
    """The escape probability and flux for a large velocity gradient (LVG)
    sphere (Expanding sphere in RADEX terminology), using the same
    formula for the escape probability as the RADEX code"""

    def __init__(self):
        self.beta = escape_probability_functions.beta_LVG_sphere_RADEX
