---
title: 'pythonradex: a python implementation of the non-LTE radiative transfer code RADEX with additional functionality'
tags:
  - python
  - astronomy
  - radiative transfer
  - non-LTE
authors:
  - name: Gianni Cataldi
    corresponding: true
    orcid: 0000-0002-2700-9676
    affiliation: 1
affiliations:
 - name: National Astronomical Observatory of Japan
   index: 1
   ror: 052rrw050

date: 27 February 2025
bibliography: paper.bib

---

# Summary

A common task in astronomical research is to estimate the physical parameters (temperature, mass, density etc.) of a gas by using observed line emission. This often requires a calculation of how the radiation propagates via emission and absorption ("radiative transfer"). In radio and infrared astronomy, the Fortran code RADEX [@vanderTak2007] is a popular tool to solve the non-LTE radiative transfer of a uniform medium in a simplified geometry. I present a python implementation of RADEX: pythonradex. Written in pure python, it provides an easy and intuitive user interface as well as additional functionality not included in RADEX (continuum effects and overlapping lines).

# Statement of need

Modern facilities such as the Atacama Large Millimeter/submillimeter Array (ALMA) or the James Webb Space Telescope (JWST) are providing a wealth of line emission data at radio and infrared wavelengths. These data are crucial to constrain the physical and chemical properties of various astrophysical environments.

To interpret such data, a radiative transfer calculation is typically used (see @Rybicki1985 for an introduction to radiative transfer). For a given set of input parameters describing the source (temperature, density, geometry, etc.), one calculates the amount of radiation reaching the telescope. The input parameters are then adjusted such that the predicted flux matches the observations.

If the medium is dense enough, local thermodynamic equilibrium (LTE) maybe be assumed. This considerably simplifies the radiative transfer calculation. On the other hand, a non-LTE calculation is considerably more complex and computationally expensive because the fractional population of the molecular energy levels needs to be calculated explicitly.

Various codes are available to solve the radiative transfer. Codes solving the radiative transfer in 3D are used for detailed calculations of sources with well-known geometries. Examples include RADMC-3D [@Dullemond2012] and LIME [@Brinch2010]. However, a full 3D calculation is often too computationally expensive if a large parameter space needs to be explored, in particular in a non-LTE scenario. 1D codes that quickly provide an approximate solution are a common alternative. In this respect, the 1D non-LTE code RADEX [@vanderTak2007] has gained considerable popularity: as of February 28, 2025, the RADEX paper by @vanderTak2007 has 1361 citations. The Fortan code RADEX solves the radiative transfer of a uniform medium using an escape probability formalism.

Some python wrappers of RADEX exist, for example pyradex [@pyradex] or ndradex [@ndradex]. There is also a Julia version available [@jadex]. However, no python version has been published so far, despite python being very widely used in astronomy. Furthermore, RADEX cannot take into account the effects of an internal continuum field (typically arising from dust that is mixed with the gas), nor cross-excitation effects arising when transitions overlap in frequency. The pythonradex code addresses these concerns. It is written in pure python and includes the effects of an internal continuum field and of overlapping transitions.

# Implementation

pythonradex implements the Accelerated Lambda Iteration (ALI) scheme presented by @Rybicki1992. Like RADEX, an escape probability equation is used to calculate the radiation field for a given level population. This allows solving the radiative transfer iteratively. To speed up the convergence, ng-acceleration [@Ng1974] is employed. To make the code faster, critical parts are just-in-time compiled using the numba package [@Lam2015]. pythonradex supports four geometries (static sphere, large velocity gradient (LVG) sphere, static slab, LVG slab).

# Benchmarking

pythonradex was benchmarked against RADEX for a number of example problems, generally with excellent agreement. To test the treatment of overlapping lines, pythonradex was tested against the MOLPOP-CEP code [@AsensioRamos2018], again showing good agreement.

# Performance

The code architecture is optimised for the typical use case of a parameter space exploration. On a laptop with four i7-7700HQ cores (eight virtual cores), running a grid of CO models was three times faster with pythonradex compared to RADEX. However, the CPU usage of pythonradex was much higher than RADEX during this test. Still, it demonstrates that pythonradex might be faster than RADEX depending on the setup.

# Some differences between RADEX and pythonradex

## Output flux

The line fluxes output by RADEX are "background subtracted". More concretely, RADEX computes the intensity as $(B_\nu(T_\mathrm{ex})-I_\mathrm{bg})(1-e^{-\tau_\nu})$, where $B_\nu$ is the Planck function, $I_\mathrm{bg}$ the external background and $\tau_\nu$ the optical depth. The idea is that these fluxes can directly be compared to observations. However, whether this formula is correct or not depends on the telescope (for example, it is not correct for an interferometer like ALMA). On the other hand, pythonradex simply outputs the pure line flux without any background subtraction (for a slab geometry, the intensity is computed as $B_\nu(T_\mathrm{ex})(1-e^{-\tau_\nu})$). This gives the user more flexibility for the comparison to observations.

## Different flux for spherical geometry

RADEX produces inconsistent fluxes in spherical geometries because it always uses the flux formula for a slab, regardless of the adopted geometry. This can be demonstrated by considering the optically thin limit where the total flux (in [W/m$^2$]) is simply given by
\begin{equation}
F_\mathrm{thin} = V_\mathrm{sphere}n_2A_{21}\Delta E \frac{1}{4\pi d^2}
\end{equation}
with $V_\mathrm{sphere}=\frac{4}{3}R^3\pi$ is the volume of the sphere, $n$ the (constant) number density, $x_2$ the fractional level population of the upper level, $A_{21}$ the Einstein coefficient, $\Delta E$ the energy of the transition and $d$ the distance of the source. pythonradex correctly reproduces this limiting case by using the formula by @Osterbrock1974. RADEX overestimates the optically thin flux by a factor 1.5.

# Dependencies

pythonradex depends on the following three packages:

* numpy [@Harris2020]
* scipy [@Virtanen2020]
* numba [@Lam2015]

# Acknowledgements

I would like to thank Simon Bruderer for his helpful clarifications about the ALI method. I also thank Andrés Asensio Ramos for helpful discussions about the LVG geometry and help with the MOLPOP-CEP code that was used to benchmark pythonradex.

# References