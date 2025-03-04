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

date: 4 March 2025
bibliography: paper.bib

---

# Summary

A common task in astronomical research is to estimate the physical parameters (temperature, mass, density etc.) of a gas by using observed line emission. This often requires a calculation of how the radiation propagates via emission and absorption ("radiative transfer"). In radio and infrared astronomy, the Fortran code RADEX [@vanderTak2007] is a popular tool to solve the non-LTE radiative transfer of a uniform medium in a simplified geometry. I present a python implementation of RADEX: pythonradex. Written in pure python, it provides an easy and intuitive user interface as well as additional functionality not included in RADEX (continuum effects and overlapping lines).

# Statement of need

Modern facilities such as the Atacama Large Millimeter/submillimeter Array (ALMA) or the James Webb Space Telescope (JWST) are providing a wealth of line emission data at radio and infrared wavelengths. These data are crucial to constrain the physical and chemical properties of various astrophysical environments.

To interpret such data, a radiative transfer calculation is typically used (see @Rybicki1985 for an introduction to radiative transfer). For a given set of input parameters describing the source (temperature, density, geometry, etc.), one calculates the amount of radiation reaching the telescope. The input parameters are then adjusted such that the predicted flux matches the observations.

If the medium is dense enough, local thermodynamic equilibrium (LTE) maybe be assumed. This considerably simplifies the radiative transfer calculation. A non-LTE calculation is considerably more complex and computationally expensive because the fractional population of the molecular energy levels needs to be calculated explicitly.

Various codes are available to solve the radiative transfer. Codes solving the radiative transfer in 3D are used for detailed calculations of sources with well-known geometries. Examples include RADMC-3D [@Dullemond2012] and LIME [@Brinch2010]. However, a full 3D calculation is often too computationally expensive if a large parameter space needs to be explored, in particular in non-LTE. 1D codes that quickly provide an approximate solution are a commonly used alternative. In this respect, the 1D non-LTE code RADEX [@vanderTak2007] has gained considerable popularity: as of February 28, 2025, the RADEX paper by @vanderTak2007 has 1361 citations. The Fortan code RADEX solves the radiative transfer of a uniform medium using an escape probability formalism.

The python programming language is now very widely used in astronomy. Still, no python version of RADEX is available, although some python wrappers (for example SpectralRadex [@SpectralRadex] or ndradex [@ndradex]) and even a Julia version [@jadex] exist. Furthermore, RADEX cannot take into account the effects of an internal continuum field (typically arising from dust that is mixed with the gas), nor cross-excitation effects arising when transitions overlap in frequency. The pythonradex code addresses these concerns.

# Implementation

pythonradex is written in pure python and implements the Accelerated Lambda Iteration (ALI) scheme presented by @Rybicki1992. Like RADEX, an escape probability equation is used to calculate the radiation field for a given level population. This allows solving the radiative transfer iteratively. To speed up the convergence, ng-acceleration [@Ng1974] is employed. Critical parts of the code are just-in-time compiled using numba [@Lam2015]. pythonradex supports four geometries: static sphere, large velocity gradient (LVG) sphere, static slab and LVG slab. Effects of internal continuum and overlapping lines can be included for the static geometries.

\autoref{fig:HCN_spectrum} illustrates the capability of pythonradex to solve the radiative transfer for overlapping lines. Note that treating overlapping lines adds considerable computational cost because averages over line profiles need to be calculated.

![Spectrum of HCN around 177.3 GHz computed with pythonradex. The blue solid and orange dashed lines show the spectrum calculated with cross-excitation effects turned on and off, respectively. The positions and widths of the individual hyperfine transitions are illustrated by the black dotted lines.\label{fig:HCN_spectrum}](HCN_spec.pdf)

# Benchmarking

pythonradex was benchmarked against RADEX for a number of example problems, generally with excellent agreement. To test the treatment of overlapping lines, pythonradex was tested against the MOLPOP-CEP code [@AsensioRamos2018], again showing good agreement.

# Performance

pythonradex is optimised for the use case of a parameter space exploration. On a laptop with four i7-7700HQ cores (eight virtual cores), running a grid of CO models was three times faster with pythonradex compared to RADEX. However, the CPU usage of pythonradex was much higher than RADEX during this test. Still, it demonstrates that pythonradex might be faster than RADEX depending on the setup.

# Additional differences between RADEX and pythonradex

## Output flux

RADEX computes line fluxes based on a "background subtracted" intensity given by $(B_\nu(T_\mathrm{ex})-I_\mathrm{bg})(1-e^{-\tau_\nu})$, where $B_\nu$ is the Planck function, $T_\mathrm{ex}$ the excitation temperature, $I_\mathrm{bg}$ the external background and $\tau_\nu$ the optical depth. This may or may not be the right quantity to be compared to observations. In contrast, pythonradex does not apply any observational correction, thus giving more flexibility to the user.

## Flux for spherical geometry

Regardless of the adopted geometry, RADEX always uses the flux formula for a slab geometry, resulting in inconsistencies. Consider the optically thin limit where the total flux (in [W/m$^2$]) for a sphere is simply given by
\begin{equation}
F_\mathrm{thin} = V_\mathrm{sphere}n_2A_{21}\Delta E \frac{1}{4\pi d^2}
\end{equation}
with $V_\mathrm{sphere}=\frac{4}{3}R^3\pi$ the volume of the sphere, $n$ the number density, $x_2$ the fractional level population of the upper level, $A_{21}$ the Einstein coefficient, $\Delta E$ the energy of the transition and $d$ the distance. pythonradex correctly reproduces this limiting case by using the formula by @Osterbrock1974, while RADEX overestimates the optically thin flux by a factor 1.5.

# Dependencies

pythonradex depends on the following packages:

* numpy [@Harris2020]
* scipy [@Virtanen2020]
* numba [@Lam2015]

# Acknowledgements

I thank Simon Bruderer for his helpful clarifications about the ALI method, and Andrés Asensio Ramos for helpful discussions about the LVG geometry and the MOLPOP-CEP code.

# References