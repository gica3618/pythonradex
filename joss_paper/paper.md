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

A common task in astronomical research is to estimate the physical parameters (temperature, mass, density etc.) of a gas by using observed line emission. This often requires a radiative transfer calculation, that is, a calculation of how the radiation propagates in the medium via emission and absorption. In radio and infrared astronomy, the Fortran code RADEX [@vanderTak:2007] is a popular tool to solve the non-LTE radiative transfer of a uniform medium in a simplified geometry. I present a python implementation of RADEX: pythonradex. Written in pure python, pythonradex provides an easy and intuitive user interface. Furthermore, pythonradex provides additional functionality not included in RADEX:

1. treatment of dust continuum effects
2. treatment of overlapping lines

# Statement of need

Observations of molecular line emission at radio and infrared wavelengths are crucial to constrain the physical and chemical properties of various astrophysical environments. Modern astronomical facilities such as the Atacama Large Millimeter/submillimeter Array (ALMA) or the James Webb Space Telescope (JWST) are providing a wealth of line emission data that is used in various sub-fields of astrophysics.

To interpret observations of line emission, a radiative transfer calculation is typically used. In such a calculation, the amount of radiation reaching the telescope for a given set of input parameters describing the source (temperature, density, geometry, etc.) is determined. This calculation needs to take into account the emission and absorption of radiation in the medium. For basic discussion of radiative transfer, see for example @Rybicki:1985.

Typically, the input parameters of the radiative transfer calculation are adjusted such that the predicted flux matches the observations. If the medium is dense enough, local thermodynamic equilibrium (LTE) maybe be assumed. This considerably simplifies the calculation because the fractional population of the energy levels of the molecules, which determines emission and absorption, is equal to the Boltzmann distribution. On the other hand, in non-LTE, the fractional level population needs to be calculated explicitly by assuming statistical equilibrium (that is, the rate of de-excitation a level is set equal to the rate of excitation), which adds considerable complexity and computational cost.

Various codes exist to solve the radiative transfer for a range of application scenarios. Codes solving the radiative transfer in 3D are used for detailed calculations of  sources with well-known geometries. Examples include RADMC-3D [@Dullemond:2012] and LIME [@Brinch:2010]. However, a full 3D calculation is often too computationally expensive if a large number of model calculations is needed to explore the parameter space. This applies especially in the case of non-LTE. In such a situation, 1D codes that are able to quickly provide an approximate solution are a common alternative. In this respect, the 1D non-LTE code RADEX [@vanderTak:2007] has gained considerable popularity: as of February 28, 2025, the RADEX paper by @vanderTak:2007 has 1361 citations. RADEX is a code written in Fortran that solves the radiative transfer of a uniform medium using an escape probability formalism. For an input column density, kinetic temperature, line width and collider density, the program outputs excitation temperatures and line fluxes.

However, the Fortran nature of RADEX makes it somewhat difficult to use for the younger generation of astronomers that is more used to scripting with python. Some python wrappers of RADEX exist, for example pyradex [@pyradex] or ndradex [@ndradex]. There is also a Julia version available [@jadex]. However, no version in pure python has been published so far. Furthermore, RADEX only considers an external background radiation field. It is not possible to take into account the effects of an internal continuum field (typically arising from dust that is mixed with the gas). RADEX is also unable to treat cross-excitation effects that arise when the rest frequencies of two transitions are so similar that photons emitted by one transition can be absorbed by the other transitions (the Fortran code MOLPOP-CEP is able to treat overlapping lines, see @AsensioRamos:2018).

The pythonradex code addresses this concerns. It is written in pure python, thus providing a simple and intuitive user interface. Also, pythonradex is able to include the effects of an internal continuum field and of overlapping transitions.

# Implementation

pythonradex implements the Accelerated Lambda Iteration (ALI) scheme presented by @Rybicki1992 in their section 2.3 ("full preconditioning strategy"). Same as RADEX, an escape probability equation (which depends on the geometry chosen by the user) is used to calculate the radiation field for a given level population, allowing to solve the radiative transfer iteratively. To speed up the convergence, ng-acceleration [@Ng:1974] is employed. To make the code faster, critical parts are just-in-time compiled using the numba package [@Lam:2015].

pythonradex allows the user to choose among the following geometries:

* static sphere
* large velocity gradient (LVG) sphere
* static slab
* LVG slab

Note that including effects of internal continuum or overlapping lines is only supported for the static geometries.

# Benchmarking

pythonradex was benchmarked against RADEX for a number of example problems, generally with excellent agreement. To test the treatment of overlapping lines, pythonradex was tested against the MOLPOP-CEP code [@AsensioRamos:2018], again showing good agreement.

# Performance

The code architecture is designed to be optimised for the typical use case of a parameter space exploration. In particular, all calculations that are independent of the parameter grid that is explored are performed before the model grid is calculated, in order to avoid unnecessary repetition of calculations.

On a laptop with four i7-7700HQ cores (eight virtual cores), running a grid of CO models was three times faster with pythonradex compared to RADEX. However, the CPU usage of pythonradex was much higher than RADEX during this test. Still, it demonstrates that for an average user, pythonradex might be faster than RADEX.

# Some differences between RADEX and pythonradex

Besides the additional functionality provided by pythonradex (treatment of internal dust continuum and overlapping transitions), there are a few additional, subtle differences between the two codes.

## Output flux

The line fluxes output by RADEX are "background subtracted". More concretely, RADEX outputs fluxes based on an intensity given by $(B_\nu(T_\mathrm{ex})-I_\mathrm{bg})(1-e^{-\tau_\nu})$, where $B_\nu$ is the Planck function, $I_\mathrm{bg}$ the external background and $\tau_\nu$ the optical depth. The idea is that these fluxes can directly be compared to the observed fluxes. However, whether the adopted formula is correct or not depends on the telescope (for example, it is not correct for an interferometer like ALMA). On the other hand, pythonradex simply outputs the pure line flux without any background subtraction (for a slab geometry, an intensity simply given by $B_\nu(T_\mathrm{ex})(1-e^{-\tau_\nu})$ is used). It is up to the user to decide how to compare these fluxes to observations

## Different flux for spherical geometry

Regardless of the adopted geometry, RADEX uses the formula for a slab to calculate the flux. If a spherical geometry is adopted, this produces an inconsistent flux. This can be demonstrated by considering the optically thin limit where each photon escapes the cloud. The total flux (in [W/m$^2$]) is then simply given by
\begin{equation}
F_\mathrm{thin} = V_\mathrm{sphere}n_2A_{21}\Delta E \frac{1}{4\pi d^2}
\end{equation}
where $V_\mathrm{sphere}=\frac{4}{3}R^3\pi$ is the volume of the sphere, $n$ the constant number density, $x_2$ the fractional level population of the upper level, $A_{21}$ the Einstein coefficient, $\Delta E$ the energy difference between the upper and lower level, and $d$ the distance of the source. pythonradex correctly reproduces this limiting case by using the formula of by @Osterbrock:1974. On the other hand, RADEX overestimates the optically thin flux by a factor 1.5.


A crucial quantity determining the emission and absorption of photons is the fractional level population, that is, the fraction of molecules residing in each energy level. This is because emission and absorption of photons happens via transitions between the different energy levels. The level population of a specific transition is often characterised by the *excitation temperature* $T_\mathrm{ex}$, defined by
\begin{equation}
\frac{n_2}{n_1} = \frac{g_2}{g_1}e^{-\Delta E/(kT_\mathrm{ex})}
\end{equation}
Here, $n_2$ and $n_1$ are the number densities of molecules in the upper and lower level of the transition, respectively, $g_2$ and $g_1$ are the statistical weights, $\Delta E$ is the energy difference between the levels and $k$ is the Boltzmann constant.

In local thermodynamic equilibrium (LTE), the excitation temperature equals the kinetic temperature for all transitions. LTE applies if transitions induced by collisions are frequent enough to thermalise the level population. In other words, the number density of colliders needs to be high enough for LTE to apply. Thus, assuming LTE, the level population is known (for a given kinetic gas temperature), which considerably simplifies the problem.

On the other hand, in a non-LTE situation, the level population needs to be explicitly calculated.


Line emission occurs when a molecule transitions from a higher to a lower energy level, thereby emitting a photon with a wavelength that corresponds to the two levels. This can either happen spontaneously (spontaneous emission) or by interaction with another photon of the same wavelength (stimulated emission). On the other hand, a molecule can also absorb a photon, thereby transitioning from a lower to a higher energy level. Finally, transitions (both excitation and de-excitation) can also occur when the molecule collides with another particle. In this latter case, the energy is exchanged in the form of kinetic energy and no photons are involved.

Clearly, to calculate the amount of emission and absorption occurring in the gas, we need to know the fraction of molecules residing in each energy level (the *fractional level population*). The level population of a specific transition is often characterised by the *excitation temperature* $T_\mathrm{ex}$, defined by
\begin{equation}
\frac{n_2}{n_1} = \frac{g_2}{g_1}e^{-\Delta E/(kT_\mathrm{ex})}
\end{equation}
Here, $n_2$ and $n_1$ are the number densities of molecules in the upper and lower level of the transitions, respectively, $g_2$ and $g_1$ are the statistical weights, $\Delta E$ is the energy difference between the levels and $k$ is the Boltzmann constant.

If, for each transition, $T_\mathrm{ex}$ is equal to the kinetic temperature of the gas, the gas is said to be in *local thermodynamic equilibrium* (LTE). This occurs if collisional de-excitation happens more frequently than spontaneous emission. If LTE applies, the level population is known (for a given kinetic temperature), and solving the radiative transfer becomes straightforward, at least in principle.

On the other hand, if the density of colliders is too low for LTE to apply, the level population is unknown and needs to be calculated. In such a situation, one typically assumes *statistical equilibrium*: for a given level, the rate of excitation equals the rate of de-excitation. This can be expressed with the following equation: 


`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

I would like to thank Simon Bruderer for his helpful clarifications about the ALI method. I also thank Andrés Asensio Ramos for helpful discussions about the LVG geometry and help with the MOLPOP-CEP code that was used to benchmark pythonradex.

# References
