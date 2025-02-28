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

A common task in astronomical research is to use observed line emission to estimate the physical parameters (temperature, mass, density etc.) of a gas. This requires a radiative transfer calculation, that is, a calculation of how the radiation propagates in the medium via emission and absorption. In radio and infrared astronomy, the Fortran code RADEX [@vanderTak:2007] is a popular tool to solve the non-LTE radiative transfer of a uniform medium in a simplified geometry. I present a python implementation of RADEX: pythonradex. Written in pure python, pythonradex provides an easy and intuitive user interface. Furthermore, pythonradex provides additional functionality not included in RADEX:
1. pythonradex is able to treat the effects of a continuum field (i.e. dust)
2. pythonradex is able to correctly treat overlapping lines

# Statement of need

Line emission occurs when a molecule transitions from a higher to a lower energy level, thereby emitting a photon with a wavelength that corresponds to the energy difference between the upper and lower level. This can either happen spontaneously (spontaneous emission) or by interaction with another photon of the same wavelength (stimulated emission). On the other hand, a molecule can also absorb a photon, thereby transitioning from a lower to a higher energy level. Finally, transitions can also occur when the molecule collides with another particle. In this latter case, the energy is exchanged in the form of kinetic energy and no photons are involved.

Clearly, to calculate the amount of emission and absorption occurring in the gas, we need to know the fraction of molecules residing in each energy level (the *fractional level population*). The level population of a specific transition is often characerised by the *excitation* temperature defined by
\begin{equation}
\frac{n_2}{n_1} = \frac{g_2}{g_1}e^{-\Delta E/(kT_\mathrm{ex})}
\end{equation}


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
