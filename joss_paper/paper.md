---
title: 'pythonradex: a fast Python re-implementation of RADEX with extended functionality'
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

A common task in astronomical research is to estimate the physical parameters (temperature, mass, density etc.) of a gas by using observed line emission. This often requires a calculation of how the radiation propagates via emission and absorption ("radiative transfer"). In radio and infrared astronomy, the Fortran code `RADEX` [@vanderTak2007] is a popular tool to solve the non-LTE radiative transfer of a uniform medium in a simplified geometry. I present a python implementation of `RADEX`: `pythonradex`. Written in python, it provides an easy and intuitive user interface, **improved performance** as well as additional functionality not included in `RADEX` (continuum effects and overlapping lines).

# Statement of need

Modern **astronomical** facilities such as the Atacama Large Millimeter/submillimeter Array (ALMA) or the James Webb Space Telescope (JWST) are providing a wealth of line emission data at radio and infrared wavelengths. These data are crucial to constrain the physical and chemical properties of various astrophysical environments.

To interpret such data, a radiative transfer calculation is typically used (see @Rybicki1985 for an introduction to radiative transfer). For a given set of input parameters describing the source (temperature, density, geometry, etc.), one calculates the amount of radiation reaching the telescope. The input parameters are then adjusted such that the predicted flux matches the observations.

If the medium is dense enough, local thermodynamic equilibrium (LTE) maybe be assumed. This considerably simplifies the radiative transfer calculation. A non-LTE calculation is considerably more complex and computationally expensive **because the fractional population of the molecular energy levels needs to be solved for numerically [e.g. @vanderTak2007]. Typically, an iterative approach is used: from a first guess of the level populations, the radiation field is calculated. This radiation field is then used to solve for updated level populations. The iterations continue until convergence is reached.**

Various codes are available to solve the radiative transfer. Codes solving the radiative transfer in 3D are used for detailed calculations of sources with well-known geometries. Examples include `RADMC-3D` [@Dullemond2012] and `LIME` [@Brinch2010]. However, a full 3D calculation is often too computationally expensive if a large parameter space needs to be explored, in particular in non-LTE. 1D codes that quickly provide an approximate solution are a commonly used alternative. In this respect, the 1D non-LTE code `RADEX` [@vanderTak2007] has gained considerable popularity: as of November 10, 2025, the paper presenting `RADEX` [@vanderTak2007] has 1431 citations. The Fortan code `RADEX` solves the radiative transfer of a uniform medium using an escape probability formalism.

The python programming language is now very widely used in astronomy. Still, no python version of `RADEX` is available, although some python wrappers (for example `SpectralRadex` [@SpectralRadex] or `ndradex` [@ndradex]) and even a Julia version [`Jadex`, @jadex] exist. Furthermore, `RADEX` cannot take into account the effects of an internal continuum field (typically arising from dust that is mixed with the gas), nor cross-excitation effects arising when transitions overlap in frequency. The `pythonradex` code addresses these concerns.

# Implementation

`pythonradex` is written in python and implements the Accelerated Lambda Iteration (ALI) scheme presented by @Rybicki1992. Like `RADEX`, an escape probability equation is used to calculate the radiation field for a given level population. This allows solving the radiative transfer iteratively. To speed up the convergence, ng-acceleration [@Ng1974] is employed. **To improve performance, ** critical parts of the code are just-in-time compiled using `numba` [@Lam2015].

`pythonradex` supports four geometries: **two static geometries (slab and sphere), and two large-velocity-gradient (LVG) geometries (again slab and sphere). In the LVG approximation, it is assumed that all regions of the source are Doppler shifted with respect to each other due to a velocity gradient. This means that all photons escape the source, unless absorbed locally [i.e. close to the emission location; e.g. @Elitzur1992].**

**Currently, effects of internal continuum and overlapping lines can only be included for the static geometries. Another limitation is that only a single molecule can be considered at the time. Thus, solving the radiative transfer of overlapping lines of different molecules is not supported yet. Also, treating overlapping lines adds considerable computational cost because averages over line profiles need to be calculated.**

**Like `RADEX`, `pythonradex` needs a file in LAMDA-format as input to read the molecular data. Such files can for example be downloaded from the LAMDA [@Schoier2005] or EMAA [@EMAA] databases.**

# Benchmarking

`pythonradex` was benchmarked against `RADEX` for a number of example problems, generally with excellent agreement **(see \autoref{fig:pythonradex_vs_radex} for an example).** To test the treatment of overlapping lines, `pythonradex` was tested against the `MOLPOP-CEP` code [@AsensioRamos2018], again showing good agreement, **as illustrated in \autoref{fig:HCN_spectrum}.**

![**The ratio of CO 2-1 excitation temperature (left column) and optical depth** (right column) computed with `pythonradex` and `RADEX` for a **static sphere**. Each panel shows a parameter space of **H$_2$ density and column density for a fixed kinetic temperature. Values exceeding the colorbar range are shown in black with the corresponding value in white text.**\label{fig:pythonradex_vs_radex}](code/pythonradex_vs_radex.pdf)

![Spectrum of HCN around 177.3 GHz computed with `pythonradex` **and `MOLPOP-CEP` for a static slab. Good agreement is found when treating line overlap. Interestingly, the spectra differ somewhat when ignoring overlap.** The positions and widths of the individual hyperfine components are illustrated by the black dotted lines.\label{fig:HCN_spectrum}](code/HCN_spec.pdf)


# Performance advantage

<!-- Numbers in this section are from performance_comparison.py -->
**Both `pythonradex` and `RADEX` are single-threaded. To compare their performance, we consider the calculation of a grid of models over a parameter space spanning 20 values in each of kinetic temperature, column density and H$_2$ density (i.e. a total of 8000 models). We consider a few different molecules: C (small number of levels and transitions), SO (large number of levels and transitions) as well as CO and HCO$^+$ (intermediate). On a laptop with i7-7700HQ cores running on Ubuntu 22.04, `pythonradex` computed the model grid faster than `RADEX` by factors  of approximately 1.5 (C), 6 (SO), 7 (CO) and 3 (HCO$^+$). Running the same test on the Multi-wavelength Data Analysis System (MDAS) operated by the National Astronomical Observatory of Japan (Rocky Linux 8.9 with AMD EPYC 7543 CPUs) resulted in a even larger performance advantage: `pythonradex` calculated the grid faster by factors of 13 (C), 10 (SO), 13 (CO) and 12 (HCO$^+$)[^1].**

[^1]: **`Jadex` [@jadex] claims a performance advantage of a factor ~110 over `RADEX`.**

# Additional differences between `RADEX` and `pythonradex`

## Output flux

`RADEX` computes line fluxes based on a "background subtracted" intensity given by $(B_\nu(T_\mathrm{ex})-I_\mathrm{bg})(1-e^{-\tau})$, where $B_\nu$ is the Planck function, $T_\mathrm{ex}$ the excitation temperature, $I_\mathrm{bg}$ the external background and $\tau$ the (frequency-dependent) optical depth. This may or may not be the right quantity to be compared to observations (for example, it is not appropriate when considering data from interferometers like ALMA). `pythonradex` does not apply any observational correction, giving the user the flexibility to decide how the computed fluxes are compared to observations.

## Flux for spherical geometry

**To calculate line fluxes, `pythonradex` uses different formulae depending on the geometry (slab or sphere; see the [`pythonradex` documentation](https://pythonradex.readthedocs.io/en/latest/index.html) for more details). On the other hand, `RADEX` always uses the formula for a slab. \autoref{fig:flux_comparison_sphere} illustrates the consequences. For a static sphere, `RADEX` overestimates the flux by a factor 1.5[^2] in the optically thin limit. The optically thin flux can easily be computed directly[^3], confirming that the flux computed by `pythonradex` is correct. In the optically thick case, only the surface of the static sphere is visible, so both codes agree despite using different formulae. On the other hand, for the LVG sphere, the difference is always a factor 1.5 regardless of optical depth. This is a consequence of LVG assumption that photons always escape unless absorbed locally.**

[^2]: **The factor 1.5 simply corresponds to the volume ratio of a "spherical slab" (i.e. a cylinder) to a sphere.**
[^3]: **In units of W/m$^2$, $F_\mathrm{thin} = V_\mathrm{sphere}n_2A_{21}\Delta E \frac{1}{4\pi d^2}$ with $V_\mathrm{sphere}=\frac{4}{3}R^3\pi$ the volume of the sphere, $n$ the number density, $x_2$ the fractional level population of the upper level, $A_{21}$ the Einstein coefficient, $\Delta E$ the energy of the transition and $d$ the distance.**

<!-- This figure comes from  compare_emerging_flux_formula.py -->
![Flux ratio between `RADEX` and `pythonradex` for spherical geometries as a function of optical depth.\label{fig:flux_comparison_sphere}](flux_comparison_spherical_geometries.pdf)


# Dependencies

`pythonradex` depends on the following packages:

* `numpy` [@Harris2020]
* `scipy` [@Virtanen2020]
* `numba` [@Lam2015]

# Acknowledgements

I thank Simon Bruderer for his helpful clarifications about the ALI method, and Andrés Asensio Ramos for helpful discussions about the LVG approximation and the `MOLPOP-CEP` code. **Performance testing was in part carried out on the Multi-wavelength Data Analysis System operated by the Astronomy Data Center (ADC), National Astronomical Observatory of Japan.**

# References
