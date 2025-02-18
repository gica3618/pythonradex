#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:32:03 2024

@author: gianni
"""

#TODO remove unnecessary nb.jit stuff

import numba as nb
from scipy import constants
from pythonradex import helpers
import numpy as np
import warnings


class FluxCalculator():

    tau_peak_fraction = 1e-2
    nu_per_FHWM = 20
    n_nu_elements = {'regular':15,'LVG sphere':51}

    def __init__(self,emitting_molecule,level_population,geometry_name,
                 V_LVG_sphere,compute_flux_nu,tau_nu0_individual_transitions,
                 tau_dust,S_dust):
        self.emitting_molecule = emitting_molecule
        self.level_population = level_population
        self.geometry_name = geometry_name
        self.V_LVG_sphere = V_LVG_sphere
        self.Tex = self.emitting_molecule.get_Tex(level_population=level_population)
        self.compute_flux_nu = compute_flux_nu
        self.tau_nu0_individual_transitions = tau_nu0_individual_transitions
        self.tau_dust = tau_dust
        self.S_dust = S_dust
        self.is_LVG_sphere = self.geometry_name == 'LVG sphere'

    def get_flux_parameters_for_rectangular_flux(self,transitions):
        n_nu_elements = self.n_nu_elements['LVG sphere'] if self.is_LVG_sphere\
                        else self.n_nu_elements['regular']
        assert n_nu_elements%2 == 1,\
                         'need odd number of elements to include nu0 in the array'
        middle_index = n_nu_elements//2
        nu0 = self.emitting_molecule.nu0[transitions]
        width_nu = self.emitting_molecule.width_v/constants.c*nu0
        # a matrix where first index is transition, second index is nu
        nu = np.linspace(nu0-width_nu/2,nu0+width_nu/2,n_nu_elements,axis=-1)
        tau_nu = 1/nu**2
        tau_nu0 = self.tau_nu0_individual_transitions[transitions]
        tau_nu *= tau_nu0[:,None]/tau_nu[:,middle_index][:,None]
        source_function = helpers.B_nu(T=self.Tex[transitions][:,None],nu=nu)
        return nu,tau_nu,source_function

    def fast_line_fluxes_rectangular_without_overlap(self,solid_angle,transitions):
        #note that integrated fluxes only make sense for non-overlapping lines,
        #so this function can only be used for non-overlapping lines
        nu,tau_nu,source_function = self.get_flux_parameters_for_rectangular_flux(
                                          transitions=transitions)
        kwargs = {'tau_nu':tau_nu,'source_function':source_function,
                  'solid_angle':solid_angle}
        if self.is_LVG_sphere:
            kwargs['V'] = self.V_LVG_sphere
        flux_nu = self.compute_flux_nu(**kwargs)
        flux = np.trapezoid(flux_nu,nu,axis=1)
        return flux

    #following function is very similar to
    #fast_line_fluxes_rectangular_regular_without_overlap,
    #but they are difficult to merge because of numba...
    # @staticmethod
    # @nb.jit(nopython=True,cache=True)
    # def fast_line_fluxes_rectangular_LVG_sphere_without_overlap(
    #              solid_angle,transitions,tau_nu0_individual_transitions,Tex,nu0,
    #              compute_flux_nu,width_v,V_LVG_sphere):
    #     obs_line_fluxes = []
    #     #LVG sphere spectrum is not flat, so need more nu elements than for
    #     #rectangular spectrum
    #     n_nu_elements = 51
    #     for i in transitions:
    #         tau_nu0 = tau_nu0_individual_transitions[i]
    #         nu,tau_nu,source_function = get_flux_parameters_for_rectangular_flux(
    #                                     width_v=width_v,nu0=nu0[i],tau_nu0=tau_nu0,
    #                                     Tex=Tex[i],n_nu_elements=n_nu_elements)
    #         flux_nu = compute_flux_nu(tau_nu=tau_nu,
    #                                   source_function=source_function,
    #                                   solid_angle=solid_angle,nu=nu,nu0=nu0[i],
    #                                   V=V_LVG_sphere)
    #         flux = np.trapezoid(flux_nu,nu)
    #         obs_line_fluxes.append(flux)
    #     return np.array(obs_line_fluxes)

    # def fast_line_fluxes_rectangular_without_overlap(self,solid_angle,transitions):
    #     assert self.emitting_molecule.line_profile_type == 'rectangular'
    #     kwargs = {'tau_nu0_individual_transitions':self.tau_nu0_individual_transitions,
    #               'nu0':self.emitting_molecule.nu0,'Tex':self.Tex,
    #               'compute_flux_nu':self.compute_flux_nu,
    #               'width_v':self.emitting_molecule.width_v,'solid_angle':solid_angle,
    #               'transitions':transitions}
    #     if self.is_LVG_sphere:
    #         return self.fast_line_fluxes_rectangular_LVG_sphere_without_overlap(
    #                    **kwargs,V_LVG_sphere=self.V_LVG_sphere)
    #     else:
    #         return self.fast_line_fluxes_rectangular_regular_without_overlap(**kwargs)

    def fast_line_fluxes_Gaussian_without_overlap(self,solid_angle,transitions):
        nu0 = self.emitting_molecule.nu0[transitions]
        FWHM_nu = self.emitting_molecule.width_v/constants.c*nu0
        sigma_nu = helpers.FWHM2sigma(FWHM_nu)
        #for thin emission, I just want to include out to a certain fraction
        #of the peak
        #but for thick emission, the spectrum is saturated, so a fraction of the
        #peak is not useful; in those cases, I need to set an absolute value
        #for the minimum tau to include
        tau_nu0 = self.tau_nu0_individual_transitions[transitions]
        peak_frac = self.tau_peak_fraction*tau_nu0
        min_tau = np.where(peak_frac>0.01,0.01,peak_frac)
        #calculate the distance in freq spac between tau_nu0 and min_tau:
        #(min_tau = tau_nu0*exp(-(nu-nu0)**2/(2*sigma_nu**2))
        Delta_nu = sigma_nu*np.sqrt(-2*np.log(min_tau/tau_nu0))
        n_nu = np.max((2*Delta_nu/FWHM_nu * self.nu_per_FHWM).astype(int))
        #first index transition, second index frequency:
        nu = np.linspace(nu0-Delta_nu,nu0+Delta_nu,n_nu,axis=-1)
        #doing an approximation: in principle, tau has an additional 1/nu**2 dependence,
        #but if Delta_nu is small compared to nu0, that dependence is negligible
        phi_nu_shape = np.exp(-(nu-nu0[:,None])**2/(2*sigma_nu[:,None]**2))
        tau_nu = tau_nu0[:,None]*phi_nu_shape
        source_function = helpers.B_nu(T=self.Tex[:,None][transitions],nu=nu)
        spectrum = self.compute_flux_nu(tau_nu=tau_nu,source_function=source_function,
                                        solid_angle=solid_angle)
        flux = np.trapezoid(spectrum,nu,axis=1)
        return flux

    def all_transitions(self):
        return list(range(self.emitting_molecule.n_rad_transitions))

    def determine_tau_of_overlapping_lines(self,transitions):
        total_tau = []
        for i in transitions:
            tau = 0
            for j in self.emitting_molecule.overlapping_lines[i]:
                tau += self.tau_nu0_individual_transitions[j]
            total_tau.append(tau)
        return np.array(total_tau)

    def fluxes_of_individual_transitions(self,solid_angle,transitions=None):
        max_acceptable_tau = 0.1 #for dust and overlapping lines
        if transitions is None:
            transitions = self.all_transitions()
        if self.emitting_molecule.any_line_has_overlap(transitions):
            tau_overlapping_lines = self.determine_tau_of_overlapping_lines(
                                                        transitions=transitions)
            if np.any(tau_overlapping_lines>max_acceptable_tau):
                raise ValueError('fluxes of individual lines can only be calculated'
                                 +' for non-overlapping lines or thin overlapping lines')
        if np.any(self.tau_dust(self.emitting_molecule.nu0) > max_acceptable_tau):
            raise ValueError('dust is not optically thin, cannot calculate'+
                             ' fluxes of individual lines')
        if self.emitting_molecule.line_profile_type == 'Gaussian':
            fast_flux = self.fast_line_fluxes_Gaussian_without_overlap(
                                solid_angle=solid_angle,transitions=transitions)
        elif self.emitting_molecule.line_profile_type == 'rectangular':
            fast_flux = self.fast_line_fluxes_rectangular_without_overlap(
                                        solid_angle=solid_angle,transitions=transitions)
        else:
            raise ValueError(f'line profile {self.emitting_molecule.line_profile_type} '
                             +'unknown')
        return np.squeeze(fast_flux)

    def set_nu(self,nu):
        self.nu = nu
        self.nu_selected_lines,self.nu_selected_line_indices = self.identify_lines()
        self.tau_dust_nu = self.tau_dust(self.nu)
        self.set_tau_nu_lines()
        self.set_tau_nu_tot()

    def identify_lines(self):
        nu_min,nu_max = np.min(self.nu),np.max(self.nu)
        selected_lines = []
        selected_line_indices = []
        for i,line in enumerate(self.emitting_molecule.rad_transitions):
            if nu_min <= line.nu0 and line.nu0 <= nu_max:
                selected_line_indices.append(i)
                selected_lines.append(line)
        return selected_lines,selected_line_indices

    def construct_tau_nu_individual_line(self,line,nu,tau_nu0):
        #tau is prop to phi_nu/nu**2
        normalisation = line.line_profile.phi_nu0/line.nu0**2 / tau_nu0
        return line.line_profile.phi_nu(nu)/nu**2 / normalisation

    def set_tau_nu_lines(self):
        self.tau_nu_lines = []
        for line_index,line in zip(self.nu_selected_line_indices,self.nu_selected_lines):
            tau = self.construct_tau_nu_individual_line(
                              line=line,nu=self.nu,
                              tau_nu0=self.tau_nu0_individual_transitions[line_index])
            self.tau_nu_lines.append(tau)

    def set_tau_nu_tot(self):
        #note that this function automatically takes overlapping lines into account
        #note also that I cannot use the get_tau_tot_nu method of the
        #emitting_molecule, because that method only considers a single line
        #and its overlaps
        self.tau_nu_tot = np.sum(self.tau_nu_lines,axis=0) + self.tau_dust_nu

    def get_S_tot(self):
        S_nu = np.zeros_like(self.nu)
        for i,line in enumerate(self.nu_selected_lines):
            x1 = self.level_population[line.low.number]
            x2 = self.level_population[line.up.number]
            S_line = line.source_function(x1=x1,x2=x2)
            S_nu += self.tau_nu_lines[i]*S_line
        S_nu += self.S_dust(self.nu)*self.tau_dust_nu
        S_tot = np.where(self.tau_nu_tot==0,0,S_nu/self.tau_nu_tot)
        return S_tot

    def spectrum(self,solid_angle):
        #note that this function gives the right answer also in cases where
        #line overlap treatment is not necessary
        if self.is_LVG_sphere:
            #assumption for LVG sphere: lines do not overlap and there is no dust,
            #so I can just sum them up
            #this also works for overlapping lines in the optically thin regime
            #reason for this special case: the formula to compute the flux for an LVG
            #sphere depends on nu0, which is not well defined if there are several
            #lines
            if self.emitting_molecule.any_line_has_overlap(
                                line_indices=self.nu_selected_line_indices):
                warnings.warn('LVG sphere geometry: lines are overlapping, '
                              +'output spectrum will only be correct if all lines'
                              +' are optically thin!')
            for dust_func in (self.S_dust,self.tau_dust):
                assert np.all(dust_func(self.nu)==0),'LVG does not support dust'
            spectrum = np.zeros_like(self.nu)
            for line_index,line,tau_nu_line in zip(self.nu_selected_line_indices,
                                                   self.nu_selected_lines,
                                                   self.tau_nu_lines):
                LVG_sphere_kwargs = {'nu':self.nu,'nu0':line.nu0,'V':self.V_LVG_sphere}
                source_function = helpers.B_nu(T=self.Tex[line_index],nu=self.nu)
                spectrum += self.compute_flux_nu(
                                    tau_nu=tau_nu_line,source_function=source_function,
                                    solid_angle=solid_angle,**LVG_sphere_kwargs)
        else:
            source_function = self.get_S_tot()
            spectrum = self.compute_flux_nu(
                               tau_nu=self.tau_nu_tot,source_function=source_function,
                               solid_angle=solid_angle)
        return spectrum