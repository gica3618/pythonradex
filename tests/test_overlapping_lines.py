#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:07:00 2024

@author: gianni
"""

from pythonradex import radiative_transfer,helpers
import os
from scipy import constants
import numpy as np
import pytest
import itertools


class Test_Overlapping():

    here = os.path.dirname(os.path.abspath(__file__))
    #transitions 8 and 10 of CN are separated by ~650 km/s
    datafilepath = os.path.join(here,'LAMDA_files/cn.dat')
    Tkin = 300
    collider_densities = {'LTE':{'e':1e10/constants.centi**3},
                          'non-LTE':{'e':1e1/constants.centi**3}}
    line_profile_types = ['Gaussian','rectangular']
    N = {'thin':1e14/constants.centi**2,
         #'intermediate':1e16/constants.centi**2,
         'thick':1e21/constants.centi**2}
    solid_angle = 1
    geometries = ('static sphere','static slab')

    def generate_cloud(self,N,line_profile_type,treat_line_overlap,coll_dens,
                       geometry):
        src = radiative_transfer.Source(
                              datafilepath=self.datafilepath,geometry=geometry,
                              line_profile_type=line_profile_type,
                              width_v=1000*constants.kilo,
                              use_Ng_acceleration=True,
                              treat_line_overlap=treat_line_overlap)
        src.update_parameters(ext_background=0,N=N,Tkin=self.Tkin,
                              collider_densities=coll_dens,T_dust=0,tau_dust=0)
        self.check_overlapping(src)
        src.solve_radiative_transfer()
        return src
    
    @staticmethod
    def check_overlapping(source):
        assert source.emitting_molecule.overlapping_lines[8] == [9,10]
        assert source.emitting_molecule.overlapping_lines[9] == [8,10]
        assert source.emitting_molecule.overlapping_lines[10] == [8,9]

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    def test_optically_thin(self):
        #overlaps should not play a role because all photons escape anyway
        level_pops = []
        for treat_line_overlap,lp,geo\
                  in itertools.product((True,False),self.line_profile_types,
                                        self.geometries):
            source = self.generate_cloud(N=self.N['thin'],line_profile_type=lp,
                                        treat_line_overlap=treat_line_overlap,
                                        coll_dens=self.collider_densities['non-LTE'],
                                        geometry=geo)
            assert np.all(source.tau_nu0_individual_transitions[:3] < 1e-2)
            #make sure we are in non-LTE:
            LTE_level_pop = source.emitting_molecule.LTE_level_pop(T=self.Tkin)
            assert not np.allclose(source.level_pop,LTE_level_pop,rtol=0,atol=1e-2)
        for level_pop in level_pops:
            assert np.allclose(level_pops[0],level_pop,atol=0,rtol=1e-2)
    
    def test_LTE(self):
        #thin or thick, for high collider density I expect LTE
        for lp,(ID,Nvalue),geo in\
                  itertools.product(self.line_profile_types,self.N.items(),
                                    self.geometries):
            source = self.generate_cloud(N=Nvalue,line_profile_type=lp,
                                        treat_line_overlap=True,
                                        coll_dens=self.collider_densities['LTE'],
                                        geometry=geo)
            if ID == 'thin':
                assert np.all(source.tau_nu0_individual_transitions[:3] < 1e-2)
            elif ID == 'thick':
                assert np.all(source.tau_nu0_individual_transitions[:3] > 10)
            else:
                raise ValueError
            LTE_level_pop = source.emitting_molecule.LTE_level_pop(T=self.Tkin)
            assert np.allclose(source.level_pop,LTE_level_pop,atol=0,rtol=1e-2)

    @staticmethod
    def generate_nu_for_spectrum(source):
        #cover transitions 8,9 and 10
        nu0 = source.emitting_molecule.rad_transitions[9].nu0
        width_nu = 1500*constants.kilo/constants.c*nu0
        nu = np.linspace(nu0-width_nu/2,nu0+width_nu/2,2000)
        min_nu,max_nu = np.min(nu),np.max(nu)
        for line in source.emitting_molecule.rad_transitions[8:11]:
            assert min_nu < line.nu0 < max_nu
        return nu

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    @pytest.mark.filterwarnings("ignore:lines are overlapping, spectrum")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_spectra_thin(self):
        for lp,geo in itertools.product(self.line_profile_types,self.geometries):
            spectra = []
            for treat_line_overlap in (True,False):
                source = self.generate_cloud(N=self.N['thin'],line_profile_type=lp,
                                            treat_line_overlap=treat_line_overlap,
                                            coll_dens=self.collider_densities['non-LTE'],
                                            geometry=geo)
                assert np.all(source.tau_nu0_individual_transitions[:3] < 1e-2)
                #make sure we are in non-LTE:
                LTE_level_pop = source.emitting_molecule.LTE_level_pop(T=self.Tkin)
                assert not np.allclose(source.level_pop,LTE_level_pop,rtol=0,atol=1e-2)
                nu = self.generate_nu_for_spectrum(source=source)
                spectra.append(source.spectrum(solid_angle=self.solid_angle,nu=nu))
            assert np.allclose(*spectra,atol=0,rtol=3e-2)

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    @pytest.mark.filterwarnings("ignore:lines are overlapping")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_spectra_thick_LTE(self):
        #since the source functions will all be B_nu(Tkin), the spectrum should
        #be a black body whether line overlap is treated or not
        for treat_line_overlap,geo in itertools.product((True,False),self.geometries):
            source = self.generate_cloud(N=self.N['thick'],line_profile_type='rectangular',
                                        treat_line_overlap=treat_line_overlap,
                                        coll_dens=self.collider_densities['LTE'],
                                        geometry=geo)
            nu = self.generate_nu_for_spectrum(source=source)
            spectrum = source.spectrum(solid_angle=self.solid_angle,nu=nu)
            black_body = helpers.B_nu(nu=nu,T=self.Tkin)
            bb_flux = black_body*self.solid_angle
            overlapping_lines = source.emitting_molecule.rad_transitions[8:11]
            assert source.emitting_molecule.line_profile_type == 'rectangular'
            summed_phi_nu = np.zeros_like(nu)
            for line in overlapping_lines:
                summed_phi_nu += line.line_profile.phi_nu(nu)
            expected_spectrum = np.where(summed_phi_nu>0,bb_flux,0)
            assert np.allclose(spectrum,expected_spectrum,atol=0,rtol=5e-2)