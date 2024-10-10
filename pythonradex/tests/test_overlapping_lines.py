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
    N = {'thin':1e14/constants.centi**2,'intermediate':1e16/constants.centi**2,
         'thick':1e21/constants.centi**2}
    solid_angle = 1
    geometries = ('uniform sphere','uniform slab')
    iteration_modes = ('LI','ALI')

    def generate_cloud(self,N,line_profile_type,treat_line_overlap,coll_dens,
                       geometry,iteration_mode):
        cld = radiative_transfer.Cloud(
                              datafilepath=self.datafilepath,geometry=geometry,
                              line_profile_type=line_profile_type,width_v=1000*constants.kilo,
                              iteration_mode=iteration_mode,use_NG_acceleration=True,
                              average_over_line_profile=True,
                              treat_line_overlap=treat_line_overlap)
        cld.set_parameters(ext_background=helpers.zero_background,N=N,Tkin=self.Tkin,
                           collider_densities=coll_dens)
        self.check_overlapping(cld)
        cld.solve_radiative_transfer()
        return cld
    
    @staticmethod
    def check_overlapping(cloud):
        assert cloud.emitting_molecule.overlapping_lines[8] == [9,10]
        assert cloud.emitting_molecule.overlapping_lines[9] == [8,10]
        assert cloud.emitting_molecule.overlapping_lines[10] == [8,9]

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    def test_optically_thin(self):
        #overlaps should not play a role because all photons escape anyway
        level_pops = []
        for treat_line_overlap,lp,geo,iter_mode\
                  in itertools.product((True,False),self.line_profile_types,
                                        self.geometries,self.iteration_modes):
            cloud = self.generate_cloud(N=self.N['thin'],line_profile_type=lp,
                                        treat_line_overlap=treat_line_overlap,
                                        coll_dens=self.collider_densities['non-LTE'],
                                        geometry=geo,iteration_mode=iter_mode)
            assert np.all(cloud.tau_nu0[:3] < 1e-2)
            #make sure we are in non-LTE:
            LTE_level_pop = cloud.emitting_molecule.LTE_level_pop(T=self.Tkin)
            assert not np.allclose(cloud.level_pop,LTE_level_pop,rtol=0,atol=1e-2)
        for level_pop in level_pops:
            assert np.allclose(level_pops[0],level_pop,atol=0,rtol=1e-2)
    
    def test_LTE(self):
        #thin or thick, for high collider density I expect LTE
        for lp,(ID,Nvalue),geo,iter_mode in\
                  itertools.product(self.line_profile_types,self.N.items(),
                                    self.geometries,self.iteration_modes):
            cloud = self.generate_cloud(N=Nvalue,line_profile_type=lp,
                                        treat_line_overlap=True,
                                        coll_dens=self.collider_densities['LTE'],
                                        geometry=geo,iteration_mode=iter_mode)
            if ID == 'thin':
                assert np.all(cloud.tau_nu0[:3] < 1e-2)
            elif ID == 'thick':
                assert np.all(cloud.tau_nu0[:3] > 10)
            else:
                raise ValueError
            LTE_level_pop = cloud.emitting_molecule.LTE_level_pop(T=self.Tkin)
            assert np.allclose(cloud.level_pop,LTE_level_pop,atol=0,rtol=1e-2)

    @staticmethod
    def generate_nu_for_spectrum(cloud):
        #cover transitions 8,9 and 10
        nu0 = cloud.emitting_molecule.rad_transitions[9].nu0
        width_nu = 1500*constants.kilo/constants.c*nu0
        nu = np.linspace(nu0-width_nu/2,nu0+width_nu/2,2000)
        min_nu,max_nu = np.min(nu),np.max(nu)
        for line in cloud.emitting_molecule.rad_transitions[8:11]:
            assert min_nu < line.nu0 < max_nu
        return nu

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    @pytest.mark.filterwarnings("ignore:lines are overlapping, spectrum")
    def test_spectra_thin(self):
        for lp,geo,iter_mode in\
                 itertools.product(self.line_profile_types,self.geometries,
                                   self.iteration_modes):
            spectra = []
            for treat_line_overlap in (True,False):
                cloud = self.generate_cloud(N=self.N['thin'],line_profile_type=lp,
                                            treat_line_overlap=treat_line_overlap,
                                            coll_dens=self.collider_densities['non-LTE'],
                                            geometry=geo,iteration_mode=iter_mode)
                assert np.all(cloud.tau_nu0[:3] < 1e-2)
                #make sure we are in non-LTE:
                LTE_level_pop = cloud.emitting_molecule.LTE_level_pop(T=self.Tkin)
                assert not np.allclose(cloud.level_pop,LTE_level_pop,rtol=0,atol=1e-2)
                nu = self.generate_nu_for_spectrum(cloud=cloud)
                spectra.append(cloud.spectrum(solid_angle=self.solid_angle,nu=nu))
            assert np.allclose(*spectra,atol=0,rtol=3e-2)

    @pytest.mark.filterwarnings("ignore:some lines are overlapping")
    @pytest.mark.filterwarnings("ignore:lines are overlapping")
    def test_spectra_thick_LTE(self):
        #since the source functions will all be B_nu(Tkin) even if line overlap
        #is not treated, the spectrum should be a black body whether line overlap
        #is treated or not
        for treat_line_overlap,geo,iter_mode in\
                itertools.product((True,False),self.geometries,self.iteration_modes):
            cloud = self.generate_cloud(N=self.N['thick'],line_profile_type='rectangular',
                                        treat_line_overlap=treat_line_overlap,
                                        coll_dens=self.collider_densities['LTE'],
                                        geometry=geo,iteration_mode=iter_mode)
            nu = self.generate_nu_for_spectrum(cloud=cloud)
            spectrum = cloud.spectrum(solid_angle=self.solid_angle,nu=nu)
            black_body = helpers.B_nu(nu=nu,T=self.Tkin)
            bb_flux = black_body*self.solid_angle
            overlapping_lines = cloud.emitting_molecule.rad_transitions[8:11]
            # if treat_line_overlap:
            assert cloud.emitting_molecule.line_profile_type == 'rectangular'
            summed_phi_nu = np.zeros_like(nu)
            for line in overlapping_lines:
                summed_phi_nu += line.line_profile.phi_nu(nu)
            expected_spectrum = np.where(summed_phi_nu>0,bb_flux,0)
            # else:
            #     expected_spectrum = np.zeros_like(nu)
            #     for line in overlapping_lines:
            #         expected_spectrum += np.where(line.line_profile.phi_nu(nu)>0,
            #                                       bb_flux,0)
            assert np.allclose(spectrum,expected_spectrum,atol=0,rtol=5e-2)

    def test_LI_vs_ALI(self):
        #LI and ALI should give similar results
        for geo,lp,(Ntype,N),(coll_dens_type,coll_dense) in\
                    itertools.product(self.geometries,self.line_profile_types,
                                      self.N.items(),
                                      self.collider_densities.items()):
            if Ntype == 'thick' and coll_dens_type == 'non-LTE':
                continue
            level_pops = []
            for iter_mode in self.iteration_modes:
                cloud = self.generate_cloud(N=N,line_profile_type=lp,
                                            treat_line_overlap=True,
                                            coll_dens=coll_dense,geometry=geo,
                                            iteration_mode=iter_mode)
                level_pops.append(cloud.level_pop)
                if Ntype == 'intermediate':
                    print(np.max(cloud.tau_nu0))
            assert np.allclose(*level_pops,atol=0,rtol=1e-2)