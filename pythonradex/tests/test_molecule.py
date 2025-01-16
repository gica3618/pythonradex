# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:35:53 2017

@author: gianni
"""
import os
from pythonradex import molecule,atomic_transition,LAMDA_file,helpers
import numpy as np
from scipy import constants

here = os.path.dirname(os.path.abspath(__file__))
lamda_filepath = os.path.join(here,'LAMDA_files/co.dat')

test_data = LAMDA_file.read(datafilepath=lamda_filepath,read_frequencies=False)
line_profile_type = 'rectangular'
width_v = 1*constants.kilo

molecule_lamda = molecule.Molecule(
                                datafilepath=lamda_filepath,read_frequencies=False)
molecule_lamda_frequencies = molecule.Molecule(
                                datafilepath=lamda_filepath,read_frequencies=True)

emitting_molecule_lamda = molecule.EmittingMolecule(
                           datafilepath=lamda_filepath,
                           line_profile_type=line_profile_type,width_v=width_v,
                           read_frequencies=False)
emitting_molecule_lamda_frequencies = molecule.EmittingMolecule(
                                        datafilepath=lamda_filepath,
                                        line_profile_type=line_profile_type,
                                        width_v=width_v,read_frequencies=True)

test_molecules_no_freq = [molecule_lamda,emitting_molecule_lamda]
test_molecules_freq = [molecule_lamda_frequencies,emitting_molecule_lamda_frequencies]
test_molecules = test_molecules_no_freq+test_molecules_freq
emitting_molecules = [emitting_molecule_lamda,
                      emitting_molecule_lamda_frequencies]

def test_Molecule_levels():
    for i,level in enumerate(test_data['levels']):
        for mol in test_molecules:
            for attribute in ('g','E','number'):
                assert getattr(level,attribute)\
                                  == getattr(mol.levels[i],attribute)

def test_Molecule_rad_transitions():
    for i,rad_trans in enumerate(test_data['radiative transitions']):
        for mol in test_molecules:
            for attribute in ('Delta_E','A21'):
                assert getattr(rad_trans,attribute)\
                                  == getattr(mol.rad_transitions[i],attribute)
        #following attributes depend on nu0, so change if nu0 is taken from the LAMDA file
        for attribute in ('B12','nu0','B21'):
            for mol in test_molecules_no_freq:
                assert getattr(rad_trans,attribute)\
                             == getattr(mol.rad_transitions[i],attribute)
            for mol in test_molecules_freq:
                assert np.isclose(a=getattr(rad_trans,attribute),
                                  b=getattr(mol.rad_transitions[i],attribute),
                                  atol=0,rtol=1e-4)

def test_Molecule_coll_transitions():
    colliders = sorted(test_data['collisional transitions'].keys())
    for mol in test_molecules:
        assert colliders == sorted(mol.coll_transitions.keys())
    for collider,coll_transitions in test_data['collisional transitions'].items():
        for i,coll_trans in enumerate(coll_transitions):
            for mol in test_molecules:
                for attribute in ('K21_data','Tkin_data'):
                    assert np.all(getattr(coll_trans,attribute)\
                                      == getattr(mol.coll_transitions[collider][i],
                                                 attribute))

# def test_level_transitions():
#     test_mol = molecule.EmittingMolecule(
#                    datafilepath=os.path.join(here,'LAMDA_files/cn.dat'),
#                                             line_profile_type=line_profile_type,
#                                             width_v=width_v,read_frequencies=True)
#     assert test_mol.downward_rad_transitions[0] == []
#     assert test_mol.downward_rad_transitions[1] == [0,]
#     assert test_mol.downward_rad_transitions[3] == [2,3]
#     assert test_mol.level_transitions[0] == [0,1]
#     assert test_mol.level_transitions[1] == [0,3]
#     assert test_mol.level_transitions[2] == [1,2,4]

def test_setting_partition_function():
    test_mol = molecule.Molecule(
                        datafilepath=lamda_filepath,
                        partition_function=lambda x: -10)
    assert test_mol.Z(100) == -10

def test_partition_func():
    T = 50
    for mol in test_molecules:
        Q = 0
        for level in mol.levels:
            Q += level.g*np.exp(-level.E/(constants.k*T))
        assert np.isclose(Q,mol.Z(T),atol=0,rtol=1e-10)

def test_LTE_level_pop():
    T = 30
    for mol in test_molecules:
        LTE_level_pop = mol.LTE_level_pop(T=T)
        assert np.isclose(np.sum(LTE_level_pop),1)
        Q = mol.Z(T=T)
        for i,level in enumerate(mol.levels):
            expected_pop = level.g*np.exp(-level.E/(constants.k*T)) / Q
            assert expected_pop == LTE_level_pop[i]

def test_get_transition_number():
    for mol in test_molecules:
        rad_trans_number = mol.get_rad_transition_number('11-10')
        assert rad_trans_number == 10

rng = np.random.default_rng(seed=0)

def test_tau():
    N_values = np.array((1e12,1e14,1e16,1e18))/constants.centi**2
    for N in N_values:
        for mol in emitting_molecules:
            level_population = rng.random(mol.n_levels)
            level_population /= np.sum(level_population)
            tau_nu0 = mol.get_tau_nu0(N=N,level_population=level_population)
            expected_tau_nu0 = []
            for i,rad_trans in enumerate(mol.rad_transitions):
                n_up = rad_trans.up.number
                N2 = N*level_population[n_up]
                n_low = rad_trans.low.number
                N1 = N*level_population[n_low]
                t = atomic_transition.fast_tau_nu(
                         A21=rad_trans.A21,phi_nu=rad_trans.line_profile.phi_nu(rad_trans.nu0),
                         g_low=rad_trans.low.g,g_up=rad_trans.up.g,N1=N1,N2=N2,nu=rad_trans.nu0)
                expected_tau_nu0.append(t)
            assert np.all(expected_tau_nu0==tau_nu0)

def test_tau_LTE():
    N_values = np.array((1e12,1e14,1e16,1e18))/constants.centi**2
    T = 123
    for N in N_values:
        for mol in emitting_molecules:
            level_population = mol.LTE_level_pop(T=T)
            tau_nu0_LTE = mol.get_tau_nu0_LTE(N=N,T=T)
            expected_tau_nu0 = []
            for i,rad_trans in enumerate(mol.rad_transitions):
                n_up = rad_trans.up.number
                N2 = N*level_population[n_up]
                n_low = rad_trans.low.number
                N1 = N*level_population[n_low]
                t = atomic_transition.fast_tau_nu(
                         A21=rad_trans.A21,phi_nu=rad_trans.line_profile.phi_nu(rad_trans.nu0),
                         g_low=rad_trans.low.g,g_up=rad_trans.up.g,N1=N1,N2=N2,nu=rad_trans.nu0)
                expected_tau_nu0.append(t)
            assert np.all(expected_tau_nu0==tau_nu0_LTE)
    
def test_LTE_Tex():
    T = 123
    for mol in emitting_molecules:
        LTE_level_pop = mol.LTE_level_pop(T=T)
        Tex = mol.get_Tex(level_population=LTE_level_pop)
        assert np.allclose(a=T,b=Tex,atol=0,rtol=1e-10)

def test_get_tau_line_nu():
    N = 1e15/constants.centi**2
    for mol in emitting_molecules:
        level_pop = mol.LTE_level_pop(T=214)
        tau_line_funcs = [mol.get_tau_line_nu(line_index=i,level_population=level_pop,N=N)
                          for i in range(mol.n_rad_transitions)]
        for i,line in enumerate(mol.rad_transitions):
            width_nu = width_v/constants.c*line.nu0
            nu = np.linspace(line.nu0-width_nu,line.nu0+width_nu,100)
            expected_tau = line.tau_nu(N1=N*level_pop[line.low.number],
                                       N2=N*level_pop[line.up.number],nu=nu)
            assert np.all(expected_tau==tau_line_funcs[i](nu))

def test_Tex():
    for mol in emitting_molecules:
        level_population = rng.random(mol.n_levels)
        level_population /= np.sum(level_population)
        Tex = mol.get_Tex(level_population=level_population)
        expected_Tex = []
        for i,rad_trans in enumerate(mol.rad_transitions):
            n_up = rad_trans.up.number
            x2 = level_population[n_up]
            n_low = rad_trans.low.number
            x1 = level_population[n_low]
            tex = atomic_transition.fast_Tex(
                        Delta_E=rad_trans.Delta_E,g_low=rad_trans.low.g,
                        g_up=rad_trans.up.g,x1=x1,x2=x2)
            expected_Tex.append(tex)
        assert np.all(Tex==expected_Tex)

def get_molecule(line_profile_type,width_v,datafilename):
    return molecule.EmittingMolecule(
               datafilepath=os.path.join(here,'LAMDA_files',datafilename),
               line_profile_type=line_profile_type,width_v=width_v)

class TestOverlappingLines():

    line_profile_types = ('rectangular','Gaussian')    

    def get_HCl_molecule(self,line_profile_type,width_v):
        return get_molecule(line_profile_type=line_profile_type,width_v=width_v,
                                 datafilename='hcl.dat')

    def get_CO_molecule(self,line_profile_type,width_v):
        return get_molecule(line_profile_type=line_profile_type,width_v=width_v,
                                 datafilename='co.dat')

    def test_overlapping_lines(self):
        #first three transitions of HCl are separated by ~8 km/s and 6 km/s respectively
        overlapping_3lines = [self.get_HCl_molecule(line_profile_type='rectangular',
                                                    width_v=16*constants.kilo),
                              self.get_HCl_molecule(line_profile_type='Gaussian',
                                                    width_v=10*constants.kilo)
                              ]
        for ol in overlapping_3lines:
            assert ol.overlapping_lines[0] == [1,2]
            assert ol.overlapping_lines[1] == [0,2]
            assert ol.overlapping_lines[2] == [0,1]
        overlapping_2lines = [self.get_HCl_molecule(line_profile_type='rectangular',
                                                    width_v=8.5*constants.kilo),
                              self.get_HCl_molecule(line_profile_type='Gaussian',
                                                    width_v=3.5*constants.kilo)
                              ]
        for ol in overlapping_2lines:
            assert ol.overlapping_lines[0] == [1,]
            assert ol.overlapping_lines[1] == [0,2]
            assert ol.overlapping_lines[2] == [1,]
        #transitions 4-11 are separated by ~11.2 km/s
        overlapping_8lines = [self.get_HCl_molecule(line_profile_type='rectangular',
                                                    width_v=11.5*constants.kilo),
                              self.get_HCl_molecule(line_profile_type='Gaussian',
                                                    width_v=4*constants.kilo)
                              ]
        for ol in overlapping_8lines:
            for i in range(3,11):
                assert ol.overlapping_lines[i] == [index for index in range(3,11)
                                                   if index!=i]
        for line_profile_type in self.line_profile_types:
            CO_molecule = self.get_CO_molecule(line_profile_type=line_profile_type,
                                               width_v=1*constants.kilo)
            for overlap_lines in CO_molecule.overlapping_lines:
                assert overlap_lines == []
            HCl_molecule = self.get_HCl_molecule(line_profile_type=line_profile_type,
                                                 width_v=0.01*constants.kilo)
            assert HCl_molecule.overlapping_lines[0] == []
            assert HCl_molecule.overlapping_lines[11] == []

    def test_any_overlapping(self):
        for line_profile_type in self.line_profile_types:
            HCl_molecule = self.get_HCl_molecule(line_profile_type=line_profile_type,
                                                 width_v=10*constants.kilo)
            assert HCl_molecule.any_line_has_overlap(line_indices=[0,1,2,3,4])
            assert HCl_molecule.any_line_has_overlap(line_indices=[0,])
            assert HCl_molecule.any_line_has_overlap(
                          line_indices=list(range(len(HCl_molecule.rad_transitions))))
            HCl_molecule = self.get_HCl_molecule(line_profile_type=line_profile_type,
                                                 width_v=1*constants.kilo)
            assert not HCl_molecule.any_line_has_overlap(line_indices=[0,1,2])
            CO_molecule = self.get_CO_molecule(line_profile_type=line_profile_type,
                                               width_v=1*constants.kilo)
            assert not CO_molecule.any_line_has_overlap(
                       line_indices=list(range(len(CO_molecule.rad_transitions))))

class TestTotalQuantities():

    CO_molecule = get_molecule(line_profile_type='Gaussian',width_v=1*constants.kilo,
                               datafilename='co.dat')
    HCl_molecule = get_molecule(line_profile_type='Gaussian',width_v=10*constants.kilo,
                                datafilename='hcl.dat')   
    N_CO = 1e15/constants.centi**2
    N_HCl = 1e14/constants.centi**2
    test_tau_dust = 1
    test_S = helpers.B_nu(nu=230*constants.giga,T=100)

    @staticmethod
    def tau_dust_zero(nu):
        return np.zeros_like(nu)

    def tau_dust_nonzero(self,nu):
        return np.ones_like(nu)*self.test_tau_dust

    def tau_dust_iterator(self):
        for tau_d,tau_dust in zip((0,self.test_tau_dust),
                                  (self.tau_dust_zero,self.tau_dust_nonzero)):
            yield tau_d,tau_dust

    def test_no_overlap(self):
        line_index = 3
        line = self.CO_molecule.rad_transitions[line_index]
        width_nu = line.line_profile.width_nu
        nu = np.linspace(line.nu0-3*width_nu,line.nu0+3*width_nu,200)
        level_population = self.CO_molecule.LTE_level_pop(T=23)
        x1 = level_population[line.low.number]
        x2 = level_population[line.up.number]
        tau_line = line.tau_nu(nu=nu,N1=x1*self.N_CO,N2=x2*self.N_CO)
        for tau_d,tau_dust in self.tau_dust_iterator():
            tau_tot = self.CO_molecule.get_tau_tot_nu(
                          line_index=line_index,level_population=level_population,
                          N=self.N_CO,tau_dust=tau_dust)(nu)
            x1 = level_population[line.low.number]
            x2 = level_population[line.up.number]
            expected_tau_tot = line.tau_nu(N1=self.N_CO*x1,N2=self.N_CO*x2,nu=nu)\
                                + tau_d
            assert np.all(tau_tot==expected_tau_tot)

    def test_with_overlap(self):
        line_index = 1
        line = self.HCl_molecule.rad_transitions[line_index]
        assert self.HCl_molecule.overlapping_lines[line_index] == [0,2]
        width_nu = line.line_profile.width_nu
        nu_start = self.HCl_molecule.rad_transitions[0].nu0-3*width_nu
        nu_end = self.HCl_molecule.rad_transitions[2].nu0+3*width_nu
        nu = np.linspace(nu_start,nu_end,500)
        level_population = self.CO_molecule.LTE_level_pop(T=23)

        for tau_d,tau_dust in self.tau_dust_iterator():
            tau_tot = self.HCl_molecule.get_tau_tot_nu(
                          line_index=line_index,level_population=level_population,
                          N=self.N_HCl,tau_dust=tau_dust)(nu)
            expected_tau_tot = np.zeros_like(nu)
            for i in (0,1,2):
                line_i = self.HCl_molecule.rad_transitions[i]
                x1 = level_population[line_i.low.number]
                x2 = level_population[line_i.up.number]
                expected_tau_tot += line_i.tau_nu(N1=self.N_HCl*x1,N2=self.N_HCl*x2,
                                                  nu=nu)
            expected_tau_tot += tau_d
            assert np.all(tau_tot==expected_tau_tot)