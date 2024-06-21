# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:48:00 2017

@author: gianni
"""
from scipy import constants
import numpy as np
from pythonradex import LAMDA_file,atomic_transition
import numba as nb

#Note: the partition function is just for convenience when using the Molecule class
#directly; it is not used to solve the non-LTE radiative transfer

class Molecule():

    "Represents an atom or molecule"

    def __init__(self,levels,rad_transitions,coll_transitions,partition_function=None):
        '''levels is a list of instances of the Level class
        rad_transitions is a list of instances of the RadiativeTransition class
        coll_transitions is a dictionary with an entry for each collision partner, where
        each entry is a list of instances of the CollisionalTransition class
        partition_function is a user-defined partition function'''
        self.levels = levels
        self.rad_transitions = rad_transitions
        self.coll_transitions = coll_transitions 
        self.n_levels = len(self.levels)
        self.n_rad_transitions = len(self.rad_transitions)
        self.set_partition_function(partition_function=partition_function)

    @classmethod
    def from_LAMDA_datafile(cls,datafilepath,read_frequencies=False,
                            partition_function=None):
        """Alternative constructor using a LAMDA data file"""
        data = LAMDA_file.read(datafilepath=datafilepath,
                               read_frequencies=read_frequencies)
        return cls(levels=data['levels'],rad_transitions=data['radiative transitions'],
                   coll_transitions=data['collisional transitions'],
                   partition_function=partition_function)

    def set_partition_function(self,partition_function):
        if partition_function is None:
            self.Z = self.Z_from_atomic_data
        else:
            self.Z = partition_function

    def Z_from_atomic_data(self,T):
        '''Computes the partition function for a given temperature T. T can
        be a float or an array'''
        T = np.array(T)
        weights = np.array([l.g for l in self.levels])
        energies = np.array([l.E for l in self.levels])
        if T.ndim > 0:
            shape = [self.n_levels,]+[1 for i in range(T.ndim)] #needs to come before T is modified
            T = np.expand_dims(T,axis=0) #insert new axis at first position (axis=0)
            weights = weights.reshape(shape)
            energies = energies.reshape(shape)
        return np.sum(weights*np.exp(-energies/(constants.k*T)),axis=0)

    def LTE_level_pop(self,T):
        '''Computes the level populations in LTE for a given temperature T.
        Axis 0 of the output runs along levels, the other axes (if any)
        correspond to the shape of T'''
        T = np.array(T)
        Z = self.Z(T)
        pops = [l.LTE_level_pop(T=T,Z=Z) for l in self.levels]
        if T.ndim > 0:
            shape = [1,]+list(T.shape)
            return np.concatenate([p.reshape(shape) for p in pops],axis=0)
        else:
            return np.array(pops)

    def get_rad_transition_number(self,transition_name):
        '''Returns the transition number for a given transition name'''
        candidate_numbers = [i for i,line in enumerate(self.rad_transitions) if
                             line.name==transition_name]
        assert len(candidate_numbers) == 1
        return candidate_numbers[0]


class EmittingMolecule(Molecule):

    "Represents an emitting molecule, i.e. a molecule with a specified line profile"
    
    def __init__(self,levels,rad_transitions,coll_transitions,line_profile_cls,
                 width_v,partition_function=None):
        '''levels is a list of instances of the Level class
        rad_transitions is a list of instances of the RadiativeTransition class
        coll_transitions is a dictionary with an entry for each collision partner, where
        each entry is a list of instances of the CollisionalTransition class
        line_profile_cls is the line profile class used to represent the line profile
        width_v is the width of the line in velocityy space in [m/s]
        partition_function is a user-defined partition function'''
        Molecule.__init__(self,levels=levels,rad_transitions=rad_transitions,
                          coll_transitions=coll_transitions,
                          partition_function=partition_function)
        #convert radiative transitions to emission lines (but keep the same attribute name)
        self.rad_transitions = [atomic_transition.EmissionLine.from_radiative_transition(
                               radiative_transition=rad_trans,
                               line_profile_cls=line_profile_cls,width_v=width_v)
                               for rad_trans in self.rad_transitions]
        #collecting parameters, needed for numba-compiled functions:
        self.A21_lines = np.array([line.A21 for line in self.rad_transitions])
        self.nu0_lines = np.array([line.nu0 for line in self.rad_transitions])
        self.phi_nu0_lines = np.array([line.line_profile.phi_nu(line.nu0) for line
                                       in self.rad_transitions])
        self.gup_lines = np.array([line.up.g for line in self.rad_transitions])
        self.glow_lines = np.array([line.low.g for line in self.rad_transitions])
        self.low_number_lines = np.array([line.low.number for line in
                                          self.rad_transitions])
        self.up_number_lines = np.array([line.up.number for line in
                                         self.rad_transitions])
        self.Delta_E_lines = np.array([line.Delta_E for line in self.rad_transitions])
        self.ordered_colliders = sorted(self.coll_transitions.keys())
        self.coll_trans_low_up_number = nb.typed.List([])
        self.coll_Tkin_data = nb.typed.List([])
        self.coll_K21_data = nb.typed.List([])
        self.coll_gups = nb.typed.List([])
        self.coll_glows = nb.typed.List([])
        self.coll_DeltaEs = nb.typed.List([])
        for collider in self.ordered_colliders:
            transitions = self.coll_transitions[collider]
            n_low_up = [[trans.low.number,trans.up.number] for trans in transitions]
            self.coll_trans_low_up_number.append(np.array(n_low_up))
            Tkin = [trans.Tkin_data for trans in transitions]
            self.coll_Tkin_data.append(np.array(Tkin))
            K21 = [trans.K21_data for trans in transitions]
            self.coll_K21_data.append(np.array(K21))
            gup = [trans.up.g for trans in transitions]
            glow = [trans.low.g for trans in transitions]
            self.coll_gups.append(np.array(gup))
            self.coll_glows.append(np.array(glow))
            DeltaE = [trans.Delta_E for trans in transitions]
            self.coll_DeltaEs.append(np.array(DeltaE))

    @classmethod
    def from_LAMDA_datafile(cls,datafilepath,line_profile_cls,width_v,
                            read_frequencies=False,partition_function=None):
        """Alternative constructor using a LAMDA data file"""
        data = LAMDA_file.read(datafilepath=datafilepath,
                               read_frequencies=read_frequencies)
        return cls(levels=data['levels'],rad_transitions=data['radiative transitions'],
                   coll_transitions=data['collisional transitions'],
                   line_profile_cls=line_profile_cls,width_v=width_v,
                   partition_function=partition_function)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_tau_nu0(N,level_population,low_number_lines,up_number_lines,
                     A21_lines,phi_nu0_lines,gup_lines,glow_lines,nu0_lines):
        n_lines = low_number_lines.size
        tau_nu0 = np.empty(n_lines)
        for i in range(n_lines):
            N1 = level_population[low_number_lines[i]]*N
            N2 = level_population[up_number_lines[i]]*N
            tau_nu0[i] = atomic_transition.fast_tau_nu(
                             A21=A21_lines[i],phi_nu=phi_nu0_lines[i],
                             g_up=gup_lines[i],g_low=glow_lines[i],N1=N1,N2=N2,
                             nu=nu0_lines[i])
        return tau_nu0
    
    def get_tau_nu0(self,N,level_population):
        '''For a given total column density N and level population,
        compute the optical depth at line center for all radiative transitions'''
        return self.fast_tau_nu0(
                N=N,level_population=level_population,
                low_number_lines=self.low_number_lines,
                up_number_lines=self.up_number_lines,A21_lines=self.A21_lines,
                phi_nu0_lines=self.phi_nu0_lines,gup_lines=self.gup_lines,
                glow_lines=self.glow_lines,nu0_lines=self.nu0_lines)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_Tex(level_population,low_number_lines,up_number_lines,Delta_E_lines,
                 gup_lines,glow_lines):
        n_lines = low_number_lines.size
        Tex = np.empty(n_lines)
        for i in range(n_lines):
            Tex[i] = atomic_transition.fast_Tex(
                       Delta_E=Delta_E_lines[i],g_up=gup_lines[i],
                       g_low=glow_lines[i],x1=level_population[low_number_lines[i]],
                       x2=level_population[up_number_lines[i]])
        return Tex

    def get_Tex(self,level_population):
        '''For a given level population, compute the excitation temperature
        for all radiative transitions'''
        return self.fast_Tex(
                 level_population=level_population,low_number_lines=self.low_number_lines,
                 up_number_lines=self.up_number_lines,Delta_E_lines=self.Delta_E_lines,
                 gup_lines=self.gup_lines,glow_lines=self.glow_lines)