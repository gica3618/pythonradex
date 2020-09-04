# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:48:00 2017

@author: gianni
"""
from scipy import constants
import numpy as np
from pythonradex import LAMDA_file,atomic_transition


class Molecule():

    "Represents an atom or molecule"

    def __init__(self,levels,rad_transitions,coll_transitions,partition_function=None):
        '''levels is a list of instances of the Level class
        rad_transitions is a list of instances of the RadiativeTransition class
        coll_transitions is a list of instances of the CollisionalTransition class'''
        self.levels = levels
        self.rad_transitions = rad_transitions #list
        #dictionary with list of collisional transitions for each collision
        #partner:
        self.coll_transitions = coll_transitions 
        self.n_levels = len(self.levels)
        self.n_rad_transitions = len(self.rad_transitions)
        self.set_partition_function(partition_function=partition_function)

    @classmethod
    def from_LAMDA_datafile(cls,data_filepath,partition_function=None):
        """Alternative constructor using a LAMDA data file"""
        data = LAMDA_file.read(data_filepath)
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
        coll_transitions is a list of instances of the CollisionalTransition class
        line_profile_cls is the line profile class used to represent the line profile
        width_v is the width of the line in velocity'''
        Molecule.__init__(self,levels=levels,rad_transitions=rad_transitions,
                          coll_transitions=coll_transitions,
                          partition_function=partition_function)
        #convert radiative transitions to emission lines (but keep the same attribute name)
        self.rad_transitions = [atomic_transition.EmissionLine.from_radiative_transition(
                               radiative_transition=rad_trans,
                               line_profile_cls=line_profile_cls,width_v=width_v)
                               for rad_trans in self.rad_transitions]

    @classmethod
    def from_LAMDA_datafile(cls,data_filepath,line_profile_cls,
                            width_v,partition_function=None):
        """Alternative constructor using a LAMDA data file"""
        data = LAMDA_file.read(data_filepath)
        return cls(levels=data['levels'],rad_transitions=data['radiative transitions'],
                   coll_transitions=data['collisional transitions'],
                   line_profile_cls=line_profile_cls,width_v=width_v,
                   partition_function=partition_function)

    def get_tau_nu0(self,N,level_population):
        '''For a given total column density N and level population,
        compute the optical depth at line center for all radiative transitions'''
        tau_nu0 = []
        for line in self.rad_transitions:
            x1 = level_population[line.low.number]
            x2 = level_population[line.up.number]
            tau_nu0.append(line.tau_nu0(N1=x1*N,N2=x2*N))
        return np.array(tau_nu0)

    def get_Tex(self,level_population):
        '''For a given level population, compute the excitation temperature
        for all radiative transitions'''
        Tex = []
        for line in self.rad_transitions:
            x1 = level_population[line.low.number]
            x2 = level_population[line.up.number]
            Tex.append(line.Tex(x1=x1,x2=x2))
        return np.array(Tex)