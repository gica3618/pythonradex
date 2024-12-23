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

    '''Represents an atom or molecule with data read from a LAMDA file
    
    Attributes:
        levels (list of pythonradex.atomic_transition.Level): The energy levels
            of the molecule, in the same order as in the LAMDA file.
        rad_transitions (list of pythonradex.atomic_transition.RadiativeTransition):
            The radiative transitions of the molecule, in the same order as in the
            LAMDA file.
        coll_transitions (dict): The collisional transitions of the molecule. The
            dictionary keys correspond ot the different colliders. Each entry of
            the dictionary is a list of pythonradex.atomic_transition.CollisionalTransition
            objects, in the same order as in the LAMDA file.
        n_levels (:obj:`int`): the total number of levels
        n_rad_transitions (:obj:`int`): the total number of radiative transitions
    
    '''

    def __init__(self,datafilepath,read_frequencies=False,partition_function=None):
        """Constructs a new instance of the Molecule class using a LAMDA datafile
        
        Args:
            datafilepath (:obj:`str`): The filepath to the LAMDA file.
            read_frequencies (:obj:`bool`): Whether to read the frequencies of the
                radiative transitions from the file or not. If False, calculates the
                frequencies from the level energies given in the file. Setting this to
                True can be useful since frequencies are sometimes given with more
                significant digits than level energies. However, the LAMDA format does not
                force a file to contain the frequencies, so they might not be present.
            partition_function (func): A user-supplied partition function of one argument
                (temperature). Defaults to None. If equal to None, the partition function
                will be calculated by using the data from the provided LAMDA file.
        """
        data = LAMDA_file.read(datafilepath=datafilepath,
                               read_frequencies=read_frequencies)
        self.levels = data['levels']
        self.rad_transitions = data['radiative transitions']
        self.coll_transitions = data['collisional transitions'] 
        self.n_levels = len(self.levels)
        self.n_rad_transitions = len(self.rad_transitions)
        self.set_partition_function(partition_function=partition_function)

    def set_partition_function(self,partition_function):
        if partition_function is None:
            self.Z = self.Z_from_atomic_data
        else:
            self.Z = partition_function

    def Z_from_atomic_data(self,T):
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
        '''Computes the level populations in LTE for all levels
        
        Args:
            T (:obj:`float` or numpy.ndarray): The temperature.
        
        Returns:
            numpy.ndarray: The fractional level population. If only one temperature
            was given as input, the output array is one-dimensional, with the level
            populations corresponding to the levels in the order of the LAMDA file.
            If an array of temperatures was given as input, the output array has
            two dimensions, with the second corresponding to the different temperatures.
            
        '''
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
    
    '''Represents a an emitting molecule, i.e. a molecule with a specified line profile
    
    Attributes:
        levels (list of pythonradex.atomic_transition.Level): The energy levels
            of the molecule, in the same order as in the LAMDA file.
        rad_transitions (list of pythonradex.atomic_transition.EmissionLine):
            The radiative transitions of the molecule, in the same order as in the
            LAMDA file.
        coll_transitions (dict): The collisional transitions of the molecule. The
            dictionary keys correspond ot the different colliders. Each entry of
            the dictionary is a list of pythonradex.atomic_transition.CollisionalTransition
            objects, in the same order as in the LAMDA file.
        n_levels (:obj:`int`): the total number of levels
        n_rad_transitions (:obj:`int`): the total number of radiative transitions
    '''
    
    def __init__(self,datafilepath,line_profile_type,width_v,read_frequencies=False,
                 partition_function=None):
        """Constructs a new instance of the Molecule class using a LAMDA datafile
        
        Args:
            datafilepath (:obj:`str`): The filepath to the LAMDA file.
            line_profile_type (:obj:`str`): The type of the line profile. Either
                'rectangular' or 'Gaussian'
            width_v (:obj:`float`): The width of the line profile in [m/s]. For a Gaussian
                line profile, this corresponds to the FWHM.
            read_frequencies (:obj:`bool`): Whether to read the frequencies of the
                radiative transitions from the file or not. If False, calculates the
                frequencies from the level energies given in the file. Setting this to
                True can be useful since frequencies are sometimes given with more
                significant digits than level energies. However, the LAMDA format does not
                force a file to contain the frequencies, so they might not be present.
            partition_function (func): A user-supplied partition function of one argument
                (temperature). Defaults to None. If equal to None, the partition function
                will be calculated by using the data from the provided LAMDA file.
        """
        Molecule.__init__(self,datafilepath=datafilepath,
                          read_frequencies=read_frequencies,
                          partition_function=partition_function)
        self.width_v = width_v
        self.line_profile_type = line_profile_type
        #convert radiative transitions to emission lines (but keep the same attribute name)
        self.rad_transitions = [atomic_transition.EmissionLine.from_radiative_transition(
                               radiative_transition=rad_trans,
                               line_profile_type=line_profile_type,width_v=width_v)
                               for rad_trans in self.rad_transitions]
        self.identify_overlapping_lines()
        if self.any_line_has_overlap(line_indices=range(self.n_rad_transitions)):
            self.has_overlapping_lines = True
        else:
            self.has_overlapping_lines = False
        #collecting parameters, needed for numba-compiled functions:
        self.A21 = np.array([line.A21 for line in self.rad_transitions])
        self.B21 = np.array([line.B21 for line in self.rad_transitions])
        self.B12 = np.array([line.B12 for line in self.rad_transitions])
        self.nu0 = np.array([line.nu0 for line in self.rad_transitions])
        self.phi_nu0 = np.array([line.line_profile.phi_nu(line.nu0) for line
                                 in self.rad_transitions])
        self.gup_rad_transitions = np.array([line.up.g for line in self.rad_transitions])
        self.glow_rad_transitions = np.array([line.low.g for line in self.rad_transitions])
        self.nlow_rad_transitions = np.array([line.low.number for line in
                                              self.rad_transitions])
        self.nup_rad_transitions = np.array([line.up.number for line in
                                             self.rad_transitions])
        #for each level, the rad transitions going downward:
        self.downward_rad_transitions = []
        #for each level, the rad transitions involving that level: 
        self.level_transitions = []
        for i in range(self.n_levels):
            down_transitions = []
            level_transitions = []
            for j,trans in enumerate(self.rad_transitions):
                if trans.up.number == i:
                    down_transitions.append(j)
                if trans.up.number == i or trans.low.number == i:
                    level_transitions.append(j)
            self.downward_rad_transitions.append(down_transitions)
            self.level_transitions.append(level_transitions)
        self.Delta_E_rad_transitions = np.array([line.Delta_E for line in
                                                 self.rad_transitions])
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

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_tau_nu0(N,level_population,nlow_rad_transitions,nup_rad_transitions,
                     A21,phi_nu0,gup_rad_transitions,glow_rad_transitions,nu0):
        n_lines = nlow_rad_transitions.size
        tau_nu0 = np.empty(n_lines)
        for i in range(n_lines):
            N1 = level_population[nlow_rad_transitions[i]]*N
            N2 = level_population[nup_rad_transitions[i]]*N
            tau_nu0[i] = atomic_transition.fast_tau_nu(
                             A21=A21[i],phi_nu=phi_nu0[i],
                             g_up=gup_rad_transitions[i],g_low=glow_rad_transitions[i],
                             N1=N1,N2=N2,nu=nu0[i])
        return tau_nu0
    
    def get_tau_nu0(self,N,level_population):
        r'''Compute the optical depth at the rest frequency
        
        Args:
            N (:obj:`float`): the column density in [m\ :sup:`-2`]
            level_population (numpy.ndarray): the fractional population of each level,
                where the levels are in the order of the LAMDA file
        
        Returns:
            numpy.ndarray: the optical depth at the rest frequency
        '''
        return self.fast_tau_nu0(
                N=N,level_population=level_population,
                nlow_rad_transitions=self.nlow_rad_transitions,
                nup_rad_transitions=self.nup_rad_transitions,A21=self.A21,
                phi_nu0=self.phi_nu0,gup_rad_transitions=self.gup_rad_transitions,
                glow_rad_transitions=self.glow_rad_transitions,nu0=self.nu0)

    def get_tau_nu0_LTE(self,N,T):
        r'''Compute the optical depth at the rest frequency in LTE
        
        Args:
            N (:obj:`float`): the column density in [m\ :sup:`-2`]
            T (:obj:`float`): the temperature in [K]
        
        Returns:
            numpy.ndarray: the optical depth at the rest frequency assuming LTE
        '''
        level_population = self.LTE_level_pop(T=T)
        return self.fast_tau_nu0(
                N=N,level_population=level_population,
                nlow_rad_transitions=self.nlow_rad_transitions,
                nup_rad_transitions=self.nup_rad_transitions,A21=self.A21,
                phi_nu0=self.phi_nu0,gup_rad_transitions=self.gup_rad_transitions,
                glow_rad_transitions=self.glow_rad_transitions,nu0=self.nu0)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_Tex(level_population,nlow_rad_transitions,nup_rad_transitions,
                 Delta_E_rad_transitions,gup_rad_transitions,glow_rad_transitions):
        n_lines = nlow_rad_transitions.size
        Tex = np.empty(n_lines)
        for i in range(n_lines):
            Tex[i] = atomic_transition.fast_Tex(
                       Delta_E=Delta_E_rad_transitions[i],g_up=gup_rad_transitions[i],
                       g_low=glow_rad_transitions[i],
                       x1=level_population[nlow_rad_transitions[i]],
                       x2=level_population[nup_rad_transitions[i]])
        return Tex

    def get_Tex(self,level_population):
        r'''Compute the excitation temperature for all radiative transitions
        
        Args:
            level_population (numpy.ndarray): the fractional population of each level,
                where the levels are in the order of the LAMDA file
        
        Returns:
            numpy.ndarray: the excitation temperature for each radiative transition,
                in the order as in the LAMDA file
        '''
        return self.fast_Tex(
                 level_population=level_population,
                 nlow_rad_transitions=self.nlow_rad_transitions,
                 nup_rad_transitions=self.nup_rad_transitions,
                 Delta_E_rad_transitions=self.Delta_E_rad_transitions,
                 gup_rad_transitions=self.gup_rad_transitions,
                 glow_rad_transitions=self.glow_rad_transitions)

    def identify_overlapping_lines(self):
        self.overlapping_lines = []
        for i,line in enumerate(self.rad_transitions):
            width_nu = self.width_v/constants.c*line.nu0
            overlapping_lines = []
            for j,overlap_line in enumerate(self.rad_transitions):
                if j == i:
                    continue
                nu0_overlap_line = overlap_line.nu0
                width_nu_overlap_line = self.width_v/constants.c*nu0_overlap_line
                if self.line_profile_type == 'rectangular':
                    if np.abs(line.nu0-nu0_overlap_line)\
                                         <= width_nu/2 + width_nu_overlap_line/2:
                        overlapping_lines.append(j)
                elif self.line_profile_type == 'Gaussian':
                    #here I need to be conservative because the Gaussian profile can,
                    #in theory, be arbitrarily broad (for arbitrarily high optical depth)
                    #take +- 1.57 FWHM, where the Gaussian is 0.1% of the peak
                    if np.abs(line.nu0-nu0_overlap_line)\
                                    <= 1.57*width_nu + 1.57*width_nu_overlap_line:
                        overlapping_lines.append(j)
            self.overlapping_lines.append(overlapping_lines)

    def any_line_has_overlap(self,line_indices):
        for index in line_indices:
            if len(self.overlapping_lines[index]) > 0:
                return True
        return False

    def get_tau_line_nu(self,line_index,level_population,N):
        line = self.rad_transitions[line_index]
        def tau_line_nu(nu):
            return line.tau_nu(N1=N*level_population[line.low.number],
                               N2=N*level_population[line.up.number],nu=nu)
        return tau_line_nu

    def get_tau_tot_nu(self,line_index,level_population,N,tau_dust):
        overlapping_indices = self.overlapping_lines[line_index]
        line = self.rad_transitions[line_index]
        overlapping_lines = [self.rad_transitions[i] for i in overlapping_indices]
        def tau_tot_nu(nu):
            tau_tot = line.tau_nu(N1=N*level_population[line.low.number],
                                  N2=N*level_population[line.up.number],nu=nu)
            for ovl_line in overlapping_lines:
                x1 = level_population[ovl_line.low.number]
                x2 = level_population[ovl_line.up.number]
                tau_tot += ovl_line.tau_nu(N1=x1*N,N2=x2*N,nu=nu)
            tau_tot += tau_dust(nu)
            return tau_tot
        return tau_tot_nu