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
            dictionary keys correspond to the different colliders. Each entry of
            the dictionary is a list of pythonradex.atomic_transition.CollisionalTransition
            objects, in the same order as in the LAMDA file.
        n_levels (:obj:`int`): the total number of levels
        n_rad_transitions (:obj:`int`): the total number of radiative transitions
    
    '''

    def __init__(self,datafilepath,partition_function=None):
        """Constructs a new instance of the Molecule class using a LAMDA datafile
        
        Args:
            datafilepath (:obj:`str`): The filepath to the LAMDA file.
            partition_function (func): A user-supplied partition function of one argument
                (temperature). Defaults to None. If equal to None, the partition function
                will be calculated by using the data from the provided LAMDA file.
        """
        try:
            #reading frequencies can be useful since frequencies are sometimes
            #given with more significant digits than level energies. However,
            #the LAMDA format does not orce a file to contain the frequencies,
            #so they might not be present.
            data = LAMDA_file.read(datafilepath=datafilepath,
                                   read_frequencies=True)
        except:
            data = LAMDA_file.read(datafilepath=datafilepath,
                                   read_frequencies=False)
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
            dictionary keys correspond to the different colliders. Each entry of
            the dictionary is a list of pythonradex.atomic_transition.CollisionalTransition
            objects, in the same order as in the LAMDA file.
        n_levels (:obj:`int`): the total number of levels
        n_rad_transitions (:obj:`int`): the total number of radiative transitions
    '''
    
    def __init__(self,datafilepath,line_profile_type,width_v,read_frequencies=False,
                 partition_function=None):
        """Constructs a new instance of the EmittingMolecule class using a LAMDA datafile
        
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
        self.Tkin_data = {}
        self.Tkin_data_limits = {}
        for collider,coll_transitions in self.coll_transitions.items():
            self.Tkin_data[collider] = coll_transitions[0].Tkin_data
            assert np.all(self.Tkin_data[collider][:-1] < self.Tkin_data[collider][1:])
            for coll_trans in coll_transitions:
                assert np.all(coll_trans.Tkin_data==self.Tkin_data[collider])
            self.Tkin_data_limits[collider] = np.min(self.Tkin_data[collider]),\
                                               np.max(self.Tkin_data[collider])
        self.set_K21_matrix()
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
        self.Delta_E_rad_transitions = np.array([line.Delta_E for line in
                                                 self.rad_transitions])
        self.coll_gups = {}
        self.coll_glows = {}
        self.coll_DeltaEs = {}
        self.coll_nup = {}
        self.coll_nlow = {}
        for collider,coll_transitions in self.coll_transitions.items():
            self.coll_glows[collider] = np.array([c.low.g for c in coll_transitions])
            self.coll_gups[collider] = np.array([c.up.g for c in coll_transitions])
            self.coll_DeltaEs[collider] = np.array([c.Delta_E for c in
                                                    coll_transitions])
            self.coll_nlow[collider] = np.array([c.low.number for c in coll_transitions])
            self.coll_nup[collider] = np.array([c.up.number for c in coll_transitions])


    def get_tau_nu0_lines(self,N,level_population):
        r'''Compute the optical depth at the rest frequency of all lines (dust
        and overlapping lines are ignored)
        
        Args:
            N (:obj:`float`): the column density in [m\ :sup:`-2`]
            level_population (numpy.ndarray): the fractional population of each level,
                where the levels are in the order of the LAMDA file
        
        Returns:
            numpy.ndarray: the optical depth at the rest frequency
        '''
        N1 = level_population[self.nlow_rad_transitions]*N
        N2 = level_population[self.nup_rad_transitions]*N
        tau_nu0 = atomic_transition.tau_nu(
                         A21=self.A21,phi_nu=self.phi_nu0,
                         g_up=self.gup_rad_transitions,g_low=self.glow_rad_transitions,
                         N1=N1,N2=N2,nu=self.nu0)
        return tau_nu0

    def get_tau_nu0_lines_LTE(self,N,T):
        r'''Compute the optical depth at the rest frequency in LTE for all lines
        (dust and overlapping lines are ignored)
        
        Args:
            N (:obj:`float`): the column density in [m\ :sup:`-2`]
            T (:obj:`float`): the temperature in [K]
        
        Returns:
            numpy.ndarray: the optical depth at the rest frequency assuming LTE
        '''
        level_population = self.LTE_level_pop(T=T)
        return self.get_tau_nu0_lines(N=N,level_population=level_population)

    def get_Tex(self,level_population):
        r'''Compute the excitation temperature for all radiative transitions
        
        Args:
            level_population (numpy.ndarray): the fractional population of each level, 
                where the levels are in the order of the LAMDA file
        
        Returns:
            numpy.ndarray: the excitation temperature for each radiative transition,
            in the order as in the LAMDA file
        '''
        Tex = atomic_transition.Tex(
                   Delta_E=self.Delta_E_rad_transitions,g_up=self.gup_rad_transitions,
                   g_low=self.glow_rad_transitions,
                   x1=level_population[self.nlow_rad_transitions],
                   x2=level_population[self.nup_rad_transitions])
        return Tex

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

    def set_K21_matrix(self):
        self.K21_matrix = {collider:np.zeros((len(coll_transitions),
                                              self.Tkin_data[collider].size))
                           for collider,coll_transitions in
                           self.coll_transitions.items()}
        for collider,coll_transitions in self.coll_transitions.items():
            for i,coll_trans in enumerate(coll_transitions):
                self.K21_matrix[collider][i,:] = coll_trans.K21_data
    
    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def construct_K_matrix(n_levels,K12,K21,nlow,nup):
        K = np.zeros((n_levels,n_levels))
        for i in range(len(K12)):
            nl = nlow[i]
            nu = nup[i]
            #production of upper level from lower level:
            K[nu,nl] += K12[i]
            #destruction of lower level by transitions to upper level:
            K[nl,nl] += -K12[i]
            #production lower level from upper level:
            K[nl,nu] += K21[i]
            #destruction of upper level by transition to lower level:
            K[nu,nu] += -K21[i]
        return K

    def interpolate_K(self,Tkin,collider):
        Tlimits = self.Tkin_data_limits[collider]
        assert Tlimits[0] <= Tkin <= Tlimits[1]
        output = {}
        Tkin_data = self.Tkin_data[collider]
        j = np.searchsorted(Tkin_data,Tkin,side='left')
        if j == 0:
            K21 = self.K21_matrix[collider][:,0]
        else:
            i = j-1
            x0 = Tkin_data[i]
            y0 = self.K21_matrix[collider][:,i]
            x1 = Tkin_data[j]
            y1 = self.K21_matrix[collider][:,j]
            x = Tkin
            K21 = (y0*(x1-x) + y1*(x-x0)) / (x1-x0)
        K12 = atomic_transition.compute_K12(
                  K21=K21,g_up=self.coll_gups[collider],
                  g_low=self.coll_glows[collider],
                  Delta_E=self.coll_DeltaEs[collider],Tkin=Tkin)
        output['K21'] = K21
        output['K12'] = K12
        return output

    def get_GammaC(self,Tkin,collider_densities):
        GammaC = np.zeros((self.n_levels,)*2)
        for collider,coll_dens in collider_densities.items():
            interpK = self.interpolate_K(Tkin=Tkin,collider=collider)
            K = self.construct_K_matrix(
                     n_levels=self.n_levels,K12=interpK['K12'],K21=interpK['K21'],
                     nlow=self.coll_nlow[collider],nup=self.coll_nup[collider])
            GammaC += K*coll_dens
        return GammaC