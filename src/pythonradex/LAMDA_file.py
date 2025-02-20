# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:05:01 2017

@author: gianni
"""
from pythonradex import atomic_transition
from scipy import constants
import numpy as np

def is_comment(line):
    if line.replace(' ','')[0] == '!':
        return True
    else:
        return False

def read(datafilepath,read_frequencies,read_quantum_numbers=False):
    '''
    Read a LAMDA data file. LAMDA files can be downloaded from the LAMDA database
    at http://home.strw.leidenuniv.nl/~moldata

    Args:
        datafilepath (:obj:`str`): The filepath to the LAMDA file.
        read_frequencies (:obj:`bool`): Whether to read the frequencies of the
            radiative transitions from the file or not. If False, calculates the
            frequencies from the level energies given in the file. Setting this to
            True can be useful since frequencies are sometimes given with more
            significant digits than level energies. However, the LAMDA format does not
            force a file to contain the frequencies, so they might not be present.
        read_quantum_numbers (:obj:`bool`): Whether to read the quantum numbers
            from the file. Defaults to False. The LAMDA format does not force a
            file to list quantum numbers, so they might not be present.
    
    Returns:
        dict: A dictionary containing the data read from the file.
            The following keys give acces to lists containing the data.
            The elements of these lists are in the order they appear in the file.
            'levels' (a list of pythonradex.atomic_transition.Level objects);
            'radiative transitions' (a list of 
            pythonradex.atomic_transitions.RadiativeTransition objects);
            'collisional transitions' (a list of
            pythonradex.atomic_transition.CollisionalTransition objects);
            'quantum numbers' (list of str, empty if read_quantum_numbers
            was set to False)
    '''
    #identifiers used in the LAMDA database files:
    LAMDA_coll_ID = {'1':'H2','2':'para-H2','3':'ortho-H2','4':'e',
                     '5':'H','6':'He','7':'H+'}
    datafile = open(datafilepath,'r')
    levels = []
    rad_transitions = []
    coll_transitions = {}
    quantum_numbers = []
    comment_offset = 0
    for i,line in enumerate(datafile):
        if is_comment(line) or line=='' or line=='\n':
            comment_offset += 1
            continue
        if i == 0+comment_offset:
            continue
        if i == 1+comment_offset:
            continue
        if i == 2+comment_offset:
            n_levels = int(line)
            continue
        if 3+comment_offset <= i < 3+comment_offset+n_levels:
            line_entries = line.split()
            leveldata = [float(string) for string in line_entries[:3]]
            assert int(leveldata[0]) == i-2-comment_offset,\
                                     'level numeration not consistent'
            #transforming energy from cm-1 to J; level numbers starting from 0:
            lev = atomic_transition.Level(
                        g=leveldata[2],
                        E=constants.c*constants.h*leveldata[1]/constants.centi,
                        number=int(leveldata[0])-1)
            levels.append(lev)
            if read_quantum_numbers:
                quantum_numbers.append(line_entries[3])
            continue
        if i == 3+comment_offset+n_levels:
            n_rad_transitions = int(line)
            continue
        if 4+comment_offset+n_levels <= i < 4+comment_offset+n_levels+n_rad_transitions:
            radtransdata = [float(string) for string in line.split()]
            up = next(level for level in levels if level.number==radtransdata[1]-1)
            low = next(level for level in levels if level.number==radtransdata[2]-1)
            rad_trans_kwargs = {'up':up,'low':low,'A21':radtransdata[3]}
            if read_frequencies:
                nu0 = radtransdata[4]*constants.giga
                Delta_E = up.E-low.E
                assert np.isclose(nu0,Delta_E/constants.h,atol=0,rtol=1e-3),\
                     'read frequency is inconsistent with level energies'
                rad_trans_kwargs['nu0'] = nu0
            rad_trans = atomic_transition.RadiativeTransition(**rad_trans_kwargs) 
            rad_transitions.append(rad_trans)
            continue
        if i == 4+comment_offset+n_levels+n_rad_transitions:
            coll_partner_offset = 0
            continue
        if i == 5+comment_offset+n_levels+n_rad_transitions + coll_partner_offset:
            coll_ID = LAMDA_coll_ID[line[0]]
            coll_transitions[coll_ID] = []
            continue
        if i == 6+comment_offset+n_levels+n_rad_transitions + coll_partner_offset:
            n_coll_transitions = int(line)
            continue
        if i == 7+comment_offset+n_levels+n_rad_transitions + coll_partner_offset:
            continue #this lines contains the number of temperature elements
        if i == 8+comment_offset+n_levels+n_rad_transitions + coll_partner_offset:
            coll_temperatures = np.array([float(string) for string in line.split()])
            continue
        last_coll_trans_i = 9+comment_offset+n_levels+n_rad_transitions\
                               +coll_partner_offset+n_coll_transitions-1
        if 9+comment_offset+n_levels+n_rad_transitions+coll_partner_offset <= i <=\
                 last_coll_trans_i:
            coll_trans_data = [float(string) for string in line.split()]
            up = next(level for level in levels if level.number==coll_trans_data[1]-1)
            low = next(level for level in levels if level.number==coll_trans_data[2]-1)
            K21_data = np.array(coll_trans_data[3:])*constants.centi**3
            coll_trans = atomic_transition.CollisionalTransition(
                                  up=up,low=low,K21_data=K21_data,
                                  Tkin_data=coll_temperatures)
            coll_transitions[coll_ID].append(coll_trans)
            if i == last_coll_trans_i:
                coll_partner_offset += 4+n_coll_transitions
                continue
    datafile.close()
    return {'levels':levels,'radiative transitions':rad_transitions,
            'collisional transitions':coll_transitions,'quantum numbers':quantum_numbers}