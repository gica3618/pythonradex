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

def is_comment(line):
    if line.replace(' ','')[0] == '!':
        return True
    else:
        return False

def read(datafilepath):
    '''
    Read a LAMDA data file.

    Reads a LAMDA data file and returns the data in the form of a dictionary.
    The LAMDA database can be found at http://home.strw.leidenuniv.nl/~moldata/molformat.html

    Parameters
    ----------
    datafilepath : str
        path to the file

    Returns
    -------
    dict
        Dictionary containing the data read from the file. The dictionary has the
        following keys:
        
        - 'levels': list of levels (instances of the Level class)

        - 'radiative transitions': list of radiative transitions (instances of RadiativeTransition class)

        - 'collisional transitions': dict, containing lists of instances of the CollisionalTransition class for each collision partner appearing in the file

        The elements of these lists are in the order they appear in the file
    '''
    #identifiers used in the LAMDA database files:
    LAMDA_coll_ID = {'1':'H2','2':'para-H2','3':'ortho-H2','4':'e',
                     '5':'H','6':'He','7':'H+'}
    datafile = open(datafilepath,'r')
    levels = []
    rad_transitions = []
    coll_transitions = {}
    for i,line in enumerate(datafile):
        if i<5:
            continue
        if is_comment(line):
            continue
        if line=='' or line=='\n':
            continue
        if i == 5:
            n_levels = int(line)
            continue
        if 6 < i <= 6+n_levels:
            leveldata = [float(string) for string in line.split()[:3]]
            #transforming energy from cm-1 to J; level numbers starting from 0:
            lev = atomic_transition.Level(
                        g=leveldata[2],
                        E=constants.c*constants.h*leveldata[1]/constants.centi,
                        number=int(leveldata[0])-1)
            levels.append(lev)
            continue
        if i == 8+n_levels:
            n_rad_transitions = int(line)
            continue
        if 9+n_levels < i <= 9+n_levels+n_rad_transitions:
            radtransdata = [float(string) for string in line.split()]
            up = next(level for level in levels if level.number==radtransdata[1]-1)
            low = next(level for level in levels if level.number==radtransdata[2]-1)
            rad_trans = atomic_transition.RadiativeTransition(
                                             up=up,low=low,A21=radtransdata[3])
            rad_transitions.append(rad_trans)
            continue
        if i == 11+n_levels+n_rad_transitions:
            coll_partner_offset = 0
            continue
        if i == 13+n_levels+n_rad_transitions + coll_partner_offset:
            coll_ID = LAMDA_coll_ID[line[0]]
            coll_transitions[coll_ID] = []
            continue
        if i == 15+n_levels+n_rad_transitions + coll_partner_offset:
            n_coll_transitions = int(line)
            continue
        if i == 17+n_levels+n_rad_transitions + coll_partner_offset:
            continue #this lines contains the number of temperature elements
        if i == 19+n_levels+n_rad_transitions + coll_partner_offset:
            coll_temperatures = np.array([float(string) for string in line.split()])
            continue
        if 20+n_levels+n_rad_transitions+coll_partner_offset < i <=\
             20+n_levels+n_rad_transitions+coll_partner_offset+n_coll_transitions:
            coll_trans_data = [float(string) for string in line.split()]
            up = next(level for level in levels if level.number==coll_trans_data[1]-1)
            low = next(level for level in levels if level.number==coll_trans_data[2]-1)
            K21_data = np.array(coll_trans_data[3:])*constants.centi**3
            coll_trans = atomic_transition.CollisionalTransition(
                                  up=up,low=low,K21_data=K21_data,
                                  Tkin_data=coll_temperatures)
            coll_transitions[coll_ID].append(coll_trans)
            if i == 20+n_levels+n_rad_transitions+coll_partner_offset+n_coll_transitions:
                coll_partner_offset += 9+n_coll_transitions
                continue
    datafile.close()
    return {'levels':levels,'radiative transitions':rad_transitions,
            'collisional transitions':coll_transitions}
