# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:37:59 2017

@author: gianni
"""

from pythonradex import LAMDA_file
import numpy as np

data_filepath = './co.dat' #relative or absolute path to the LAMDA datafile

data = LAMDA_file.read(data_filepath)

levels = data['levels']
rad_transitions = data['radiative transitions']
coll_transitions = data['collisional transitions']

print('Third level statistical weight: {:g}'.format(levels[2].g))
print('Third level energy: {:g} J'.format(levels[2].E))
print('Third level number: {:d}'.format(levels[2].number)) #index is 0 based
print('\n')

print('There are {:d} radiative transitions'.format(len(rad_transitions)))
#choose some random radiative transition:
rad_trans = rad_transitions[10]
print('Upper level stat weight: {:g}'.format(rad_trans.up.g))
print('Lower level energy: {:g} J'.format(rad_trans.low.E))
print('frequency: {:g} Hz'.format(rad_trans.nu0))
print('Energy difference: {:g} J'.format(rad_trans.Delta_E))
print('Einstein A21: {:g}'.format(rad_trans.A21))
print('example excitation temperature:')
print(rad_trans.Tex(x1=0.3,x2=0.1))
#one can also give numpy arrays as input:
x1 = np.array((0.1,0.5,0.15))
x2 = np.array((0.05,0.1,0.07))
print(rad_trans.Tex(x1=x1,x2=x2))
print('\n')

print(coll_transitions.keys())
coll_transitions_ortho_H2 = coll_transitions['ortho-H2']
print('there are {:d} ortho-H2 coll transitions'.format(len(coll_transitions_ortho_H2)))
#choose random collisional transition:
coll_trans = coll_transitions['ortho-H2'][99]
print('number of upper level: {:d}'.format(coll_trans.up.number))
print('stat weight of lower level: {:g}'.format(coll_trans.low.g))
print('energy difference of transitions: {:g} J'.format(coll_trans.Delta_E))
print('transition name: {:s}'.format(coll_trans.name))
Tkin = 100.5
print('coll coeff K21 (at T={:g} K): {:g} m3/s'.format(Tkin,coll_trans.coeffs(Tkin)['K21']))
#one can also give numpy arrays as input:
Tkin = np.array((52.3,70.4,100.2,150.4))
print(coll_trans.coeffs(Tkin=Tkin))