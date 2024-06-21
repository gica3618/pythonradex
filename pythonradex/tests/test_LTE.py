# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:09:51 2017

@author: gianni
"""

from scipy import constants
from pythonradex import nebula,helpers
import os
import itertools
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
datafolder = os.path.join(here,'LAMDA_files')
Ntot_values = {'co':np.array((1e12,1e15,1e18))/constants.centi**2,
               'hcl':np.array((1e10,1e12,1e14))/constants.centi**2,
               'ocs@xpol':np.array((1e10,1e13,1e15))/constants.centi**2,
               'c':np.array((1e12,1e15,1e20))/constants.centi**2}
collider_densities = {'co':{'ortho-H2':1e9/constants.centi**3},
                      'hcl':{'ortho-H2':1e11/constants.centi**3},
                      'ocs@xpol':{'H2':1e9/constants.centi**3},
                      'c':{'e':1e6/constants.centi**3}}
#for HCL, some levels have relativel low LTE population (~1e-3), but pythonradex gives 0
#so I have to restrict to only the most highest populated levels for HCL
min_level_pop = {'co':0,'hcl':1e-2,'ocs@xpol':0,'c':0}
Tkin = 101
width_v = 2*constants.kilo
ext_background = helpers.generate_CMB_background()

filenames = ['co.dat','hcl.dat','ocs@xpol.dat','c.dat']
line_profiles = ('rectangular','Gaussian')
geometries = tuple(nebula.Nebula.geometries.keys())
iteration_modes = ('ALI','std')
use_ng_options = (True,False)
average_beta_options = (True,False)

def test_LTE():
    max_taus = []
    for filename,geo,lp,iter_mode,ng,avg_beta in itertools.product(
                        filenames,geometries,line_profiles,iteration_modes,
                        use_ng_options,average_beta_options):
        specie = filename.split('.')[0]
        datafilepath = os.path.join(datafolder,filename)
        neb = nebula.Nebula(datafilepath=datafilepath,geometry=geo,
                            line_profile=lp,width_v=width_v,iteration_mode=iter_mode,
                            use_NG_acceleration=ng,average_beta_over_line_profile=avg_beta)
        cloud_params = {'Tkin':Tkin,'ext_background':ext_background,
                        'collider_densities':collider_densities[specie]}
        for Ntot in Ntot_values[specie]:
            cloud_params['Ntot'] = Ntot
            neb.set_cloud_parameters(**cloud_params)
            neb.solve_radiative_transfer()
            LTE_level_pop = neb.emitting_molecule.LTE_level_pop(T=Tkin)
            selection = LTE_level_pop > min_level_pop[specie]*np.max(LTE_level_pop)
            assert np.allclose(neb.level_pop[selection],LTE_level_pop[selection],
                               atol=1e-6,rtol=1e-2)
            max_taus.append(np.max(neb.tau_nu0))
    print(f'max tau: {np.max(max_taus)}')