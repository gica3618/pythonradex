#Minimum example of the usage of pythonradex

from pythonradex import nebula,helpers
from scipy import constants
import numpy as np

data_filepath = './co.dat' #relative or absolute path to the LAMDA datafile
geometry = 'uniform sphere'
#spectral radiance of the background in units of [W/m2/Hz/sr].
#This is simply a function of nu. Here using a pre-defined function, namely the
#Planck function at 2.73 K (CMB), but one can define its own if wished
ext_background = helpers.CMB_background
Tkin = 150 #kinetic temperature of the colliders
#number density of the collsion partners in [m-3]
coll_partner_densities = {'para-H2':100/constants.centi**3,
                          'ortho-H2':250/constants.centi**3}
Ntot = 1e16/constants.centi**2 #column density in [m-2], i.e. here 1e16 cm-2
line_profile = 'square' #type of the line profile
width_v = 2*constants.kilo #witdh of the line in m/s

example_nebula = nebula.Nebula(
                        data_filepath=data_filepath,geometry=geometry,
                        ext_background=ext_background,Tkin=Tkin,
                        coll_partner_densities=coll_partner_densities,
                        Ntot=Ntot,line_profile=line_profile,width_v=width_v)
example_nebula.solve_radiative_transfer()
example_nebula.print_results() #outputs a table with results for all transitions

#examples of how to access the results directly:
#excitation temperature of second (as listed in the LAMDA file) transition:
print('Tex for second transition: {:g} K'.format(example_nebula.Tex[1]))
#fractional population of 4th level:
print('fractional population of 4th level: {:g}'.format(example_nebula.level_pop[3]))
#optical depth of lowest transition at line center:
print('optical depth lowest transition: {:g}'.format(example_nebula.tau_nu0[0]))

#compute the flux observed for a target at 20 pc with a radius of 3 au
d_observer = 20*constants.parsec
source_radius = 3*constants.au
source_surface = 4/3*source_radius**3*np.pi
#a list of observed fluxes for all transitions:
obs_fluxes = example_nebula.observed_fluxes(
                     source_surface=source_surface,d_observer=d_observer)
print('observed flux of second transition: {:g} W/m2'.format(obs_fluxes[1]))
