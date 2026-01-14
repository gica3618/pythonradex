#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 19:03:24 2019

@author: gianni
"""

#fluxes mostly agree, except for these cases:
#- non-LTE, optically thick case, LIME does not agree, maybe because it has difficulties
#to converge?
#- non-LTE, general case, LIME does not agree in terms of flux, don't know why...
#- for the sphere, optically thin: RADEX methods do not agree, presumably because
#the flux formula is not correct, i.e. isotropic flux density is assumed, which is not
#true

#NOTE: If RADEX is crashing, check if it is because the path to the outfile is too long.
#if that is the case, recompile RADEX with a modified radex.inc

import sys
sys.path.append('../lime')
import pyLime
from pythonradex import radiative_transfer, helpers
from scipy import constants
sys.path.append('../RADEX_wrapper')
import radex_wrapper
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

filename = 'co.dat'
Tkin = 100
coll_partner_density_cases = {'LTE':{'ortho-H2':1e8/constants.centi**3},
                              'non-LTE':{'ortho-H2':1e2/constants.centi**3}}
#coll_partner_density_cases = {'non-LTE':{'ortho-H2':1e2/constants.centi**3}}
N_cases = {'thick':5e18/constants.centi**2,'thin':1e15/constants.centi**2,
           'general':1e17/constants.centi**2}
ext_background = helpers.generate_CMB_background()
T_background = 2.73
line_profile_types = ['rectangular','Gaussian']
nu0 = 345.7959899*constants.giga
trans_number = 2
width_v = 1*constants.kilo
distance = 10*constants.parsec
#this is SO stupid, but LIME crashes when the filepath is too long...
#So I set it to something short on my machine...
filepath = os.path.join('/home/gianni/science/LAMDA_database_files',filename)
width_nu = width_v/constants.c*nu0
epsilon_nu = 1*constants.giga
freq_interval = radex_wrapper.Interval(min=nu0-epsilon_nu,max=nu0+epsilon_nu)

#run with LIME output enable first, to check that it runs fine, then suppress it
suppress_LIME_stdout_stderr = True
n_threads = 4

'''
################################################################
general_geometry = 'slab'
geometries = ['static slab','LVG slab']
slab_size = 100*constants.au
slab_surface = slab_size**2
Omega = slab_surface/distance**2
x = np.linspace(-slab_size,slab_size,60)
y = x.copy()
depth = slab_size/10
z = np.linspace(-2*depth,2*depth,60)
x3D,y3D,z3D = np.meshgrid(x,y,z,indexing='ij')
emission_region_3D = (np.abs(x3D)<slab_size/2) & (np.abs(y3D)<slab_size/2)\
                     & (np.abs(z3D)<depth/2)
def emission_region_2D(X,Y):
    return (np.abs(X)<slab_size/2) & (np.abs(Y)<slab_size/2)
lime_radius = 2*slab_size
def density(N):
    return N/depth
################################################################

'''
################################################################
general_geometry = 'sphere'
geometries = ['static sphere','static sphere RADEX']
r = 100*constants.au
Omega = np.pi*r**2/distance**2
x = np.linspace(-2*r,2*r,60)
y = x.copy()
z = x.copy()
x3D,y3D,z3D = np.meshgrid(x,y,z,indexing='ij')
r3D = np.sqrt(x3D**2+y3D**2+z3D**2)
emission_region_3D = r3D <= r
def emission_region_2D(X,Y):
    r2D = np.sqrt(X**2+Y**2)
    return r2D <= r
lime_radius = 4*r
def density(N):
    return N/(2*r)
################################################################

radex_wrapper_geo = {'sphere':'static sphere',
                     'slab':'LVG slab'}

T = np.ones((x.size,y.size,z.size))*Tkin
axes = {'x':x,'y':y,'z':z}
velocity = {'x':np.zeros_like(T),'y':np.zeros_like(T)}
broadening_param = width_v/(2*np.sqrt(np.log(2)))
velres = width_v/10
bandwidth = 8*width_v
nchan = int(bandwidth/velres)
n_pixels = 200
img_size = lime_radius/distance
imgres = img_size/n_pixels
general_img_kwargs = {'nchan':nchan,'velres':velres,'trans':trans_number,'pxls':n_pixels,
                      'imgres':imgres,'distance':distance,'phi':0,'units':'2 4'}
image = pyLime.LimeImage(theta=0,filename='image.fits',molI=0,**general_img_kwargs)
images = [image,]
n_solve_iters = {'LTE':7,'non-LTE':15}

print('general geometry of this run: {:s}'.format(general_geometry))

for N_case,LTE_case in itertools.product(N_cases,coll_partner_density_cases):
    N = N_cases[N_case]
    coll_partner_densities = coll_partner_density_cases[LTE_case]
    print('considering {:s} case ({:s})'.format(N_case,LTE_case))
    tau = {}
    Tex = {}
    obs_flux = {}
    obs_flux_density = {}
    lineprofile_nu0 = {}
    for geo,line_profile_type in itertools.product(geometries,line_profile_types):
        try:
            source = radiative_transfer.Source(
                        datafilepath=filepath,geometry=geo,
                        line_profile_type=line_profile_type,width_v=width_v)
        except ValueError:
            continue
        source.update_parameters(ext_background=ext_background,N=N,Tkin=Tkin,
                                collider_densities=coll_partner_densities,
                                T_dust=0,tau_dust=0)
        source.solve_radiative_transfer()
        #differs slightly from the nu0 given in the LAMDA file, because I calculate it
        #if I use the nu0 from LAMDA file, the line profile is 0 at nu0
        pythonradex_nu0 =  source.emitting_molecule.rad_transitions[trans_number].\
                              line_profile.nu0
        lineprofile_nu0[line_profile_type] =\
                           source.emitting_molecule.rad_transitions[trans_number].\
                            line_profile.phi_nu(pythonradex_nu0)
        key = '{:s} {:s}'.format(geo,line_profile_type)
        tau[key] = source.tau_nu0_individual_transitions[trans_number]
        Tex[key] = source.Tex[trans_number] 
        obs_flux[key] = source.frequency_integrated_emission(
                                   output_type="flux",solid_angle=Omega,
                                   transitions=[trans_number,])
        nu0 = source.emitting_molecule.rad_transitions[trans_number].nu0
        width_nu = width_v/constants.c*nu0
        nu = np.linspace(nu0-width_nu,nu0+width_nu,100)
        spec = source.spectrum(output_type="flux density",solid_angle=Omega,nu=nu)
        obs_flux_density[key] = np.max(spec)

    radex_input = radex_wrapper.RadexInput(
                     data_filename=filename,frequency_interval=freq_interval,
                     Tkin=Tkin,coll_partner_densities=coll_partner_densities,
                     T_background=T_background,column_density=N,Delta_v=width_v)
    wrapper = radex_wrapper.RadexWrapper(geometry=radex_wrapper_geo[general_geometry])
    results = wrapper.compute(radex_input)
    RADEX_intensity = (helpers.B_nu(nu=nu0,T=results['Tex'])-ext_background(nu0))\
                      * (1-np.exp(-results['tau']))
    RADEX_observed_flux_density = RADEX_intensity*Omega
    tau['RADEX'] = results['tau']
    Tex['RADEX'] = results['Tex']
    obs_flux_density['RADEX'] = RADEX_observed_flux_density
    obs_flux['RADEX'] = RADEX_observed_flux_density*width_nu

    const_density = density(N=N)
    n = np.where(emission_region_3D,const_density,0)
    n_orthoH2 = np.ones_like(n)*coll_partner_densities['ortho-H2']
    colliders = [pyLime.Collider(name='orthoH2',density=n_orthoH2),]
    radiating_species = [pyLime.RadiatingSpecie(moldatfile=filepath,density=n),]
    lime = pyLime.Lime(axes=axes,T=T,colliders=colliders,
                       radiating_species=radiating_species,
                       velocity=velocity,radius=lime_radius,
                       broadening_param=broadening_param,
                       images=images,level_population_filename='levelpop.fits',
                       n_solve_iters=n_solve_iters[LTE_case],
                       suppress_stdout_stderr=suppress_LIME_stdout_stderr,
                       n_threads=n_threads)
    lime.run()
    output_flux = pyLime.LimeFitsOutputFluxSI('image_SI.fits')
    output_flux.compute_projections()
    output_flux.plot_mom0()
    output_flux.plot_pv()
    output_tau = pyLime.LimeFitsOutputTau('image_Tau.fits')
    output_tau.compute_max_map()
    output_tau.plot_max_map()
    region = emission_region_2D(X=output_tau.X*distance,Y=output_tau.Z*distance)
    if general_geometry == 'slab':
        tau['Lime'] = np.median(output_tau.max_map[region])
    elif general_geometry == 'sphere':
        #need the opt depth at the center to compare to other codes
        tau['Lime'] = np.max(output_tau.max_map[region])
    obs_flux['Lime'] = output_flux.total_flux()
    lime_flux_density = np.trapz(np.trapz(output_flux.data,output_flux.z,axis=1),
                                 output_flux.x,axis=0)
    obs_flux_density['Lime'] = np.max(lime_flux_density)
    level_pop = pyLime.LimeLevelPopOutput('levelpop.fits').levelpops['CO']
    transition = source.emitting_molecule.rad_transitions[trans_number]
    up = transition.up
    low = transition.low
    up_pop = level_pop[:,up.index]
    low_pop = level_pop[:,low.index]
    Delta_E = transition.Delta_E
    lime_Tex = np.where(low_pop>0,-Delta_E/constants.k/np.log(low.g*up_pop/(up.g*low_pop)),
                        0)
    Tex['Lime'] = np.max(lime_Tex)

    if N_case == 'thick':
        T_bb = results['Tex']
        black_body_flux_density = helpers.B_nu(nu=nu0,T=T_bb)*Omega
        bb_key = 'black body T={:g} K'.format(T_bb)
        obs_flux_density[bb_key] = black_body_flux_density
        obs_flux[bb_key] = black_body_flux_density*width_nu
    elif N_case == 'thin' and LTE_case=='LTE':
        up_level_pop = source.emitting_molecule.LTE_level_pop(Tkin)[up.index]
        if general_geometry == 'slab':
            thin_LTE_flux = up_level_pop*N*slab_surface*transition.A21*Delta_E\
                             /(4*np.pi*slab_surface)*Omega
        elif general_geometry == 'sphere':
            thin_LTE_flux = 4/3*r**3*np.pi*const_density*up_level_pop\
                            *transition.A21*Delta_E/(4*np.pi*distance**2)
        obs_flux['thin LTE'] = thin_LTE_flux
        for line_profile_type,lp_n0 in lineprofile_nu0.items():
            obs_flux_density['thin LTE {:s}'.format(line_profile_type)] =\
                                                 thin_LTE_flux*lp_n0

    def split_thick_thin(obs,obs_name):
        params['{:s} rectangular'.format(obs_name)] = \
                           {key:value for key,value in obs.items()
                           if 'rectangular' in key or 'black body' in key or
                           key=='RADEX'}
        params['{:s} Gaussian'.format(obs_name)] =\
                             {key:value for key,value in obs.items()
                             if 'Gaussian' in key or key in ('Lime',)}
    params = {'Tex':Tex,'tau':tau}
    if N_case == 'thin':
        params['obs flux'] = obs_flux
        #for flux density, line profile is important in thin case
        #(thick case: just a black body)
        split_thick_thin(obs=obs_flux_density,obs_name='obs flux density')
    elif N_case == 'thick':
        #in this case, the line profile is important even for the total flux,
        #so I should only compare codes that use the same line profile
        split_thick_thin(obs=obs_flux,obs_name='obs flux')
        params['obs flux density'] = obs_flux_density
    elif N_case == 'general':
        split_thick_thin(obs=obs_flux,obs_name='obs flux')
        split_thick_thin(obs=obs_flux_density,obs_name='obs flux density')
    for paramname,param in params.items():
        for case,value in param.items():
            print('{:s} = {:g} ({:s})'.format(paramname,value,case))
        print('\n')
    print('\n\n')
plt.show()