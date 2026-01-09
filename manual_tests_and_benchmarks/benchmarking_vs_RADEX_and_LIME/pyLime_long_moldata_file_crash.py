#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 10:21:23 2026

@author: gianni
"""

import sys
sys.path.append('../lime')
import pyLime
import numpy as np
from scipy import constants
import os


x = np.linspace(-100,100,30)*constants.au
y = x
z = np.linspace(-30,30,30)*constants.au
T = np.ones((x.size,y.size,z.size))*75
sigma_r = 20*constants.au
sigma_z = 5*constants.au
sigma_x = sigma_z
r0 = 50*constants.au
x3D,y3D,z3D = np.meshgrid(x,y,z,indexing='ij')
r = np.sqrt(x3D**2+y3D**2)
x0 = 25*constants.au
y0 = -25*constants.au
ne = 10*np.exp(-(r-r0)**2/(2*sigma_r**2))
ne = np.where(y3D<0,0,ne)
ne += 30*np.exp(-(x3D-x0)**2/(2*sigma_x**2))*np.exp(-(y3D-y0)**2/(2*sigma_x**2))
ne *= np.exp(-z3D**2/(2*sigma_z**2))
ne/=constants.centi**3
colliders = [pyLime.Collider(name='e',density=ne),
             pyLime.Collider(name='H',density=ne)]
axes = {'x':x,'y':y,'z':z}
#if I set the following to a long filepath in the pythonradex folders, LIME
#crashes...
#datafolder = '/home/gianni/science/LAMDA_database_files'
datafolder = "/home/gianni/science/projects/code/pythonradex_joss/pythonradex/tests/LAMDA_files"
radiating_species = [pyLime.RadiatingSpecie(moldatfile=os.path.join(datafolder,"c.dat"),
                                     density=ne),
                     pyLime.RadiatingSpecie(moldatfile=os.path.join(datafolder,'c+.dat'),
                                     density=ne)]
Mstar = 2e30*1.75
vkep = np.sqrt(constants.G*Mstar/r)
velocity = {'x':y3D/r*vkep,'y':-x3D/r*vkep}
radius = 300*constants.au
broadening_param = 700
general_img_kwargs = {'nchan':201,'velres':297.4,'trans':0,'pxls':200,
                      'imgres':0.1*constants.arcsec,'distance':20*constants.parsec,
                      'phi':0,'units':'2 4'}
image_c = pyLime.LimeImage(theta=0,filename='test_c.fits',molI=0,**general_img_kwargs)
image_cplus = pyLime.LimeImage(theta=0,filename='test_c+.fits',molI=1,**general_img_kwargs)
images = [image_c,image_cplus]
lime = pyLime.Lime(axes=axes,T=T,colliders=colliders,radiating_species=radiating_species,
            velocity=velocity,radius=radius,broadening_param=broadening_param,
            images=images,suppress_stdout_stderr=False)
lime.run()