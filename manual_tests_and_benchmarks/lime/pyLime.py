# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 12:49:34 2018

@author: gianni
"""

import numpy as np
from scipy import constants
import os
from astropy.io import fits
import matplotlib.pyplot as plt


class GridFile():

    def __init__(self,filepath,axes):
        self.file = open(filepath,'w')
        self.axes = axes

    def add_txt(self,txt):
        self.file.write("{:s};\n".format(txt))

    def add_constant(self,name,datatype,value):
        self.add_txt('{:s} {:s} = {:.3e}'.format(datatype,name,value))

    @staticmethod
    def array_to_Cstring(array,name):
        np.set_printoptions(threshold=np.inf)
        C_array_str = 'double {:s}'.format(name)
        for nelements in array.shape:
            C_array_str += '[{:d}]'.format(nelements)
        formatter = {'float_kind':lambda x: "{:.3e}".format(x)}
        values_str = np.array2string(array,separator=',',formatter=formatter)
        values_str = values_str.replace('\n','').replace('[','{').replace(']','}')\
                     .replace(' ','')
        C_array_str += ' = {:s}'.format(values_str)
        return C_array_str

    def add_array_values(self,array,name):
        self.add_txt(self.array_to_Cstring(array=array,name=name))

    def write_grid_axes(self):
        for ax_name,ax in self.axes.items():
            self.add_array_values(array=ax,name='grid_{:s}'.format(ax_name))
            self.add_constant(name='grid_{:s}_size'.format(ax_name),datatype='int',
                              value=ax.size)

    def check_array_shape(self,array):
        shape = array.shape
        for i,axname in enumerate(('x','y','z')):
            assert shape[i] == self.axes[axname].size,\
                        'array is not consistent with {:s} axis'.format(axname)

    def write_grid_values(self,name,array,floor):
        self.add_array_values(array=array,name='grid_{:s}'.format(name))
        self.add_constant(name='grid_{:s}_floor'.format(name),datatype='double',
                          value=floor)

    def close(self):
        self.file.close()


class LimeImageParam():

    def __init__(self,name,value,unit_change_factor=1):
        self.name = name
        self.value = value * unit_change_factor

    def value_str(self):
        raise NotImplementedError

    def input_str(self,img_number):
        return 'img[{:d}].{:s} = {:s};\n'.format(img_number,self.name,self.value_str())


class IntegerLimeImageParam(LimeImageParam):

    def value_str(self):
        return '{:d}'.format(self.value)


class FloatLimeImageParam(LimeImageParam):

    def value_str(self):
        return '{:g}'.format(self.value)


class StringLimeImageParam(LimeImageParam):

    def value_str(self):
        return '\"{:s}\"'.format(self.value)


class LimeImage():

    def __init__(self,nchan,velres,trans,molI,pxls,imgres,distance,theta,
                 phi,units,filename):
        self.params = [IntegerLimeImageParam(name='nchan',value=nchan),
                       FloatLimeImageParam(name='velres',value=velres),
                       IntegerLimeImageParam(name='trans',value=trans),
                       IntegerLimeImageParam(name='molI',value=molI),
                       IntegerLimeImageParam(name='pxls',value=pxls),
                       FloatLimeImageParam(name='imgres',value=imgres,
                                           unit_change_factor=1/constants.arcsec),
                       FloatLimeImageParam(name='distance',value=distance),
                       FloatLimeImageParam(name='theta',value=theta),
                       FloatLimeImageParam(name='phi',value=phi),
                       StringLimeImageParam(name='units',value=units),
                       StringLimeImageParam(name='filename',value=filename)]

    def write_image_params(self,file,img_number):
        for p in self.params:
            file.write(p.input_str(img_number))

    def get_param_value(self,paramname):
        candidate_params = [p for p in self.params if p.name==paramname]
        assert len(candidate_params) == 1
        return candidate_params[0].value


class RadiatingSpecie():

    def __init__(self,moldatfile,density):
        self.moldatfile = moldatfile
        self.density = density


class Collider():

    def __init__(self,name,density):
        assert name in Lime.coll_partner_IDs.keys(),\
                            'unknown collider {:s}'.format(name)
        self.name = name
        self.density = density


class Lime():

    pyLime_folderpath = os.path.dirname(os.path.realpath(__file__))
    template_input_filepath = os.path.join(pyLime_folderpath,'template_model.c')
    lime_exec = os.path.join(pyLime_folderpath,'lime-master','lime')
    grid_filename = 'grid_values.h'
    input_filename = 'model.c'
    #TODO need to investigate whether this floor stuff is really necessary
    density_floor = 1e-5
    nmol_floor = density_floor
    T_floor = 2.73
    velocity_floor = 1
    coll_partner_IDs = {'H2':'CP_H2','paraH2':'CP_p_H2','orthoH2':'CP_o_H2',
                        'e':'CP_e','H':'CP_H','He':'CP_He','Hplus':'CP_Hplus'}

    def __init__(self,axes,T,colliders,radiating_species,velocity,radius,
                 broadening_param,images,n_threads=10,n_solve_iters=14,
                 level_population_filename=None,lte_only=False,
                 suppress_stdout_stderr=False):
        self.axes = axes
        self.T = T
        self.colliders = colliders
        self.radiating_species = radiating_species
        self.velocity = velocity
        ax_extends = [np.max(np.abs(ax)) for ax in self.axes.values()]
        assert radius > np.max(ax_extends),\
                  'param radius ({:g} au) too small (should be at least {:g} au)'.format(
                          radius/constants.au,np.max(ax_extends)/constants.au)
        self.radius = radius
        self.broadening_param = broadening_param
        self.images = images
        self.assert_images_request_valid_species()
        self.n_threads = n_threads
        self.n_solve_iters = n_solve_iters
        self.level_population_filename = level_population_filename
        self.lte_only = lte_only
        self.suppress_stdout_stderr = suppress_stdout_stderr

    def assert_images_request_valid_species(self):
        valid_molIs = tuple(range(len(self.radiating_species)))
        for img in self.images:
            molI = img.get_param_value('molI')
            assert molI in valid_molIs, 'molI = {:d} is not valid. Valid molI values: {:s}'\
                                        .format(molI,str(valid_molIs))

    @staticmethod
    def grid_interpolation_string(name):
        return 'grid_interpolation_3D(x, y, z, grid_{:s}, grid_{:s}_floor)'\
                 .format(name,name)

    @staticmethod
    def density_grid_str(coll_name):
        return 'density{:s}'.format(coll_name)

    @staticmethod
    def nmol_grid_str(moldatfile):
        assert moldatfile[-4:] == '.dat'
        gasname = os.path.basename(moldatfile)[:-4]
        gasname = gasname.replace('+','plus')
        return 'nmol{:s}'.format(gasname)

    def write_grid_file(self):
        grid_file = GridFile(filepath=self.grid_filename,axes=self.axes)
        grid_file.write_grid_axes()
        for collider in self.colliders:
            grid_file.write_grid_values(name=self.density_grid_str(collider.name),
                                        array=collider.density,floor=self.density_floor)
        grid_file.write_grid_values(name='temperature',array=self.T,floor=self.T_floor)
        for radiating_specie in self.radiating_species:
            grid_file.write_grid_values(name=self.nmol_grid_str(radiating_specie.moldatfile),
                                        array=radiating_specie.density,floor=self.nmol_floor)
        for ax in ('x','y'):
            grid_file.write_grid_values(
                    name='velocity_{:s}'.format(ax),array=self.velocity[ax],
                    floor=self.velocity_floor)
        grid_file.close()

    def write_LIME_input_file(self):
        template = open(self.template_input_filepath,'r')
        input_file = open(self.input_filename,'w')
        for line in template:
            if 'par->radius' in line:
                input_file.write('par->radius = {:g}*AU;\n'.format(
                                                    self.radius/constants.au))
            elif 'par->moldatfile' in line:
                for i,radiating_specie in enumerate(self.radiating_species):
                    input_file.write('par->moldatfile[{:d}] = "{:s}";\n'.format(
                                      i,radiating_specie.moldatfile))
            elif 'par->nSolveIters' in line:
                input_file.write('par->nSolveIters = {:d};\n'.format(self.n_solve_iters))
            elif 'par->nThreads' in line:
                input_file.write('par->nThreads = {:d};\n'.format(self.n_threads))
            elif 'par->lte_only' in line:
                #using the fact that True and False are treated as integers in python:
                input_file.write('par->lte_only = {:d};\n'.format(self.lte_only))
            elif 'par->gridOutFiles[4]' in line:
                if self.level_population_filename is None:
                    pass
                else:
                    input_file.write('par->gridOutFiles[4] = "{:s}";\n'.format(
                          self.level_population_filename))
            elif 'par->collPartIds[0]' in line:
                for i,collider in enumerate(self.colliders):
                    input_file.write('par->collPartIds[{:d}] = {:s};\n'.format(
                          i,self.coll_partner_IDs[collider.name]))
            elif 'INSERT IMAGES HERE' in line:
                for i,img in enumerate(self.images):      
                    img.write_image_params(file=input_file,img_number=i)
            elif 'density[0]' in line:
                for i,collider in enumerate(self.colliders):
                    grid_name = self.density_grid_str(collider.name)
                    input_file.write('density[{:d}] = {:s};\n'.format(
                        i,self.grid_interpolation_string(grid_name)))
            elif 'nmol[0]' in line:
                for i,radiating_specie in enumerate(self.radiating_species):
                    grid_name = self.nmol_grid_str(radiating_specie.moldatfile)
                    input_file.write('nmol[{:d}] = {:s};\n'.format(
                        i,self.grid_interpolation_string(grid_name)))
            elif '*doppler =' in line:
                input_file.write('*doppler = {:g};\n'.format(self.broadening_param))
            else:
                input_file.write(line)

    def clean_up(self):
        for filename in (self.grid_filename,self.input_filename):
            os.remove(filename)

    def run(self):
        self.write_grid_file()
        self.write_LIME_input_file()
        command = f'{self.lime_exec} -n {self.input_filename}'
        if self.suppress_stdout_stderr:
            command += ' > /dev/null 2>&1'
        os.system(command)
        self.clean_up()


class LimeFitsOutput():

    def __init__(self,filename):
        self.filename = filename
        hdu = fits.open(filename)[0]
        self.header = hdu.header
        self.bunit = self.header['bunit'].replace(' ','')
        data = hdu.data[0,:,:,:]
        self.data = np.swapaxes(data,0,2)
        self.check_units()
        self.determine_axes()

    def check_units(self):
        for i in range(1,3):
            assert self.header['cunit{:d}'.format(i)] == 'DEG'
        assert self.header['cunit3'] == 'M/S'

    def determine_axes(self):
        pix_axes = []
        for i in range(1,4):
            pix_ax = np.arange(self.header['NAXIS{:d}'.format(i)])\
                       - self.header['CRPIX{:d}'.format(i)]
            pix_axes.append(pix_ax)
        self.dx = np.radians(np.abs(self.header['CDELT1']))
        self.dz = np.radians(np.abs(self.header['CDELT2']))
        self.dv = self.header['CDELT3']
        self.x = pix_axes[0]*self.dx
        self.z = pix_axes[1]*self.dz
        self.v = pix_axes[2]*self.dv
        self.nu = self.header['RESTFREQ']*(1-self.v/constants.c)
        self.X,self.Z = np.meshgrid(self.x,self.z,indexing='ij')
        self.P,self.V = np.meshgrid(self.x,self.v,indexing='ij')

    def mom0like_plot(self,data,title=None):
        fig,ax = plt.subplots()
        plot = ax.pcolormesh(self.X/constants.arcsec,self.Z/constants.arcsec,data,
                             shading='auto')
        ax.set_title(title)
        ax.set_xlabel('x [arcsec]')
        ax.set_ylabel('z [arcsec]')
        return ax,plot

    def set_colorbar(self,plot,label):
        cbar = plt.colorbar(plot)
        cbar.ax.set_ylabel(label)


class LimeFitsOutputFlux(LimeFitsOutput):

    def compute_projections(self):
        self.mom0 = -np.trapz(self.data,self.nu,axis=-1)
        self.pv_diagram = np.trapz(self.data,self.z,axis=1)

    def plot_mom0(self,title=None):
        ax,plot = self.mom0like_plot(data=self.mom0,title=title)
        self.set_colorbar(plot=plot,label=self.mom0_units_label)

    def plot_pv(self,title=None):
        plt.figure()
        plot = plt.pcolormesh(self.P/constants.arcsec,self.V/constants.kilo,
                              self.pv_diagram,shading='auto')
        plt.title(title)
        plt.xlabel('p [arcsec]')
        plt.ylabel('v [km/s]')
        self.set_colorbar(plot=plot,label=self.pv_units_label)


class LimeFitsOutputFluxSI(LimeFitsOutputFlux):

    mom0_units_label = 'W/m2/sr'
    pv_units_label = 'W/m2/Hz/rad'

    def check_units(self):
        LimeFitsOutputFlux.check_units(self)
        assert self.bunit == 'WM2HZSR'

    def total_flux(self):
        return np.trapz(np.trapz(-np.trapz(self.data,self.nu,axis=-1),self.z,axis=-1),
                        self.x)


class LimeFitsOutputfluxJy(LimeFitsOutputFlux):

    mom0_units_label = 'Jy/pixel*km/s'
    pv_units_label = 'Jy/pixel*rad'

    def check_units(self):
        LimeFitsOutputFlux.check_units(self)
        assert self.bunit == 'JY/PIXEL'


class LimeFitsOutputTau(LimeFitsOutput):

    def check_units(self):
        LimeFitsOutput.check_units(self)
        assert self.bunit == ''

    def compute_max_map(self):
        self.max_map = np.max(self.data,axis=-1)

    def plot_max_map(self,title=None):
        ax,plot = self.mom0like_plot(data=self.max_map,title=title)
        self.set_colorbar(plot=plot,label='max opt depth')
        return ax,plot


class LimeLevelPopOutput():
    '''This output is using the grid defined by LIME itself (Delaunay triangulation etc),
    so its different from the input grid'''
    def __init__(self,filename):
        self.filename = filename
        hdu = fits.open(filename)
        grid = hdu[1].data
        self.x = grid.field('x1')
        self.y = grid.field('x2')
        self.z = grid.field('x3')
        n_species = len(hdu) - 4
        self.levelpops = {}
        self.density = {}
        for i in range(n_species):
            table = hdu[i+4]
            species_name = table.header['MOL_NAME']
            self.levelpops[species_name] = table.data
            self.density[species_name] = grid.field('DENSMOL{:d}'.format(i+1))