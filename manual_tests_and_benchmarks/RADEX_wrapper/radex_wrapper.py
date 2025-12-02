# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:22:03 2016

@author: giannicataldi
"""
from scipy import constants
import os
import sys
import copy


class Interval():

    def __init__(self,min,max):
        assert min <= max, 'invalid inveral boundaries: min={:g}, max={:g}'.format(min,max)
        self.min = min
        self.max = max

    def contains(self,value):
        return self.min <= value <= self.max


class RadexInput():

    def __init__(self,data_filename,frequency_interval,Tkin,
                 coll_partner_densities,T_background,column_density,Delta_v):
        self.data_filename = data_filename
        self.frequency_interval = frequency_interval
        self.Tkin = Tkin
        self.coll_partner_densities = coll_partner_densities
        self.T_background = T_background
        self.column_density = column_density
        self.Delta_v = Delta_v
        script_folderpath = os.getcwd()
        self.input_filepath = os.path.join(script_folderpath,'radex.inp')
        self.output_filepath = os.path.join(script_folderpath,'radex.out')
        allowed_columdensity = Interval(min=1e5/constants.centi**2,
                                        max=1e25/constants.centi**2)
        assert allowed_columdensity.contains(self.column_density),\
              'column density out of allowed range'
        allowed_coll_partner_density = Interval(min=1e-3/constants.centi**3,
                                                max=1e13/constants.centi**3)
        for coll_partner,density in self.coll_partner_densities.items():
            assert allowed_coll_partner_density.contains(density),\
                'coll partner {:s} density out of allowed range'.format(coll_partner)

    def write_input_file(self):
        input_file = open(self.input_filepath,'w')
        input_file.write(self.data_filename+'\n')
        input_file.write(self.output_filepath+'\n')
        #frequency should be written in GHz
        input_file.write('{:f} {:f}\n'.format(self.frequency_interval.min/constants.giga,
                                              self.frequency_interval.max/constants.giga))
        #kinetic gas temperature:
        input_file.write('{:f}\n'.format(self.Tkin))
        input_file.write('{:d}\n'.format(len(self.coll_partner_densities)))
        for coll_partner,density in self.coll_partner_densities.items():
            input_file.write(coll_partner+'\n')
            #density should be in cm-3
            input_file.write('{:f}\n'.format(density*constants.centi**3))
        input_file.write('{:f}\n'.format(self.T_background))
        #column density has to be in cm-2
        input_file.write('{:f}\n'.format(self.column_density*constants.centi**2))
        input_file.write('{:f}\n'.format(self.Delta_v/constants.kilo)) #line width in km/s
        input_file.write('0\n') #run another calculation?
        input_file.close()

    def remove_input_file(self):
        os.remove(self.input_filepath)


class RadexOutput():

    def __init__(self,output_filepath):
        self.output_filepath = output_filepath

    def read(self):
        output_file  = open(self.output_filepath)
        output_lines = output_file.readlines()
        output_file.close()
        if (output_lines[-2].split()[-1] != '(erg/cm2/s)'):
            error_message = "Error: Ambiguous line selection (no line or more than one line)\n"\
                            + "See {:s} to find out".format(self.output_filepath)
            raise RuntimeError(error_message)
        results = output_lines[-1].split()
        output = {'flux':float(results[-1])*constants.erg/constants.centi**2,
                  'tau':float(results[-6]), 'Tex':float(results[-7]),
                  'TR':float(results[-5]),"pop_up":float(results[-4]),
                  "pop_low":float(results[-3])}
        # flux in W/m2; to get the observed flux, one has to divide by 4*pi and
        # multiply by the solid angle of the target; see 
        # https://personal.sron.nl/~vdtak/radex/index.shtml#output
        return output
                

class RadexWrapper():

    max_iter = 500
    fit_tolerance = 1e-2

    def __init__(self,geometry):
        folderpath = os.path.dirname(os.path.realpath(__file__))
        executables = {'static sphere':'radex_static_sphere',
                       'LVG sphere':'radex_LVG_sphere',
                       'LVG slab':'radex_LVG_slab'}    
        exec_paths = {ID:os.path.join(folderpath,f'../../tests/Radex/bin/{ex}') for ID,ex in
                      executables.items()}
        self.exec_path = exec_paths[geometry]

    def compute(self,radex_input):
        radex_input.write_input_file()
        if os.path.exists(radex_input.output_filepath):
            os.remove(radex_input.output_filepath)
        os.system(f'{self.exec_path} < {radex_input.input_filepath} > /dev/null')
        output = RadexOutput(radex_input.output_filepath).read()
        radex_input.remove_input_file()
        #os.remove(radex_input.output_filepath)
        os.remove('./radex.log')
        return output

    def fit_column_density(self,observed_flux,radex_input):
        flux_ratio = 0
        n_iter = 0
        #I create a copy such that the original radex_input object remains
        #unchanged:
        copy_input = copy.deepcopy(radex_input)
        while (flux_ratio > (1+self.fit_tolerance)) or\
              (flux_ratio < (1-self.fit_tolerance)):
            n_iter += 1
            result = self.compute(copy_input)
            flux_ratio = observed_flux/result['flux']
            copy_input.column_density *= flux_ratio
            if n_iter > self.max_iter:
                sys.exit('maximum number of iterations exceeded')
        print('Needed {:d} iterations to find a solution'.format(n_iter))
        return copy_input.column_density


if __name__=='__main__':
    test_freq_interval = Interval(min=50*constants.giga,max=100*constants.giga)
    test_coll_partners = {'H2':1e10}
    test_radex_input = RadexInput(data_filename='hco+.dat',
                                  frequency_interval=test_freq_interval,Tkin=30,
                                  coll_partner_densities=test_coll_partners,
                                  T_background=2.73,column_density=1e18,
                                  Delta_v=1*constants.kilo)
    radex_wrapper = RadexWrapper()
    test_results = radex_wrapper.compute(test_radex_input)
    print('result of test run:')
    for quantity,value in test_results.items():
        print('{:s} = {:g}'.format(quantity,value))

    test_flux = 3.514e-14
    expected_column_density = 1e14
    fitted_column_density = radex_wrapper.fit_column_density(observed_flux=3.514e-14,
                                                             radex_input=test_radex_input)
    print('fitted column density for test_flux = {:g}: {:g} (expected: {:g})'\
          .format(test_flux,fitted_column_density,expected_column_density))
