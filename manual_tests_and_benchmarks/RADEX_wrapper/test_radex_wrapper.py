# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:21:13 2017

@author: gianni
"""

import unittest
import sys
import os
cwd = os.getcwd()
sys.path.append(os.path.dirname(cwd))
import radex_wrapper
from scipy import constants
import copy
import numpy as np


class TestUtils(unittest.TestCase):

    def setUp(self):
        freq_interval = radex_wrapper.Interval(min=50*constants.giga,
                                               max=100*constants.giga)
        self.input_kwargs = {'data_filename':'hco+.dat',
                             'frequency_interval':freq_interval,
                             'Tkin':100,'coll_partner_densities':{'H2':1e10,'e':1e9},
                             'T_background':2,'column_density':1e18,
                             'Delta_v':1.5*constants.kilo}
        self.radexinput = radex_wrapper.RadexInput(**self.input_kwargs)

    def test_Interval(self):
        self.assertRaises(AssertionError,radex_wrapper.Interval,min=2,max=1)
        interval = radex_wrapper.Interval(min=-2.4,max=4)
        for included_value in (-2.4,0,2,4):
            self.assertTrue(interval.contains(included_value))
        for excluded_value in (-20,-2.4000001,4.1,10000):
            self.assertFalse(interval.contains(excluded_value))

    def test_column_density_limits(self):
        low_column_dens,high_column_dens = 8e8,2e31
        low_column_dens_kwargs = dict(self.input_kwargs)
        high_column_dens_kwargs = dict(self.input_kwargs)
        low_column_dens_kwargs['column_density'] = low_column_dens
        high_column_dens_kwargs['column_density'] = high_column_dens
        for kwargs in (low_column_dens_kwargs,high_column_dens_kwargs):
            self.assertRaises(AssertionError,radex_wrapper.RadexInput,**kwargs)
        
    def test_coll_partner_density_limits(self):
        low_coll_dens,high_coll_dens = 9e2,1.01e19
        low_coll_dens_kwargs = dict(self.input_kwargs)
        high_coll_dens_kwargs = dict(self.input_kwargs)
        low_coll_dens_kwargs['coll_partner_densities'] = {'H2':low_coll_dens}
        high_coll_dens_kwargs['coll_partner_densities'] = {'H2':high_coll_dens}
        for kwargs in (low_coll_dens_kwargs,high_coll_dens_kwargs):
            self.assertRaises(AssertionError,radex_wrapper.RadexInput,**kwargs)

    def test_file_writting(self):
        self.radexinput.write_input_file()
        self.assertTrue(os.path.exists(self.radexinput.input_filepath))
        input_file = open(self.radexinput.input_filepath)
        lines=input_file.readlines()
        self.assertEqual(len(lines),9+2*len(self.input_kwargs['coll_partner_densities']))
        self.assertEqual(lines[0],self.input_kwargs['data_filename']+'\n')
        self.assertEqual(lines[1],self.radexinput.output_filepath+'\n')
        min_freq = self.input_kwargs['frequency_interval'].min
        max_freq = self.input_kwargs['frequency_interval'].max
        self.assertEqual(lines[2],'{:f} {:f}\n'.format(min_freq/constants.giga,
                                                       max_freq/constants.giga))
        self.assertEqual(lines[3],'{:f}\n'.format(self.input_kwargs['Tkin']))
        n_coll_partners = len(self.input_kwargs['coll_partner_densities'])
        self.assertEqual(lines[4],'{:d}\n'.format(n_coll_partners))
        for i,(coll_partner,density) in enumerate(
                          self.input_kwargs['coll_partner_densities'].items()):
            self.assertEqual(lines[5+2*i],coll_partner+'\n')
            self.assertEqual(lines[6+2*i],'{:f}\n'.format(density*constants.centi**3))
        self.assertEqual(lines[5+2*n_coll_partners],'{:f}\n'.format(
                                          self.input_kwargs['T_background']))
        self.assertEqual(lines[6+2*n_coll_partners],'{:f}\n'.format(
                        self.input_kwargs['column_density']*constants.centi**2))
        self.assertEqual(lines[7+2*n_coll_partners],'{:f}\n'.format(
                                self.input_kwargs['Delta_v']/constants.kilo))
        self.assertEqual(lines[8+2*n_coll_partners],'0\n')
        input_file.close()
        self.radexinput.remove_input_file()
        self.assertFalse(os.path.exists(self.radexinput.input_filepath))

    def test_RadexOutput(self):
        #this is the standard output file coming included in the RADEX download:
        std_output_filepath = './example_output.out'
        std_output = radex_wrapper.RadexOutput(std_output_filepath)
        self.assertRaises(RuntimeError,std_output.read)
        #same as above, but all lines except one deleted:
        singleline_std_output_filepath = './example_singleline_output.out'
        singleline_output = radex_wrapper.RadexOutput(singleline_std_output_filepath)
        read_singleline_output=singleline_output.read()
        self.assertEqual(len(read_singleline_output),4)
        self.assertEqual(read_singleline_output['flux'],
                         1.516e-8*constants.erg/constants.centi**2)
        self.assertEqual(read_singleline_output['tau'],4.69)
        self.assertEqual(read_singleline_output['Tex'],4.507)
        self.assertEqual(read_singleline_output['TR'],1.559)

    def test_radex_wrapper(self):
        #taking the same values as in the example.inp coming with the RADEX
        #download,except freq interval:
        example_input = radex_wrapper.RadexInput(
                           data_filename='hco+.dat',
                           frequency_interval=radex_wrapper.Interval(88*constants.giga,
                                                                     90*constants.giga),
                           Tkin=20,coll_partner_densities={'H2':1e10},
                           T_background=2.73,column_density=1e17,Delta_v=1*constants.kilo)
        print(example_input.output_filepath)        
        test_radex = radex_wrapper.RadexWrapper(geometry='static sphere')
        example_result = test_radex.compute(example_input)
        example_flux = 1.516e-8*constants.erg/constants.centi**2
        self.assertEqual(example_result['flux'],example_flux)
        self.assertEqual(example_result['tau'],4.69)
        self.assertEqual(example_result['Tex'],4.507)
        self.assertEqual(example_result['TR'],1.559)
        modified_example_input = copy.deepcopy(example_input)
        modified_example_input.column_density = 1e19
        assert example_input.column_density == 1e17
        fit_column_density = test_radex.fit_column_density(
                                      observed_flux=example_flux,
                                      radex_input=modified_example_input)
        self.assertTrue(np.isclose(fit_column_density,1e17,
                                   rtol=test_radex.fit_tolerance,atol=1e14))
        test_input = copy.deepcopy(example_input)
        test_input.coll_partner_densities['e'] = 1e10
        test_input.T_background = 10
        test_input.column_density = 1e15
        test_output = test_radex.compute(test_input)
        fit_input = copy.deepcopy(test_input)
        for inp_column_dens in (1e15,1e17,1e19,1e21):
            fit_input.column_density = inp_column_dens
            test_fit = test_radex.fit_column_density(observed_flux=test_output['flux'],
                                                     radex_input=fit_input)
            self.assertTrue(np.isclose(test_fit,test_input.column_density,
                                       rtol=test_radex.fit_tolerance,
                                       atol=1e-2*test_input.column_density))