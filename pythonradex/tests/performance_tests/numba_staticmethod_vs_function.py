#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:23:01 2024

@author: gianni
"""

from numba import jit
import numpy as np

@jit(nopython=True,cache=True)
def function(x):
    return x**2*np.sin(np.cos(x))*np.exp(-x)/x

class TestClass():

    @staticmethod
    @jit(nopython=True,cache=True)
    def stat_method(x):
        return x**2*np.sin(np.cos(x))*np.exp(-x)

    def call_func(self,x):
        return function(x)

    def call_stat_method(self,x):
        return self.stat_method(x)

if __name__ == '__main__':
    import time

    def measure_time(func,args,ID):
        start = time.time()
        func(*args)
        end = time.time()
        print(f'{ID}: {end-start}')

    args = (np.linspace(1,2,5000),)
    for i in range(3):
        print(i)
        test = TestClass()
        #measure_time(func=function,args=args,ID='function')
        measure_time(func=test.call_func,args=args,ID='call function from class')
        measure_time(func=test.call_stat_method,args=args,ID='call stat method from class')
        print('\n')