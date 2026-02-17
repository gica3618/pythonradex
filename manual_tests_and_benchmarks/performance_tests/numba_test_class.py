#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 21:26:51 2024

@author: gianni
"""

# test if it works to use a compiled function inside a class method -> it works!


from numba import jit
import time


@jit(nopython=True)
def square(x):
    return x**2


start = time.time()
square(1)
end = time.time()
print(f"compilation time: {end-start:.2g}")


class TestClass:

    def square_many_times(self):
        for i in range(100):
            square(i)


if __name__ == "__main__":

    def measure_time(func, ID):
        start = time.time()
        output = func()
        end = time.time()
        print(f"{ID}: {end-start:.2g}")
        return output

    for i in range(3):
        print(f"iteration {i}")
        test = measure_time(TestClass, ID="class instance creation")
        measure_time(func=test.square_many_times, ID="square calculation 1")
        measure_time(func=test.square_many_times, ID="square calculation 2")
        print("\n")
