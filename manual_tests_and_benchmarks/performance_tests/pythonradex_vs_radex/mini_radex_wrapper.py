#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 18:31:43 2025

@author: gianni
"""

# more convenient since I can use this in several scripts, but have to make sure
# that packing this into a function does not affect performance
# the function should be tested

import time
import os
from scipy import constants

radex_collider_keys = {
    "H2": "H2",
    "para-H2": "p-H2",
    "ortho-H2": "o-H2",
    "e": "e",
    "He": "He",
}
radex_executables = {
    "static sphere": "../../../tests/Radex/bin/radex_static_sphere",
    "LVG slab": "../../../tests/Radex/bin/radex_LVG_slab",
    "LVG sphere": "../../../tests/Radex/bin/radex_LVG_sphere",
}


def run_radex(
    datafilename,
    geometry,
    collider_densities,
    Tkin,
    N,
    width_v,
    input_filepath,
    output_filepath,
):
    start_setup = time.time()
    if os.path.exists(output_filepath):
        os.remove(output_filepath)
    with open(input_filepath, mode="w") as f:
        f.write(datafilename + "\n")
        f.write(f"{output_filepath}\n")
        f.write("0 0\n")  # output all transitions
        f.write(f"{Tkin}\n")
        f.write(f"{len(collider_densities)}\n")
        for collider, density in collider_densities.items():
            f.write(radex_collider_keys[collider] + "\n")
            f.write(f"{density/constants.centi**-3}\n")
        f.write("2.73\n")
        f.write(f"{N/constants.centi**-2}\n")
        f.write(f"{width_v/constants.kilo}\n")
        f.write("0\n")
    end_setup = time.time()
    start_calc = time.time()
    os.system(f"{radex_executables[geometry]} < {input_filepath} > /dev/null")
    end_calc = time.time()
    start_assert = time.perf_counter()
    assert os.path.exists(
        output_filepath
    ), "Radex failed? Remove /dev/null to see radex output"
    end_assert = time.perf_counter()
    return {
        "setup": end_setup - start_setup,
        "calc": end_calc - start_calc,
        "assert": end_assert - start_assert,
    }


if __name__ == "__main__":
    datafilename = "c.dat"
    # datafilename = "co.dat"
    geometry = "static sphere"
    collider_densities = {
        "ortho-H2": 1e4 * constants.centi**-3,
        "para-H2": 5e5 * constants.centi**-3,
    }

    Tkin = 55
    N = 1e15 * constants.centi**-2
    width_v = 2.3 * constants.kilo
    input_filepath = "test_radex_mini_wrapper.inp"
    output_filepath = "test_radex_mini_wrapper.out"
    start = time.perf_counter()
    times = run_radex(
        datafilename=datafilename,
        geometry=geometry,
        collider_densities=collider_densities,
        Tkin=Tkin,
        N=N,
        width_v=width_v,
        input_filepath=input_filepath,
        output_filepath=output_filepath,
    )
    end = time.perf_counter()
    print(times)
    print(f"func time: {end-start}")
