# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:47:19 2017

@author: gianni
"""

from pythonradex import LAMDA_file
from pathlib import Path
from scipy import constants
import numpy as np

here = Path(__file__).resolve().parent
data_folder = here / "LAMDA_files"

filepaths = [
    data_folder / "co.dat",
    data_folder / "co_with_additional_comments.dat",
]

lamda_data = [LAMDA_file.read(filepath) for filepath in filepaths]


def test_levels():
    test_level_index = 10
    for i in range(len(filepaths)):
        levels = lamda_data[i]["levels"]
        assert len(levels) == 41
        test_level = levels[test_level_index]
        assert test_level.g == 21
        assert np.isclose(
            test_level.E / (constants.h * constants.c),
            211.404098498 / constants.centi,
            atol=0,
        )
        assert test_level.index == test_level_index


def test_rad_transitions():
    test_trans_index = 20
    # nu0_in_file = 2413.9171130 * constants.giga
    for i in range(len(filepaths)):
        rad_transitions = lamda_data[i]["radiative transitions"]
        assert len(rad_transitions) == 40
        test_trans = rad_transitions[test_trans_index]
        assert test_trans.up.index == 21
        assert test_trans.low.index == 20
        assert test_trans.A21 == 8.833e-4
        # assert np.isclose(test_trans.Delta_E, nu0_in_file * constants.h, atol=0)
        # test_trans = rad_transitions[test_trans_index]
        # assert np.isclose(test_trans.nu0, nu0_in_file, atol=0)


def test_frequencies():
    test_trans_indices = (0, 1, 10, 20, -1)
    nu0_in_file = (
        np.array((115.2712018, 230.5380000, 1267.0144860, 2413.9171130, 4564.0056399))
        * constants.giga
    )
    for data in lamda_data:
        freqs = data["frequencies"]
        assert len(freqs) == len(data["radiative transitions"])
        for i, nu0 in zip(test_trans_indices, nu0_in_file):
            assert freqs[i] == nu0


def test_coll_transitions():
    for i in range(len(filepaths)):
        coll_transitions = lamda_data[i]["collisional transitions"]
        assert len(coll_transitions) == 2
        assert "para-H2" in coll_transitions
        assert "ortho-H2" in coll_transitions
        test_coll_trans_index_H2 = 100
        test_coll_trans_pH2 = coll_transitions["para-H2"][test_coll_trans_index_H2]
        Tkin_data_pH2 = np.array(
            (
                2,
                5,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                150,
                200,
                300,
                400,
                500,
                600,
                700,
                750,
                800,
                900,
                1000,
                2000,
                3000,
            )
        )
        assert np.all(test_coll_trans_pH2.Tkin_data == Tkin_data_pH2)
        assert test_coll_trans_pH2.up.index == 14
        assert test_coll_trans_pH2.low.index == 9
        coeffs_pH2 = (
            np.array(
                (
                    1.484e-11,
                    1.507e-11,
                    1.548e-11,
                    1.579e-11,
                    1.616e-11,
                    1.677e-11,
                    1.751e-11,
                    1.831e-11,
                    1.910e-11,
                    1.987e-11,
                    2.060e-11,
                    2.129e-11,
                    2.420e-11,
                    2.643e-11,
                    2.975e-11,
                    3.224e-11,
                    3.429e-11,
                    3.605e-11,
                    3.758e-11,
                    3.827e-11,
                    3.893e-11,
                    4.013e-11,
                    4.121e-11,
                    4.756e-11,
                    4.901e-11,
                )
            )
            * constants.centi**3
        )
        _, pH2_K21 = test_coll_trans_pH2.coeffs(Tkin_data_pH2)
        assert np.allclose(coeffs_pH2, pH2_K21, atol=0)
        test_coll_trans_index_oH2 = 200
        test_coll_trans_oH2 = coll_transitions["ortho-H2"][test_coll_trans_index_oH2]
        Tkin_data_oH2 = Tkin_data_pH2
        assert np.all(test_coll_trans_oH2.Tkin_data == Tkin_data_oH2)
        assert test_coll_trans_oH2.up.index == 20
        assert test_coll_trans_oH2.low.index == 10
        coeffs_oH2 = (
            np.array(
                (
                    4.949e-13,
                    5.627e-13,
                    6.206e-13,
                    6.794e-13,
                    7.309e-13,
                    7.881e-13,
                    8.533e-13,
                    9.263e-13,
                    1.006e-12,
                    1.093e-12,
                    1.184e-12,
                    1.280e-12,
                    1.807e-12,
                    2.369e-12,
                    3.471e-12,
                    4.461e-12,
                    5.324e-12,
                    6.076e-12,
                    6.740e-12,
                    7.046e-12,
                    7.337e-12,
                    7.883e-12,
                    8.389e-12,
                    1.229e-11,
                    1.465e-11,
                )
            )
            * constants.centi**3
        )
        _, oH2_K21 = test_coll_trans_oH2.coeffs(Tkin_data_oH2)
        assert np.allclose(coeffs_oH2, oH2_K21, atol=0)


# wrong_freq_datafilepath = data_folder / "c_wrong_frequencies.dat"

# def test_freq_reading_sanity_check():
#     with pytest.warns(UserWarning):
#         LAMDA_file.read(datafilepath=wrong_freq_datafilepath)


# @pytest.mark.filterwarnings("ignore:Delta_E and nu0 are inconsistent",
#                             "ignore:.*inconsistent with level energies.*")
def test_reading_all_LAMDA_files():
    lambda_files = [
        f for f in data_folder.iterdir() if f.is_file() and f.suffix == ".dat"
    ]
    # print([f.name for f in lambda_files])
    assert len(lambda_files) > 0
    for f in lambda_files:
        # if f == wrong_freq_datafilepath:
        #     continue
        # print(f)
        LAMDA_file.read(f)
