API
=====

.. _rad_trans_API:

Solving the radiative transfer.
------------------------------------
The core of ``pyhonradex`` is the Cloud class which is used to solve the radiative transfer.

.. autoclass:: pythonradex.radiative_transfer.Cloud
    :members: __init__, set_parameters, solve_radiative_transfer, fluxes, tau_nu, spectrum, print_results 

.. _read_LAMDA_API:

Reading LAMDA files
-------------------------
``pyhonradex`` provides a convenient function in the ``LAMDA_file`` module to read files from the LAMDA database:

.. autofunction:: pythonradex.LAMDA_file.read

.. _atomic_transition_API:

Representation of molecular levels and transitions
---------------------------------------------------------
``pyhonradex`` contains a number of classes to represent molecular data such as the energy levels and transitions between these levels. The molecular data is typically read from a `LAMDA file <https://home.strw.leidenuniv.nl/~moldata>`_.

.. autoclass:: pythonradex.atomic_transition.Level
    :members:

.. autoclass:: pythonradex.atomic_transition.RadiativeTransition
    :members: Tex

.. autoclass:: pythonradex.atomic_transition.EmissionLine
    :members: Tex,tau_nu0,tau_nu

.. autoclass:: pythonradex.atomic_transition.CollisionalTransition
    :members: Tex,coeffs

.. _Molecule_API:

.. autoclass:: pythonradex.molecule.Molecule
    :members: __init__, LTE_level_pop

.. autoclass:: pythonradex.molecule.EmittingMolecule
    :members: __init__, get_tau_nu0, get_tau_nu0_LTE, get_Tex

.. _line_profiles_API:

Representation of line profiles
---------------------------------
``pyhonradex`` allows two types of line profiles: Gaussian and rectangular. These are represented by the following classes.

.. autoclass:: pythonradex.atomic_transition.GaussianLineProfile
    :members: phi_nu,phi_v
    
.. autoclass:: pythonradex.atomic_transition.RectangularLineProfile
    :members: phi_nu,phi_v

.. _helpers_API:

Convenience functions
--------------------------
The ``helpers`` module provides a number of convenience functions.

.. automodule:: pythonradex.helpers
    :members: B_nu,generate_CMB_background,zero_background,FWHM2sigma
