Differences between ``pythonradex`` and ``RADEX``
------------------------------------------------------

Overlapping lines and continuum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``pythonradex`` is able to handle excitation effects of overlapping lines (i.e. lines that are so close in frequency that photons emitted from one line can be absorbed by another line). ``pythonradex`` is also able to include effects of an internal radiation field specified by the user (typically arising from dust that is mixed with the gas). Both of these effects are not included in ``RADEX``.

Programming language and performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
While ``RADEX`` is written in Fortran, ``pythonradex`` is written in python. Fortran is generally much faster than python. Still, in the typical use case of calculating several models over a grid of parameters (e.g. column density, temperature), ``pythonradex`` can outperform RADEX by a factor of a few, depending on the input parameters. This is achieved by an architecture that avoids unnecessary or repeating calculations as much as possible and just-in-time compilation using the `numba <https://numba.readthedocs.io>`_ package.

Different output
^^^^^^^^^^^^^^^^^^^^^^
There is a difference between the outputs of ``RADEX`` and ``pythonradex``. The ``RADEX`` output :math:`T_R` (or the corresponding flux outputs) is intended to be directly compared to telescope data. To be more specific, from the computed optical depth and excitation temperature, ``RADEX`` first computes :math:`I_\mathrm{tot} = B_\nu(T_\mathrm{ex})(1-e^{-\tau_\nu}) + I_\mathrm{bg}e^{-\tau_\nu}`, i.e. the total intensity at the line centre that is recorded at the telescope, where :math:`I_\mathrm{bg}` is the background radiation. This is the sum of the radiation from the gas (first term) and the background radiation attenuated by the gas (second term). From this, ``RADEX`` assumes the observer has subtracted the background, giving :math:`I_\mathrm{measured} = I_\mathrm{tot} - I_\mathrm{bg} = (B_\nu(T_\mathrm{ex})-I_\mathrm{bg})(1-e^{-\tau_\nu})`. The ``RADEX`` output :math:`T_R` is the Rayleigh-Jeans temperature corresponding to :math:`I_\mathrm{measured}`. This output may or may not be the right quantity to be compared to observations. For example, it is almost certainly not appropriate to be compared to interferometric data. On the other hand, ``pythonradex`` outputs the pure line flux without any background subtraction, i.e. the output corresponds simply to the flux emitted by the gas (for a slab geometry, the fluxes would be based on the intensity given simply by :math:`B_\nu(T_\mathrm{ex})(1-e^{-\tau_\nu}))`. This allows the user to decide how the flux should be compared to observations, which depends on the telescope etc.

.. _sphere_flux_difference:

Different flux for spherical geometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For a given excitation temperature :math:`T_{ex}` and optical depth :math:`\tau_\nu`, ``RADEX`` calculates the flux as

.. math::
    I_\nu = B_\nu(T_{ex})(1-e^{-\tau_\nu})

for all geometries. However, this expression is only valid for slab geometries, but not for spherical geometries ("uniform sphere" and "LVG sphere"). On the other hand, ``pythonradex`` uses the correct formulae for spherical geometries (see the :ref:`section about geometries<geometries>` for more details). This can for example be demonstrated by considering the optically thin limit. In this limit, each photon escapes from the cloud. The total flux (in [W/m\ :sup:`2`]) is then simply given by

.. math::
    F_\mathrm{thin} = V_\mathrm{sphere}nx_2A_{21}\Delta E \frac{1}{4\pi d^2}

where :math:`V_\mathrm{sphere}=\frac{4}{3}R^3\pi` is the volume of the sphere, :math:`n` the constant number density, :math:`x_2` the fractional level population of the upper level, :math:`A_{21}` the Einstein coefficient, :math:`\Delta E` the energy difference between the upper and lower level, and :math:`d` the distance of the source. ``pythonradex`` correctly reproduces this limiting case. On the other hand, ``RADEX`` overestimates the optically thin flux by a factor 1.5.
