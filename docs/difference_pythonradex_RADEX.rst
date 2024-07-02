Differences between ``pythonradex`` and ``RADEX``
------------------------------------------------------

Different output
^^^^^^^^^^^^^^^^^^^^^^
There is a difference between the outputs of ``RADEX`` and ``pythonradex``. The ``RADEX`` output :math:`T_R` (or the corresponding flux outputs) is intended to be directly compared to telescope data. To be more specifc, from the computed optical depth and excitation temperature, ``RADEX`` first computes :math:`I_\mathrm{tot} = B_\nu(T_\mathrm{ex})(1-e^{-\tau}) + I_\mathrm{bg}e^{-\tau}`, i.e. the total intensity at the line centre that is recorded at the telescope, where :math:`I_\mathrm{bg}` is the background radiation. This is the sum of the radiation from the gas (first term) and the background radiation attenuated by the gas (second term). From this, the observer will subtract the background (or, in other words, the continuum), giving :math:`I_\mathrm{measured} = I_\mathrm{tot} - I_\mathrm{bg} = (B_\nu(T_\mathrm{ex})-I_\mathrm{bg})(1-e^{-\tau})`. The ``RADEX`` output :math:`T_R` is the Rayleigh-Jeans temperature corresponding to :math:`I_\mathrm{measured}`. On the other hand, ``pythonradex`` computes the line flux directly, i.e. the output corresponds simply to the flux emitted by the gas.

.. _sphere_flux_difference:

Different flux for uniform sphere geometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For a given excitation temperature :math:`T_{ex}` and optical depth :math:`\tau_\nu`, ``RADEX`` calculates the flux as

.. math::
    I_\nu = B_\nu(T_{ex})(1-e^{-\tau_\nu})

