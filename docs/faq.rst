FAQ
==========

How should negative optical depth be interpreted?
----------------------------------------------------------

As discussed by [vanderTak07]_, for certain parameters ("especially low density and/or strong radiation field", to quote [vanderTak07]_), negative optical depth can arise due to a level population inversion. This requires non-local treatement of the radiative transfer, which is not possible with ``pyhonradex``. The results should still be valid, though less accurate, for optical depths only slightly negative (:math:`\tau_\nu\gtrsim-0.1`). However, the results for more strongly negative optical depths should be ignored ([vanderTak07]_). ``pyhonradex`` provides an option to print a warning if the solution contains negative optical depths using the ``warn_negative_tau`` parameter of the :ref:`Cloud class <rad_trans_API>`.

I get an "Underlying object has vanished" error
------------------------------------------------------
This probably indicates a problem with the caching of the ``numba``-compiled functions used by ``pyhonradex``. Deleting the __pycache__ folder should solve the issue.
