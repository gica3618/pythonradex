Geometry
===============

Escape probability
--------------------------

For example, for a uniform sphere, the escape probability is given in [Osterbrock74]_:

.. math::

    \beta(\tau_\nu) = \frac{3}{2\tau_\nu}\left(1-\frac{2}{\tau_\nu^2}+\left(\frac{2}{\tau_\nu}+\frac{2}{\tau_\nu^2}\right) e^{-\tau_\nu}\right)

where :math:`\tau_\nu` is the optical depth of the diameter of the sphere. ``pyhonradex`` allows the user to choose among the following geometries (see also the :ref:`API of the Cloud class <rad_trans_API>`):

* 'uniform sphere': A uniform sphere, using the equations for escape probability and flux by [Osterbrock74]_.
* 'uniform sphere RADEX': The escape probability is computed in the same way as for 'uniform sphere', but the flux is computed as in RADEX (see more details :ref:`here <sphere_flux_difference>`)
* 'uniform slab': A semi-infinite uniform slab, using the escape probability given by [Elitzur92]_.
* 'LVG slab': A slab for which the large velocity gradient (LVG) approximation applies, see [Scoville74]_.
* 'LVG sphere': A sphere for which the large velocity gradient (LVG) approximation applies, using the equation by [Elitzur92]_.
* 'LVG sphere RADEX': Using the same equation for the escape probability as RADEX, which is based on [deJong80]_.


Observed flux
--------------------------
