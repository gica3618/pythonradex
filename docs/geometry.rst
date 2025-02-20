.. _geometries:

Cloud geometries
======================

The calculations of the escape probability and the emerging flux depend on the adopted geometry. Below we give an overview of the geometries available in ``pyhonradex``. For all geometries, we assume a homogeneous medium (i.e. constant number density). For convenience, the escape probabilities are expresses as functions of optical depth :math:`\tau_\nu` rather than the absorption coefficient :math:`\alpha_\nu`.

Uniform sphere
----------------------
A homogeneous, static sphere. The escape probability is given in [Osterbrock74]_:

.. math::
    :name: eq:beta_uniform_sphere

    \beta(\tau_\nu) = \frac{3}{2\tau_\nu}\left(1-\frac{2}{\tau_\nu^2}+\left(\frac{2}{\tau_\nu}+\frac{2}{\tau_\nu^2}\right) e^{-\tau_\nu}\right)

where :math:`\tau_\nu` is the optical depth of the diameter of the sphere. The observed flux (in [W/m\ :sup:`2`/Hz]) can also be calculated from [Osterbrock74]_ and is given by:

.. math::

    F_\nu = \frac{2B_\nu(T_\mathrm{ex})\Omega}{\tau_\nu^2}\left(\frac{\tau_\nu^2}{2}-1+(\tau_\nu+1)e^{-\tau_\nu}\right)

where :math:`B_\nu(T_\mathrm{ex})` is the Planck function evaluated at the excitation temperature :math:`T_\mathrm{ex}`, :math:`\Omega` is the solid angle of the source (given by :math:`\Omega=R^2\pi/d^2` with :math:`R` the radius of the sphere and :math:`d` the distance of the source). Note that ``RADEX`` calculates the flux with the formula for a slab geometry, which is incorrect. See :doc:`difference_pythonradex_RADEX` for more details.

Uniform slab
----------------------
A homogeneous, static slab. The escape probability is given by (e.g. [Elitzur92]_):

.. math::

    \beta(\tau_\nu) = \frac{\int_0^1 (1-e^{-\tau_\nu/\mu})\mu\mathrm{d}\mu}{\tau_\nu}

The observed flux (in [W/m\ :sup:`2`/Hz]) is given by:

.. math::
    :name: eq:flux_uniform_slab

    F_\nu = B_\nu(T_\mathrm{ex})(1-e^{-\tau_\nu})\Omega

where :math:`\Omega` is the solid angle of the emitting region.


LVG sphere
-------------------
The Large Velocity Gradient (LVG) approximation is applicable if the characteristic flow velocity along the line of sight is much larger than the local random (e.g. thermal) velocities (e.g. [Scoville74]_, [Elitzur92]_). Here, we consider the model by [Goldreich74]_: A homogeneous sphere with a constant, radial velocity gradient :math:`\mathrm{d}v/\mathrm{d}r=V/R` where :math:`V` is the velocity at the sphere surface and :math:`R` is the radius of the sphere. The escape probability is given by

.. math::
    :name: eq:beta_LVG_sphere

    \beta(\tau) = \frac{1-e^{-\tau}}{\tau}

Note that here, :math:`\tau` is the optical depth of the diameter of the sphere, and is constant in the interval :math:`[-V,V]` (and zero outside). The observed flux can be derived by using an approach similar to [deJong75]_. It is given by

.. math::

    F_\nu = B_\nu(T_\mathrm{ex})(1-e^{-\tau})\Omega\left(1-\frac{v^2}{V^2}\right)

where :math:`v=c(1-\nu/\nu_0)` with :math:`c` the speed of light and :math:`\nu_0` the rest frequency. As in the case of the static sphere, also for the LVG sphere ``RADEX`` incorrectly uses the formula for a slab geometry to calculate the flux (see :doc:`difference_pythonradex_RADEX` for more details).


LVG slab
-------------------
Here we consider a homogeneous slab with a constant velocity gradient :math:`\mathrm{d}v/\mathrm{d}z`, where :math:`z` is the coordinate perpendicular to the slab surface, along the line of sight. The escape probability is given by [Scoville74]_ as

.. math::

    \beta(\tau) = \frac{1-e^{-3\tau}}{3\tau}

Here again, :math:`\tau` is the constant optical depth over the velocity interval with a width given by :math:`\frac{\mathrm{d}v}{\mathrm{d}z}Z` with :math:`Z` the total depth of the slab. The flux is given by the same formula as for the uniform slab.

Geometries emulating ``RADEX``
-------------------------------------
Mainly for test and legacy purposes, ``pythonradex`` offers two additional geometries that emulate the behaviour of ``RADEX``.

Uniform sphere RADEX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This geometry uses the same formula (:ref:`Eq. 1 <eq:beta_uniform_sphere>`) for the escape probability as the uniform sphere, but uses the formula (:ref:`Eq. 2 <eq:flux_uniform_slab>`) for the uniform slab to calculate the flux.


LVG sphere RADEX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The paper describing ``RADEX`` ([vanderTak07]_) says that ``RADEX`` uses the formula given in :ref:`Eq. 3 <eq:beta_LVG_sphere>` to calculate the escape probability of an LVG sphere. However, by inspecting the ``RADEX`` source code, it is found that at least the current version uses a different formula that seems to be based on [deJong80]_. It is given by

.. math::

    \beta(\tau) = \frac{1}{\tau_\nu\sqrt{\ln(\tau_\nu/(2\sqrt{\pi}))}} \qquad \text{if } \tau_\nu\geq 7

and

.. math::

    \beta(\tau) = \frac{4-4e^{-2.34\tau_\nu/2}}{4.68\tau_\nu} \qquad \text{if } \tau_\nu< 7

Thus, the geometry "LVG sphere RADEX" uses this formula. For the flux, it uses the same formula (:ref:`Eq. 2 <eq:flux_uniform_slab>`) as for the uniform slab (despite the spherical geometry).

The line width parameter ``width_v``
------------------------------------------
It is immportant to understand the different interpretations of the input parameter ``width_v`` used by ``pythonradex`` (see :ref:`API of the Cloud class <rad_trans_API>`). For static geometries, this refers to the local emission width (typically the thermal width). ``pythonradex`` allows two kinds of local emission profiles (parameter ``line_profile_type``): "Gaussian" (in which case ``width_v`` refers to the FWHM) and "rectangular". On the other hand, for the LVG geometries ("LVG sphere" and "LVG slab"), ``width_v`` refers to the global velocity width of the cloud. For the "LVG sphere", ``width_v`` is equal to :math:`2V` (with :math:`V` the velocity at the sphere surface). For the "LVG slab", ``width_v`` equals :math:`\mathrm{d}v/\mathrm{d}zZ` (with :math:`Z` the depth of the slab and :math:`\mathrm{d}v/\mathrm{d}z` the constant velocity gradient). For these geometries, the parameter ``line_profile_type`` needs to be set to "rectangular". This ensures that the optical depth is calculated correctly.
