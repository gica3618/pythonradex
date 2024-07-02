Theory
=========

Basic radiative transfer
-------------------------------

We briefly discuss basic theory of radiative transfer that is relevant for ``pyhonradex``. A more detailed discussion can for example be found in [Rybicki04]_.

The radiation field in every point of space can be described by the specific intensity :math:`I_{\nu}`, defined as the energy radiated per unit of time, surface, frequency and solid angle, i.e., :math:`I_{\nu}` has units of W/m\ :sup:`2`\ /sr/Hz. The differential equation describing the change of the specific intensity along a spatial coordinate :math:`s` is given by

.. math::
    \frac{\mathrm{d}I_\nu}{\mathrm{d}s} = -\alpha_\nu I_\nu + j_\nu

Here, :math:`\alpha_\nu` is the absorption coefficient in m\ :sup:`-1`. It describes how much is removed from the beam per unit length. On the other hand, the emission coefficient :math:`j_\nu` is the energy emitted per unit time, solid angle, volume and frequency. Defining the optical depth as :math:`\mathrm{d}\tau_\nu=\alpha_\nu\mathrm{d}s`, we can rewrite the equation as

.. math::
    \frac{\mathrm{d}I_\nu}{\mathrm{d}\tau_\nu} = -I_\nu + S_\nu

with the source function :math:`S_\nu=\frac{j_\nu}{\alpha_\nu}`. In general, the goal of radiative transfer is to solve this equation. For example, for a uniform medium (the emission and absorption coefficients are the same everywhere) as assumed for ``pythonradex``, the solution reads :math:`I_\nu=I_\nu(0)e^{-\tau_\nu}+S_\nu(1-e^{-\tau_\nu})`.

Gas emission
--------------
Next, let's consider radiation from a gas. An atom can spontaneously emit a photon when it transits from an upper to a lower energy level. The transition rate is given by the Einstein coefficient for spontaneous emission, :math:`A_{21}`, in units of s\ :sup:`-1`. Thus, we can write the emission coefficient of the gas as

.. math::
    j_\nu = \frac{h\nu_0}{4\pi}n_2A_\mathrm{21}\phi_\nu

where :math:`h` is the Planck constant, :math:`\nu_0` is the central frequency of the transition, :math:`n_2` is the number density of atoms in the upper level of the transition and :math:`\phi_\nu` is the normalised line profile (i.e. :math:`\int\phi_\nu\mathrm{d}\nu=1` and :math:`\phi_\nu` describes how the energy is distributed over frequency), for example a Gaussian. Photons can also be absorbed with a transition from a lower to an upper energy level. This process is parametrised by the Einstein :math:`B_{12}` coefficient. Defining the mean intensity :math:`J_\nu` as the follwing mean over solid angle:

.. math::
    J_\nu = \frac{1}{4\pi}\int I_\nu \mathrm{d}\Omega

and :math:`\bar{J}=\int J_\nu\phi_\nu\mathrm{d}\nu` the mean over the line profile, the transition rate for absorption is :math:`B_{12}\bar{J}`, i.e. the transition rate is proportional to the flux of incoming photons. There is a third process, called stimulated emission, that results in the *emission* of a photon (and a transition from an upper to a lower level) when the atom interacts with an incoming photon. The rate for stimulated emission is :math:`B_{21}\bar{J}`. The absorption coefficient can be then be written [Rybicki04]_

.. math::
    \alpha_\nu = \frac{h\nu_0}{4\pi}(n_1B_{12}-n_2B_{21})\phi_\nu

with :math:`n_1` the number density of atoms in the lower level. Thus, stimulated emission is treated as 'negative absorption'.

Therefore, in order to compute the emission and absorption coefficients and solve the radiative transfer equation, we need to know the level population, i.e. the fraction of atoms occupying the different energy levels. Actually, using relations between the Einstein coefficients [Rybicki04]_, it can easily be shown that the source function can be written as

.. math::
    :name: eq:source_function

    S_\nu=\frac{A_{21}n_2}{n_1B_{12}-n_2B_{21}}=B_\nu(T_\mathrm{ex})

where :math:`B_\nu(T)` is the blackbody radiation field (Planck's law) and :math:`T_\mathrm{ex}` is the excitation temperature, defined as

.. math::
    \frac{n_2}{n_1}=\frac{g_2}{g_1}\exp\left(-\frac{h\nu_0}{kT_\mathrm{ex}}\right)

with :math:`k` the Boltzmann constant and :math:`g_1` and :math:`g_2` the statistical weights of the lower and upper level respectively. The excitation temperature is thus the temperatue we need to plug into the Boltzmann equation to get the observed level ratio. In LTE, the kinetic temperature equals the excitation temperature, and the levels are indeed populated according to the Boltzmann distribution. But in non-LTE, this is not true, and the excitation temperature is different from the kinetic temperature.

In summary, we need to know the level populations (i.e. the excitation temperature) in order to compute the radiation field. To do this, we first need to consider another process that can populate and depopulate the energy levels of an atom: collisions, for example with hydrogen or electrons. The rate of collision-induced transitions between two levels :math:`i` and :math:`j` is given by :math:`C_{ij}=K_{ij}(T_\mathrm{kin})n_\mathrm{col}` where :math:`K_{ij}(T_\mathrm{kin})` is the collision rate coefficient in m\ :sup:`3`\ s\ :sup:`-1` and :math:`n_\mathrm{col}` is the number density of the collision partner. The collision rate coefficient in general depends on the kinetic temperature of the collision partner. If several collision partners are present, the total rate is simply the sum of the individual rates.

We can now write down the equations of statistical equilibrium (SE) that determine the level population. In SE, we assume that processes that populate a level are balanced by processes that depopulate it. Thus, for every level :math:`i`, we write

.. math::
    :name: eq:SE

    \frac{\mathrm{d}x_i}{\mathrm{d}t} = \sum_{j>i}(x_jA_{ji}+(x_jB_{ji}-x_iB_{ij})\bar{J}_{ji}) - \sum_{j<i}(x_iA_{ij}+(x_iB_{ij}-x_jB_{ji})\bar{J}_{ij}) + \sum_{j\neq i}(x_jC_{ji}-x_iC_{ij}) = 0


where :math:`x_k=\frac{n_k}{n}` is the fractional population of level :math:`k`. In the above equation, the positive terms populate the level, while the negative terms depopulate the level.

The level populations can be computed by solving this linear system of equations. But there is a problem: we see that to solve for the level populations, we need to know the radiation field :math:`\bar{J}`. This is a fundamental issue in radiative transfer: to compute the radiation field, we need to know the level population. But in order to compute the level population, we need to know the radiation field.

Escape probability
---------------------

One way to solve this problem is to use an escape probability method to decouple the computation of the level population from the computation of the radiation field. We consider the probability :math:`\beta` of a newly created photon to escape the cloud. This probability depends on the geometry of the cloud and the optical depth. If the cloud is completely optically thick (:math:`\beta\approx 0`), we expect the radiation field to equal the source function. Thus, we write :math:`J_\nu=(1-\beta(\tau_\nu))S_\nu=(1-\beta(\tau_\nu))B_\nu(T_\mathrm{ex})`. If we plug the corresponding expression for :math:`\bar{J}` into the SE equations, they become independent of the radiation field and can be solved, because :math:`\tau_\nu` and :math:`T_\mathrm{ex}` only depend on the level population.

In practice, an iterative approach is used to solve the SE equations: one makes a first guess of the level populations and computes the corresponding escape probability, which is used to compute a new solution of the SE equations. This is repeated until convergence. Finally, the converged level population is used to compute the emitted flux and the radiative transfer problem is solved.

An external radiation field :math:`I_\mathrm{ext}` can also contribute to the excitation of the atoms. This is easily incorporated in the calculation by adding a term :math:`\beta I_\mathrm{ext}` to :math:`J_\nu`.

As mentioned above, the espace probability depends on the geometry of the emitting region. For example, for a uniform sphere, the escape probability is given in [Osterbrock74]_:

.. math::

    \beta(\tau_\nu) = \frac{3}{2\tau_\nu}\left(1-\frac{2}{\tau_\nu^2}+\left(\frac{2}{\tau_\nu}+\frac{2}{\tau_\nu^2}\right) e^{-\tau_\nu}\right)

where :math:`\tau_\nu` is the optical depth of the diameter of the sphere. ``pyhonradex`` allows the user to choose among the following geometries (see also the :ref:`API of the Cloud class <rad_trans_API>`):

* 'uniform sphere': A uniform sphere, using the equations for escape probability and flux by [Osterbrock74]_.
* 'uniform sphere RADEX': The escape probability is computed in the same way as for 'uniform sphere', but the flux is computed as in RADEX (see more details :ref:`here <sphere_flux_difference>`)
* 'uniform slab': A semi-infinite uniform slab, using the escape probability given by [Elitzur92]_.
* 'LVG slab': A slab for which the large velocity gradient (LVG) approximation applies, see [Scoville74]_.
* 'LVG sphere': A sphere for which the large velocity gradient (LVG) approximation applies, using the equation by [Elitzur92]_.
* 'LVG sphere RADEX': Using the same equation for the escape probability as RADEX, which is based on [deJong80]_.


Iteration schemes
-------------------------

Standard LAMDA Iteration (LI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As mentioned above, the equations of statistical equilibrium (:ref:`Eq. 2 <eq:SE>`) are solved iteratively. An initial guess of the level population (and thus excitation temperatures and optical depths) allows us to compute

.. math::
    :name: eq:Jbar

    \bar{J} = \beta(\tau) I_\mathrm{ext}+(1-\beta(\tau))B_\nu(T_{ex})

With :math:`\bar{J}` known, the statistical equilibrium can be solved to obtain an updated level population. Iteration is continued until convergence is reached. This scheme is referred to as LAMDA Iteration (LI; see the lectures notes of Dullemond_). It has the issue that convergence can be very slow, in particular for optically thick emission.

Accelerated LAMDA Iteration (ALI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The LI scheme can be modified to accelerate convergence. The modified scheme is known as Accelerated Lambda Iteration (ALI). Details can be found in [Rybicki91]_ and the lectures notes of Dullemond_, sections 4.4 and 7.8--7.10. In our case, the ALI scheme is found by expressing :math:`B_\nu(T_{ex})` in :ref:`Eq. 3 <eq:Jbar>` in terms of the *new* level population, rather than calculating it from the old population. By using :ref:`Eq. 1 <eq:source_function>`, the equations of statistical equilibrium when using ALI become

.. math::
    \frac{\mathrm{d}x_i}{\mathrm{d}t} = \sum_{j>i}(x_jA_{ji}\beta+(x_jB_{ji}-x_iB_{ij})\beta I_\mathrm{ext}) - \sum_{j<i}(x_iA_{ij}\beta+(x_iB_{ij}-x_jB_{ji})\beta I_\mathrm{ext}) + \sum_{j\neq i}(x_jC_{ji}-x_iC_{ij}) = 0

``pyhonradex`` allows the user to choose between LI and ALI, but ALI is strongly recommended.

Ng-acceleration
------------------------

``pyhonradex`` employs Ng-acceleration [Ng74]_ to further accelerate convergence. Ng-acceleration uses the last three iteration steps to compute the next step. See the lecture notes by Dullemond_ (Sect. 4.4.7) for more details.


.. _Dullemond: http://www.ita.uni-heidelberg.de/~dullemond/lectures/radtrans_2012/
