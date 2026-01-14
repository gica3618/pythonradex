Theory
=========

Basic radiative transfer
-------------------------------

We briefly discuss basic theory of radiative transfer that is relevant for ``pyhonradex``. A more detailed discussion can for example be found in [Rybicki04]_ or [Rybicki91]_.

The radiation field in every point of space can be described by the specific intensity :math:`I_{\nu}`, defined as the energy radiated per unit of time, surface, frequency and solid angle, i.e., :math:`I_{\nu}` has units of W/m\ :sup:`2`\ /sr/Hz. The differential equation describing the change of the specific intensity along a spatial coordinate :math:`s` is given by

.. math::
    \frac{\mathrm{d}I_\nu}{\mathrm{d}s} = -\alpha_\nu I_\nu + j_\nu

Here, :math:`\alpha_\nu` is the absorption coefficient in m\ :sup:`-1`. It describes how much is removed from the beam per unit length. On the other hand, the emission coefficient :math:`j_\nu` is the energy emitted per unit time, solid angle, volume and frequency. Defining the optical depth as :math:`\mathrm{d}\tau_\nu=\alpha_\nu\mathrm{d}s`, we can rewrite the equation as

.. math::
    \frac{\mathrm{d}I_\nu}{\mathrm{d}\tau_\nu} = -I_\nu + S_\nu

with the source function :math:`S_\nu=\frac{j_\nu}{\alpha_\nu}`. Our basic goal is to solve this equation. For example, for a uniform medium (the emission and absorption coefficients do not depend on position) the solution reads :math:`I_\nu=I_\nu(0)e^{-\tau_\nu}+S_\nu(1-e^{-\tau_\nu})`. In summary, we need to know the emission coefficient and the absorption coefficient to solve the radiative transfer.

Now considering a gas, the absorption coefficient for a transition between energy levels :math:`l` and :math:`l'` (where :math:`E_l>E_{l'}`, hereafter indicated by :math:`l\succ l'`) of a molecule is given by [Rybicki04]_

.. math::
    \alpha_{ll'} = \frac{h\nu}{4\pi}(n_{l'}B_{l'l}-n_lB_{ll'})\phi_{ll'}(\nu)

Here :math:`h` is the Planck constant, :math:`n_l` the number density of molecules in energy level :math:`l`, :math:`B_{l'l}` and :math:`B_{ll'}` are the Einstein coefficients for absorption and stimulated emission respectively,  and :math:`\phi` is the normalised line profile (i.e. :math:`\int\phi(\nu)\mathrm{d}\nu=1` and :math:`\phi` describes how the energy is distributed over frequency).

The emission coefficient is given by (where again :math:`l\succ l'`)

.. math::
    j_{ll'} = \frac{h\nu}{4\pi}n_lA_{ll'}\phi_{ll'}(\nu)

with :math:`A_{ll'}` the Einstein coefficient for spontaneous emission.

The total absorption and emission coefficients are simply given by the sum over all transitions, plus contributions from the dust continuum (:math:`\alpha_c` and :math:`j_c`):

.. math::
    \alpha_\nu = \sum_{l\succ l'}\alpha_{ll'}(\nu) + \alpha_c(\nu)

    j_\nu = \sum_{l\succ l'}j_{ll'}(\nu) + j_c(\nu)

We see that we need to know the fractional level population :math:`n_l` to calculate :math:`\alpha_\nu` and :math:`j_\nu`, which are needed to solve the radiative transfer. How can we do that? There are two kinds processes that can excite or de-excite a molecular level: radiative processes (emission or absorption of photons) or collisions. If the density of colliders (e.g. H\ :sub:`2`\  or electrons) is high enough, the energy levels become thermalised and we are in local thermodynamic equilibrium (LTE). In this case, the level population is simply given by the Boltzmann distribution:

.. math::
    n_l = n\frac{e^{-E_l/(kT_\mathrm{kin})}}{Q}

where :math:`Q=\sum_{l'} e^{-E_{l'}/(kT_\mathrm{kin})}` is the partition function, :math:`T_\mathrm{kin}` is the kinetic temperature of the gas, and :math:`n` is the number density of the molecule. Thus, in the LTE case, the radiative transfer can be solved easily. In the more general non-LTE case, the level population is not known a priori and needs to be calculated. In that case, the level population is often characterised by an excitation temperature :math:`T_\mathrm{ex}` defined by

.. math::
    \frac{n_l}{n_{l'}} = \frac{g_l}{g_{l'}}e^{-\Delta E/(kT_\mathrm{ex})}

where :math:`g` is the statistical weight and :math:`\Delta E` the energy difference between levels :math:`l` and :math:`l'` (note that in LTE, :math:`T_\mathrm{kin}=T_\mathrm{ex}`).

Now, how can we calculate the level populations if LTE does not apply? We assume *statistical equilibrium* (SE). In other words, we assume that the rate of processes that populates a level equals the rate of processes that depopulates a level:

.. math::
    \sum_{l'}n_{l'}(C_{l'l}+R_{l'l}) = \sum_{l'}n_l(C_{ll'}+R_{ll'})

Here :math:`C_{l'l}` and :math:`R_{l'l}` are the rates per volume of collisional and radiative transitions respectively, from level :math:`l'` to level :math:`l`. Thus, the left-hand side is the total rate of transitions into level :math:`l`, while the right-hand side is the total rate of transitions out of level :math:`l`. By writing down the statistical equilibrium for each level and solving the system of equations, the level populations can be calculated and the radiative transfer solved.

However, there is a problem: while the collisional rates :math:`C_{l'l}` are known, the radiative rates depend, as one might expect, on the radiation field. If :math:`l\succ l'`, we have

.. math::
    R_{ll'} = A_{ll'} + B_{ll'}\bar{J}

while if :math:`l\prec l'`

.. math::
    R_{ll'} = B_{ll'}\bar{J}

Here, :math:`\bar{J}` is the radiation field averaged over frequency and solid angle: :math:`\bar{J}=\frac{1}{4\pi}\int I_\nu\phi_{ll'}(\nu)\mathrm{d}\Omega\mathrm{d}\nu`. Thus, in order to solve the SE equations, we need to know the radiation field, which is what we were after in the first place... In order to solve this chicken and egg problem, we need to adopt an iterative technique.

Accelerated Lambda Iteration (ALI)
----------------------------------------------
For the following discussion, we introduce the Lamda Operator, which essentially is a way of writing down the formal solution of the radiative transfer:

.. math::
    I_\nu = \Lambda_\nu[S_\nu]

So, the Lambda operator computes the radiation field :math:`I_\nu` for a given source function :math:`S_\nu=\frac{j_\nu}{\alpha_\nu}`, the latter being completely determined by the level population (and the known contribution from the dust continuum). The simplest iteration scheme (the so-called Lambda Iteration scheme) is very straightforward: one starts with an initial guess for the level population and solves the radiative transfer (as formalised by the above equation). The solution :math:`I_\nu` is then inserted into the statistical equilibrium equations, resulting in an updated level population. This procedure is repeated until convergence is established.

However, the Lambda iteration scheme can suffer from extremely slow convergence in optically thick systems (see e.g. the lecture notes by Dullemond_ or [Rybicki91]_). An alternative scheme, known as Accelerated Lambda Iterations (ALI), provides much better convergence. The idea is to introduce an approximate Lambda operator :math:`\Lambda^*` and to write

.. math::
    I_\nu = \Lambda^*_\nu[S_\nu] + (\Lambda_\nu-\Lambda^*_\nu)[S_\nu^\dagger]

where the :math:`\dagger` indicates quantities from the previous iteration. This is inserted into the equations of statistical equilibrium, which can then be solved for an updated level population (and thus updated :math:`S_\nu`). See for example [Rybicki91]_ or [Hubeny03]_ for more details about ALI.

The ALI method by Rybicki & Hummer (1992)
----------------------------------------------------
``pythonradex`` implements a variation of the ALI scheme presented by [Rybicki92]_. The method is capable of treating overlapping lines and full continuum. In contrast, ``RADEX`` can only treat non-overlapping lines without continuum.

``pythonradex`` implements the "Full preconditioning strategy" presented in section 2.3 of [Rybicki92]_. Instead of a Lambda operator, the method presented in [Rybicki92]_ uses a Psi operator that acts on the emission coefficient:

.. math::
    I_\nu = \Psi[j_\nu]

The approximate iteration scheme is then based on :math:`I_\nu=\Psi^*_\nu[j_\nu] + (\Psi_\nu-\Psi^*_\nu)[j_\nu^\dagger]`, which is inserted into the equations of statistical equilibrium.

Escape probability
-----------------------------
We still need to specify the formal solution of the radiative transfer we adopt via the operator :math:`\Psi_\nu`. Same as ``RADEX``, we use an escape probability method. We consider the probability :math:`\beta` of a newly created photon to escape the source. This probability depends on the geometry of the source and the absorption coefficient (or, equivalently, optical depth). If the source is completely optically thick (:math:`\beta\approx 0`), we expect the radiation field to equal the source function :math:`S_\nu=\frac{j_\nu}{\alpha_\nu}`. Thus, we write 

.. math::
    I_\nu = \Psi[j_\nu] = \beta(\alpha_\nu^\dagger) I_\mathrm{ext} + (1-\beta(\alpha_\nu^\dagger))\frac{j_\nu}{\alpha_\nu^\dagger}

Here :math:`I_\mathrm{ext}` is an external radiation field that irradiates the source from the outside (for example the CMB). If the source is completely optically thick, external radiation cannot penetrate the source and the corresponding term vanishes. 

For the approximate Psi operator, we choose

.. math::
    \Psi^*_\nu[j_\nu] = (1-\beta(\alpha_\nu^\dagger))\frac{j_\nu}{\alpha_\nu^\dagger}

Please see the :ref:`section about source geometries<geometries>` for a list of all geometries available in ``pythonradex`` with the corresponding formulas for the escape probability.

Ng-acceleration
------------------------

``pyhonradex`` employs Ng-acceleration [Ng74]_ to further accelerate convergence. Ng-acceleration uses the last three iteration steps to compute the next step. See the lecture notes by Dullemond_ (Sect. 4.4.7) for more details.


.. _Dullemond: http://www.ita.uni-heidelberg.de/~dullemond/lectures/radtrans_2012/
