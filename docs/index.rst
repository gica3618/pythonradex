.. pythonradex documentation master file, created by
   sphinx-quickstart on Thu Dec  7 13:57:20 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the ``pyhonradex`` documentation!
=================================================

``pythonradex`` solves the radiative transfer for a uniform medium in non-LTE with an escape probability formalism. The code can be used to quickly estimate the emission from an astrophysical gas given input parameters such as the kinetic temperature, the column density and the density of the collision partners.

``pythonradex`` is a python re-implementation of the RADEX code [vanderTak07]_. Depending on the setup, it can be faster than RADEX when calculating a grid of models. It also provides additional functionality that is not included in RADEX (treatment of overlapping lines, treatment of internal dust continuum, output of spectra,...).

``pyhonradex`` also provides a convenient method to read files from the LAMDA_ database.

Please see the :ref:`example notebooks<examples>` for an overview of the functionalities offered by ``pythonradex``.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   examples
   theory
   geometry
   difference_pythonradex_RADEX
   faq
   help
   API
   citation
   


Acknowledgement
=================
I would like to thank Simon Bruderer for his helpful clarifications about the ALI method. I also thank Andrés Asensio Ramos for helpful discussions about the LVG geometry and help with the `MOLPOP-CEP` code that was used to benchmark `pythonradex`.


Bibliography
=================

.. [deJong80] de Jong, T., Boland, W., & Dalgarno, A. 1980, Astronomy and Astrophysics, 91, 68

.. [deJong75] de Jong, T., Chu, S., & Dalgarno, A. 1975, Astrophysical Journal, 199, 69 

.. [Elitzur92] Elitzur, M. 1992, *Astronomical masers*, ISBN 978-0-7923-1217-8, Springer Dordrecht

.. [Goldreich74] Goldreich, P., & Kwan, J. 1974, Astrophysical Journal, 189, 441

.. [Hubeny03] Hubeny, I. 2003, Accelerated Lambda Iteration: An Overview. Stellar Atmosphere Modeling 288, 17

.. [Ng74] Ng, K.-C. 1974, Journal of Chemical Physics, 61, 2680

.. [Osterbrock74] Osterbrock, D. E. 1974, *Astrophysics of Gaseous Nebulae*, ISBN 0-716-70348-3, W. H. Freeman

.. [Rybicki91] Rybicki, G. B., & Hummer, D. G. 1991, Astronomy and Astrophysics, 245, 171

.. [Rybicki92] Rybicki, G. B., & Hummer, D. G. 1992, Astronomy and Astrophysics, 262, 209

.. [Rybicki04] Rybicki, G. B., & Lightman, A.P. 2004, *Radiative Processes in Astrophysics*, ISBN 0-471-82759-2, Wiley-VCH

.. [Scoville74]  Scoville, N. Z., & Solomon, P. M. 1991, Astrophysical Journal, 187, L67

.. [vanderTak07]  van der Tak, F. F. S., Black, J. H., Schöier, F. L., Jansen, D. J., & van Dishoeck, E. F. 2007, Astronomy and Astrophysics, 468, 627


.. _RADEX: http://home.strw.leidenuniv.nl/~moldata/radex.html

.. _LAMDA: http://home.strw.leidenuniv.nl/~moldata/
