.. _parallelisation:

Parallelisation
========================================

The :doc:`grid search example notebook <grid_search>` describes how ``pythonradex`` can be efficiently used to calculate a grid of models. In particular, it is very important to use the ``update_parameters`` method, rather than initialising a new ``Source`` object at each iteration step.

The performance can be further improved by employing parallelisation. Again, instead of creating a new ``Source`` object at each grid point, we assign a separate ``Source`` object to each worker. The workers then calculate the grid in parallel. This can be achieved with something like this:

.. code-block:: python

	def init_worker():
	    global worker_source
	    worker_source = radiative_transfer.Source(**source_kwargs)
	    worker_source.update_parameters(**initial_parameters)

	def run_single(new_parameters):
	    worker_source.update_parameters(**new_parameters)
	    worker_source.solve_radiative_transfer()
	    return worker_source.frequency_integrated_emission(
		         output_type="intensity", transitions=0)

	if __name__ == '__main__':
	    with ProcessPoolExecutor(initializer=init_worker) as executor:
		intensities = list(executor.map(run_single, grid))


The ``init_worker`` function creates a separate ``Source`` for each worker. The ``run_single`` function defines how a worker calculates a single model. A full example script can be found `here`_.


.. _here: https://github.com/gica3618/pythonradex/tree/master/docs/examples/parallelisation.py
