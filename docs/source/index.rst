.. ../logos/gempy1.png
:width: 30%

About
=====
GemPy Probability
*****************

Overview
--------

``GemPy Probability`` is a package that extends the functionality of the
`GemPy <https://www.gempy.org>`_ package to include uncertainty quantification
and stochastic geological modeling. It is based on the `Pyro <https://pyro.ai>`_
probabilistic programming framework and allows for the integration of
probabilistic models into the geological modeling workflow.

.. Check out the documentation either in `gempy.org <https://www.gempy.org/>`_
(better option), or `read the docs <http://gempy.readthedocs.io/>`_.



Contents:

.. toctree::
   :maxdepth: 2

   self
   installation

.. toctree::
   :maxdepth: 2
   :caption: Probabilistic modeling for Structural Geology

   examples_intro/index
   examples_first_example_of_inference/index
   examples_basic_geology/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference



Stochastic geological modeling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the most advanced features that sets ``GemPy`` also apart from
available commercial packages is the full integration of stochastic
geological modeling methods.
``GemPy`` was designed from the ground up to support stochastic geological
modeling for uncertainty analysis (e.g. Monte Carlo simulations, Bayesian
inference). This was achieved by writing ``GemPy``'s core architecture
using the numerical computation library `aesara <http://deeplearning.net/software/aesara/>`_
to couple it with the probabilistic programming
framework `PyMC3 <https://pymc-devs.github.io/pymc3/notebooks/getting_started.html>`_.
This enables the use of advanced sampling methods (e.g. Hamiltonian Monte
Carlo) and is of particular relevance when considering uncertainties in
the model input data and making use of additional secondary information
in a Bayesian inference framework.

We can, for example, include uncertainties with respect to the z-position
of layer boundaries in the model space. Simple Monte Carlo simulation
via PyMC will then result in different model realizations.


.. raw:: html

   <!-- Removed images as wobble.gif not anymore included - TODO: include
   new images to represent stochastic modeling capabilities!

   <p align="center"><img src="docs/source/images/gempy_zunc.png" height="300">
   <img src="docs/source/images/model_wobble.gif" height="300"></p>

   -->


Pytorch allows the automated computation of gradients, opening the door to
the use of advanced gradient-based sampling methods
coupling ``GemPy`` and
`Pyro <https://pyro.ai>`_ (see `Pyro's documentation <http://pyro.ai/examples/intro_part_ii.html>`_
for advanced stochastic modeling. Also, the use of aesara allows making
use of GPUs through cuda (see the aesara documentation for more information.

For a more detailed elaboration of the theory behind ``GemPy``\ , we refer to the
**open access scientific publication**\ :
`\ "GemPy 1.0: open-source stochastic geological modeling and inversion"
by de la Varga et al. (2019) <https://www.geosci-model-dev.net/12/1/2019/gmd-12-1-2019.pdf>`_.

References
----------

* de la Varga, M., Schaaf, A., and Wellmann, F.: GemPy 1.0: `open-source stochastic geological modeling and inversion,` Geosci. Model Dev., 12, 1–32, https://doi.org/10.5194/gmd-12-1-2019, 2019.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`


.. image:: _static/logos/logo_CGRE.png
   :width: 40%

.. image:: _static/logos/Terranigma.png
   :width: 40%