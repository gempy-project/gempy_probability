Probabilistic modelling with GemPy
``````````````````````````````````

In structural geology, we want to combine different types of data—i.e. geometrical measurements, geophysics, petrochemical data—usually using as a prevailing model a *common earth model*. For the sake of simplicity, in this example, we will combine different type of geometric information into one single probabilistic model. Let’s build on the previous idea in order to extend the conceptual case above back to geological modelling.

Lucky for us, after we perform the first inference on the thickness, :math:`\tilde{y}_{thickness}` of the model, we find out that a colleague has been gathering data at the exact same outcrop but in his case he was recording the location of the top :math:`\tilde{y}_{top}` and bottom :math:`\tilde{y}_{bottom}` interfaces of the layer. We can relate the three data sets with simple algebra:

.. math::
   \pi(\theta_{thickness}) = \pi(\theta_{top})  - \pi(\theta_{bottom}) 

or,

.. math::
   \pi(\theta_{bottom})  = \pi(\theta_{top})  - \pi(\theta_{thickness})

now the question is which probabilistic model design is more suitable. In the end this relates directly to the question the model is trying to answer---and possible limitations on the algorithms used---since joint probability follows the commutative and associative properties.

