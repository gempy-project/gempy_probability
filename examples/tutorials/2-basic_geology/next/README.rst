Example of Inference
````````````````````

.. figure:: ../_static/Thickness_example.svg
   :align: center
   :width: 80%
   
   Prior (in blue) and posterior distributions (in red) of all the parameters of the probabilistic model :math:`\theta` (the full list of values of the simulation can be found in Appendix X). Case A) correspond to the subtraction equation while Case B) has been generated with the summation equation. Both cases yield the exact same posterior for the random variables---within the Monte Carlo error---as expected.

Correlations
````````````

Furthermore, keep in mind that the inference happens in a multidimensional manifold---although for obvious visualization reasons we display each parameter independently. This means that although in the previous figures the posterior distributions seem independent, a closer examination---e.g. joint plots---uncovers correlation in the parameters.

.. figure:: ../_static/correlation.png
   :align: center
   :width: 80%

   Correlations between parameters can be uncovered with joint plots.


License
-------
The code in this case study is copyrighted by Miguel de la Varga and licensed under the new BSD (3-clause) license:

https://opensource.org/licenses/BSD-3-Clause

The text and figures in this case study are copyrighted by Miguel de la Varga and licensed under the CC BY-NC 4.0 license:

https://creativecommons.org/licenses/by-nc/4.0/
Make sure to replace the links with actual hyperlinks if you're using a platform that supports it (e.g., Markdown or HTML). Otherwise, the plain URLs work fine for plain text.
