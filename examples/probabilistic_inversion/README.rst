Probabilistic Inversions: Joint Probability
===========================================


So far, for simplicity's sake, we have demonstrated the method with different sorts of geometric data. However, the main potential of probabilistic machine learning comes when the model is able to cover multiple types of heterogeneous data such as geophysical responses. In the following example, we use gravity as an additional constraint of the previous case. In this case, the structural model is defined by two horizons, but now they are described by 4 points each—simply for the sake of a better explanation. However, for this example, instead of using the thickness between layers, the observations are forward gravity measurements from a device on the model. To connect these two distinct data types, the model needs to be capable of simulating said observations, in this case, to be able to compute the forward gravity of the interpolated geological model.

Despite the discrepancy of the data used—i.e., XYZ coordinates in meters delimiting horizons and mGal observed in an arbitrary location at the surface—we can still infer the set of values of the model parameters that honor the observation.


License
-------
The code in this case study is copyrighted by Miguel de la Varga and licensed under the new BSD (3-clause) license:

https://opensource.org/licenses/BSD-3-Clause

The text and figures in this case study are copyrighted by Miguel de la Varga and licensed under the CC BY-NC 4.0 license:

https://creativecommons.org/licenses/by-nc/4.0/
Make sure to replace the links with actual hyperlinks if you're using a platform that supports it (e.g., Markdown or HTML). Otherwise, the plain URLs work fine for plain text.
