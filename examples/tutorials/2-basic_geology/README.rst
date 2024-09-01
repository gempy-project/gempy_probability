2 - Probabilistic Geomodel
==========================

Probability density transformation
----------------------------------


In the example above the parameters that define the likelihood function :math:`\pi_S(\theta)` were directly the prior parameters that we define--the top most level of the probability graph. The simplicity of that model allows for an easy interpretation of the whole inference as the joint probability of the prior distributions, :math:`\mu_\theta` and :math:`\sigma_\theta`, and likelihood functions :math:`\pi(y|\theta)`  (for one observation). However, there is not reason why the prior random variable—e.g. :math:`\mu_\theta` or :math:`\sigma_\theta` above—has to derive from one of the “classical” probability density functions—e.g. Normal or Gamma. This means that we can perform as many transformations of a random variables, :math:`\theta`, as we please before we plug them into a likelihood function, :math:`\pi(y|\theta)`. This is important because usually we will be able to describe :math:`\theta` as function of other random variables either because they are easier to estimate or because it will allow to grow the probabilistic model to integrate multiple data set and knowledge in a coherent fashion. 
