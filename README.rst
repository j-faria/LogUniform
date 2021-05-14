LogUniform
==========

A simple implementation of the log-uniform and modified log-uniform
distributions.

|License MIT| |Travis build| |PyPI version|

How to use
----------

Install it from pip (**LogUniform** only depends on numpy)

::

    pip install LogUniform

and it's ready to use from Python

.. code:: python

    import loguniform

**LogUniform** comes with two simple classes, ``LogUniform`` and ``ModifiedLogUniform``.
They are intended to mimic the API of ``scipy.stats`` 
(actually, the log-uniform distribution is already implemented in ``scipy.stats.reciprocal``;
the two implementations are compatible).

.. code:: python

    from loguniform import LogUniform, ModifiedLogUniform

    d1 = LogUniform(a=1, b=1000)
    d2 = ModifiedLogUniform(knee=1, b=1000)

both distributions ``d1`` and ``d2`` now have methods

-  ``pdf(x)`` and ``logpdf(x)``: the probability density function and its logarithm
-  ``cdf(x)``: cumulative density function
-  ``ppf(x)``: percent point function (inverse of cdf)
-  ``rvs(size)``: draw random samples from the distribution
-  ``support()``: support of the distribution


License
-------

Copyright 2021 João Faria.

**LogUniform** is free software made available under the MIT License. For
details see the LICENSE_ file.

.. _License: https://github.com/j-faria/LogUniform/blob/master/LICENSE
.. |Travis build| image:: https://travis-ci.org/j-faria/LogUniform.svg?branch=master
    :target: https://travis-ci.org/j-faria/LogUniform
.. |License MIT| image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
   :target: https://github.com/j-faria/LogUniform/blob/master/LICENSE
.. |PyPI version| image:: https://badge.fury.io/py/LogUniform.svg
   :target: https://pypi.org/project/LogUniform/
