# LogUniform

A simple implementation of the log-uniform and modified log-uniform distributions.


[![License MIT](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/j-faria/LogUniform/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/LogUniform.svg)](https://pypi.org/project/LogUniform/)


How to use
----------

Install it from pip (**LogUniform** only depends on numpy)

```
pip install LogUniform
```

and it's ready to use from Python

```python
import loguniform
```



**LogUniform** comes with two simple classes, `LogUniform` and `ModifiedLogUniform`.  
They are intended to mimic the API of `scipy.stats`
(actually, the log-uniform distribution is already implemented in `scipy.stats.reciprocal`;
the two implementations are compatible).

```python
from loguniform import LogUniform, ModifiedLogUniform

d1 = LogUniform(a=1, b=1000)
d2 = ModifiedLogUniform(knee=1, b=1000)
```

both distributions `d1` and `d2` now have methods

- `pdf(x)` and `logpdf(x)`
- `cdf(x)`
- `rvs(size)`

to calculate the probability density function (and its logarithm), 
the cumulative density function, and to get random samples from the distribution.

License
-------

Copyright 2018 João Faria.

LogUniform is free software made available under the MIT License. For details see the LICENSE file.
