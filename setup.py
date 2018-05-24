#!/usr/bin/env python
from setuptools import setup

setup(name='LogUniform',
      version='1.0',
      description='Implementation of the log-uniform and modified log-uniform distributions',
      author='João Faria',
      author_email='joao.faria@astro.up.pt',
      url='https://github.com/j-faria/LogUniform',
      packages=['loguniform'],
      setup_requires=['pytest-runner',],
      tests_require=['pytest',],
     )