#!/usr/bin/env python
from setuptools import setup

setup(name='LogUniform',
      version='1.0.1',
      description='Implementation of the log-uniform and modified log-uniform distributions',
      long_description=open('README.rst').read(),
      author='João Faria',
      author_email='joao.faria@astro.up.pt',
      url='https://github.com/j-faria/LogUniform',
      license='MIT',
      packages=['loguniform'],
      intall_requires=['numpy',],
      setup_requires=['pytest-runner',],
      tests_require=['pytest',],
      classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
     )