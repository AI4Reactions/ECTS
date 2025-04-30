#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import,print_function
from setuptools import setup,find_packages
import os
setup(name="EcTs",
     version='1.0',
     description='Equivariant Consistency conformation generative model for transition states sampling',
     author='Mingyuan Xu',
     license='GPL3',
     packages=find_packages(),
     zip_safe=False,
     include_package_data=True,
     )

