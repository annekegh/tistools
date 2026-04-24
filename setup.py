#!/usr/bin/env python

from glob import glob
from setuptools import setup

setup(
    name='tistools',
    version='0.1',
    package_dir={'tistools': 'lib'},
    packages=['tistools'],
    scripts=glob('scripts/*'),
)


