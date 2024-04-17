#!/bin/env/python

from glob import glob
from distutils.core import setup

setup(
    version='0.1',
    #package_dir = {'tistools': 'lib'},
    #packages = ['tistools',],  #'mcdiff.tools','mcdiff.permeability'],
    scripts=glob("scripts/*"),
)


