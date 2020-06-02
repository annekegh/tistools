#!/bin/env/python

from glob import glob
from distutils.core import setup

setup(name='tistools',
    version='0.1',
    description='Tools to deal with the output of PyRETIS and RETIS projects',
    author='An Ghysels',
    #author_email='An.Ghysels@UGent.be',
    #url='http://molmod.ugent.be/code/',
    package_dir = {'tistools': 'lib'},
    packages = ['tistools',],  #'mcdiff.tools','mcdiff.permeability'],
    scripts=glob("scripts/*"),
    classifiers=[
        #'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        #'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Topic :: Science/Engineering :: Molecular Science'
    ],
)


