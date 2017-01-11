# -*- coding: utf-8 -*-
# Copyright 2016-2017 Nate Bogdanowicz
import os
import os.path
from setuptools import setup, find_packages

classifiers = [
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
]

# Load metadata from __about__.py
base_dir = os.path.dirname(__file__)
about = {}
with open(os.path.join(base_dir, 'lentil', '__about__.py')) as f:
    exec(f.read(), about)

install_requires = ['numpy', 'matplotlib', 'pint']

if __name__ == '__main__':
    setup(
        name = about['__distname__'],
        version = about['__version__'],
        packages = find_packages(),
        author = about['__author__'],
        author_email = about['__email__'],
        url = about['__url__'],
        license = about['__license__'],
        classifiers = classifiers,
        install_requires = install_requires
    )
