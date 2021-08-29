#!/usr/bin/env python

from distutils.core import setup

dependencies = []

# root_dir = ["deepfacet"]

packages = ['', 'ScriptBuilder', 'PoreGrid', 'LogAggregator', 'CubeSymmetries']
for n, package in enumerate(packages):
    packages[n] = 'deepfacet.' + package

setup(name="DeepFacet",
      version="0.1",
      description="A collection of scripts for my master's thesis",
      author="Alexander Sexton",
      author_email="alexahs@uio.no",
      url="github.com/alexahs/DeepFacet",
      install_requires = dependencies,
      packages = packages
)
