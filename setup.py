# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from glob import glob

version = '0.1.1'

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='improc3d',
      version=version,
      description='Library for self-supervised super-resolution',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Shuo Han',
      url='https://github.com/shuohan/sssrlib',
      author_email='shan50@jhu.edu',
      license='MIT',
      install_requires=['numpy', 'scipy'],
      packages=find_packages(),
      classifiers=['Programming Language :: Python :: 3',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent']
      )
