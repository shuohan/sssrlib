# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from glob import glob

version = '0.1.1'

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='sssrlib',
      version=version,
      description='Library for self-supervised super-resolution',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Shuo Han',
      url='https://github.com/shuohan/sssrlib',
      author_email='shan50@jhu.edu',
      license='MIT',
      install_requires=['torch >= 1.6.0',
                        'improc3d @ git+https://github.com/shuohan/improc3d@0.5.0#egg=improc3d'],
      packages=find_packages(),
      classifiers=['Programming Language :: Python :: 3',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent']
      )
