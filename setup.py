import os
from setuptools import setup, find_packages
from glob import glob

version = '0.3.0'

setup(name='sssrlib',
      version=version,
      description='Library for self-supervised super-resolution',
      author='Shuo Han',
      url='https://github.com/shuohan/sssrlib',
      author_email='shan50@jhu.edu',
      license='GPLv3',
      packages=['sssrlib'],
      python_requires='>=3.7.10',
      install_requires=[
          'scipy',
          'matplotlib',
          'improc3d',
          'numpy',
          'torch>=1.8.1'
      ],
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Operating System :: OS Independent'
      ]
     )
