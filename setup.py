import os
from setuptools import setup, find_packages
from glob import glob

version = '0.2.0'

setup(name='sssrlib',
      version=version,
      description='Library for self-supervised super-resolution',
      author='Shuo Han',
      url='https://github.com/shuohan/sssrlib',
      author_email='shan50@jhu.edu',
      license='GPLv3',
      packages=['sssrlib'],
      python_requires='>=3.7.7',
      install_requires=['scipy>=1.5.2',
                        'matplotlib>=3.3.2',
                        'improc3d>=0.5.2',
                        'numpy>=1.18.5',
                        'torch>=1.6.0'],
      classifiers=['Programming Language :: Python :: 3',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent']
      )
