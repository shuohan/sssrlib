import os
from setuptools import setup, find_packages
from glob import glob

version = '0.1.0'

setup(name='sssrlib',
      version=version,
      description='Library for self-supervised super-resolution',
      author='Shuo Han',
      url='https://github.com/shuohan/sssrlib',
      author_email='shan50@jhu.edu',
      license='MIT',
      packages=['sssrlib']
      )
