import os
from setuptools import setup, find_packages
from glob import glob

version = '0.1.1'

dirname = os.path.abspath(os.path.dirname(__file__))

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open(os.path.join(dirname, 'requirements.txt'), 'r') as f:
    requirements = f.read().splitlines()

required = []
dependency_links = []

EGG_MARK = '#egg='
for line in requirements:
    if line.startswith('-e git:') or line.startswith('-e git+') or \
            line.startswith('git:') or line.startswith('git+'):
        if EGG_MARK in line:
            package_name = line[line.find(EGG_MARK) + len(EGG_MARK):]
            required.append(package_name)
            dependency_links.append(line)
        else:
            print('Dependency to a git repository should have the format:')
            print('git+ssh://git@github.com/xxxxx/xxxxxx#egg=package_name')
    else:
        required.append(line)


setup(name='sssrlib',
      version=version,
      description='Library for self-supervised super-resolution',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Shuo Han',
      url='https://github.com/shuohan/sssrlib',
      author_email='shan50@jhu.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=required,
      dependency_links=dependency_links,
      classifiers=['Programming Language :: Python :: 3',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent']
      )
