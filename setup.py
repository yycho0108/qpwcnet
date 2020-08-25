#!/usr/bin/env python3

from setuptools import setup, find_packages
packages = find_packages()
print(packages)

setup(name='qpwcnet',
      version='0.0.1',
      description='Quantized PWCNet Variants',
      url='http://github.com/yycho0108/qpwcnet',
      author='Jamie Cho',
      author_email='jchocholate@gmail.com',
      license='MIT',
      packages=packages,
      zip_safe=False,
      scripts=[],
      # FIXME(yycho0108): Update requirements here.
      install_requires=[
          'tensorflow-gpu',
          'tensorflow-addons',
      ],
      # NOTE(yycho0108): bug with latest pyqt5 prevents installation.
      # https://stackoverflow.com/questions/59711301/install-pyqt5-5-14-1-on-linux
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],
      )
