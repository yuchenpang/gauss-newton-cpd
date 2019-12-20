from setuptools import setup, find_packages
import os

PACKAGE_NAME = 'cpd'
VERSION = '0.1.0'

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    packages=find_packages(exclude=[]),
    package_data={},
    install_requires=[
        'numpy',
    ],
)
