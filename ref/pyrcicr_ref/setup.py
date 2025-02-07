from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pyrcicr_ref',
    version='0.1.0',
    description='Reference implementation of the R rcicr package for python.',
    author='Hristo Valev',
    packages=find_packages(),
    install_requires=requirements,
)
