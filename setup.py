﻿# setup.py for gempy_viewer. Requierements are numpy and matplotlib

from setuptools import setup, find_packages

with open("gempy_probability/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip("'")
            break


def read_requirements(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


setup(
    name='gempy_probability',
    version=version,
    packages=find_packages(),
    url='',
    license='EUPL',
    author='GemPy Probability Developers', 
    author_email="gempy@terranigma-solutions.com",
    description='Extra plugins for the geological modeling package GemPy',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: GIS',
        'Programming Language :: Python :: 3.10'
    ],
    python_requires='>=3.10'
)
