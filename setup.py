from distutils.util import convert_path
from typing import Dict

from setuptools import setup, find_packages


def readme() -> str:
    with open('README.md') as f:
        return f.read()


version_dict = {}  # type: Dict[str, str]
with open(convert_path('molstove/version.py')) as file:
    exec(file.read(), version_dict)

setup(
    name='molstove',
    version=version_dict['__version__'],
    description='Molecular Stove - Script for Calculating Molecular Properties',
    long_description=readme(),
    classifiers=['Programming Language :: Python :: 3.7'],
    author='Gregor Simm',
    author_email='gncs2@cam.ac.uk',
    python_requires='>=3.7',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'rdkit',
        'pyscf',
        'pyberny',
        'pandas',
        'scipy',
    ],
    zip_safe=False,
    entry_points={
        'console_scripts': ['molstove = molstove.main:hook'],
    },
)
