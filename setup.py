from distutils.core import setup
from setuptools import find_packages


required = [
    "numba>=0.15",
    "numpy>=1.9",
    "scipy>=0.14"
]


setup(
    name = "pyanalog",
    version = "0.1",
    author = "Francisco M. Alvarez, Ph.D.",
    author_email = "francisco.alvarez@noaa.gov",
    description = ("A module to quickly find meteorological analogs based "
                                   "on past forecast/reanalysis data."),
    license = "GNU",
    keywords = "meteorology atmospheric sciences weather analogs",
    url = "https://github.com/mogismog/pyanalog",
    packages=find_packages(),
    long_description='Stuff goes here later...',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
    ],
)