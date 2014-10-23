import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "py-analog",
    version = "0.1",
    author = "Francisco M. Alvarez, Ph.D.",
    author_email = "franciscomalvarez@gmail.com",
    description = ("A module to quickly find meteorological analogs based "
                                   "on past forecast/reanalysis data."),
    license = "GNU",
    keywords = "meteorology atmospheric sciences weather analogs",
    url = "https://github.com/mogismog/py-analog",
    packages=['py-analog'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
    ],
)