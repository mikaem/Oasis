#!/usr/bin/env python

from distutils.core import setup

# Version number
major = 2017
minor = 2

setup(name = "oasis",
      version = "%d.%d" % (major, minor),
      description = "Oasis - Navier-Stokes solvers in FEniCS",
      author = "Mikael Mortensen",
      author_email = "mikaem@math.uio.no",
      url = 'https://github.com/mikaem/Oasis.git',
      classifiers = [
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python ',
          'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
      packages = ["oasis",
                  "oasis.problems",
                  "oasis.solvers",
                  "oasis.common",
                  ],
      package_dir = {"oasis": "oasis"},
    )
