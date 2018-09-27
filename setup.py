#!/usr/bin/env python

from setuptools import setup

# Version number
major = 2018
minor = 1

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
                  "oasis.problems.NSfracStep",
                  "oasis.problems.NSCoupled",
                  "oasis.solvers",
                  "oasis.solvers.NSfracStep",
                  "oasis.solvers.NSfracStep.LES",
                  "oasis.solvers.NSCoupled",
                  "oasis.common",
                  ],
      package_dir = {"oasis": "oasis"},
      entry_points = {'console_scripts': ['oasis=oasis.run_oasis:main']},
    )
