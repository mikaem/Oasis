Oasis
=====
[![Build Status](https://travis-ci.org/mikaem/Oasis.svg?branch=master)](https://travis-ci.org/mikaem/Oasis)
![github-CI](https://github.com/mikaem/Oasis/workflows/github-CI/badge.svg)

<p align="center">
    <img src="https://rawgit.com/mikaem/oasis/master/figs/channel3D.gif" width="360" height="200" alt="Channel flow"/>
</p>
<p align="center">
    Turbulent channel flow
</p>

Description
-----------

Oasis is a high-level/high-performance Open Source Navier-Stokes solver written in Python. The solver has been found to scale well weakly up to 256 CPUs on the Abel supercomputer at the University of Oslo. The scaling test was using the P2P1 version of the NSfracStep solver with IPCS_ABCN.
<p align="center">
    <img src="https://rawgit.com/mikaem/oasis/master/figs/oasis_weak_scaling_loglog_1M.png" width="600" height="400" alt="Weak scaling"/>
</p>
<p align="center">
    Weak scaling on the Abel supercomputer. Timings are sampled every 10'th time step. The figure shows both the best result (over 10 time steps), the worst, and the standard deviation of the timings.
</p>


Authors
-------

Oasis is developed by

  * Mikael Mortensen
  * Kristian Valen-Sendstad
  * Joakim Bø

Licence
-------

Oasis is licensed under the GNU GPL, version 3 or (at your option) any
later version.

Oasis is Copyright (2013-2021) by the authors.

Documentation
-------------

See [wiki](https://github.com/mikaem/oasis/wiki) or [User Manual](https://github.com/mikaem/Oasis/tree/master/doc/usermanual.pdf)

If you wish to use Oasis for journal publications, please cite the following [paper](http://www.sciencedirect.com/science/article/pii/S0010465514003786).

Oasis is specifically designed with complex biomedical flows in mind. One example using Oasis for cerebrospinal fluid flow is featured [here](https://fenicsproject.org/featured/2015/csf-lpt.html).

The implementation and verification of an LES model is featured [here](https://www.researchgate.net/publication/294088673_IMPLEMENTATION_VERIFICATION_AND_VALIDATION_OF_LARGE_EDDY_SIMULATION_MODELS_IN_OASIS)

Installation
------------

Oasis requires a compatible installation of FEniCS, see the releases. 
Oasis is installed with regular distutils

  * python setup.py install --prefix='Path to where you want Oasis installed. Must be on PYTHONPATH'

Contact
-------

The latest version of this software can be obtained from

  https://github.com/mikaem/oasis

Please report bugs and other issues through the issue tracker at:

  https://github.com/mikaem/oasis/issues

