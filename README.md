Oasis
=====
[![Build Status](https://travis-ci.org/mikaem/Oasis.svg?branch=master)](https://travis-ci.org/mikaem/Oasis)

<p align="center">
    <img src="https://www.dropbox.com/s/7icm78pggot8brk/channel3D.gif?dl=1" width="360" height="200" alt="Channel flow"/>
</p>
<p align="center">
    Turbulent channel flow
</p>

Description
-----------

Oasis is i high-level/high-performance Open Source Navier-Stokes solver written in Python. The solver has been found to scale well weakly up to 256 CPUs on the Abel supercomputer at the University of Oslo. The scaling test was using the P2P1 version of the NSfracStep solver with IPCS_ABCN.
<p align="center">
    <img src="https://www.dropbox.com/s/qoyxrqhrqo6nda8/oasis_weak_scaling_loglog_1M.png?dl=1" width="600" height="400" alt="Weak scaling"/>
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

Oasis is Copyright (2013-2016) by the authors.

Documentation
-------------

See [wiki](https://github.com/mikaem/oasis/wiki) or [User Manual](https://github.com/mikaem/Oasis/tree/master/doc/usermanual.pdf)

If you wish to use Oasis for journal publications, please cite the following [paper](http://www.sciencedirect.com/science/article/pii/S0010465514003786).

Installation
------------

Oasis requires a compatible installation of FEniCS, see the releases. 
Oasis should be installed by cloning to a local repository. 

  * git clone https://github.com/mikaem/Oasis.git


Contact
-------

The latest version of this software can be obtained from

  https://github.com/mikaem/oasis

Please report bugs and other issues through the issue tracker at:

  https://github.com/mikaem/oasis/issues

