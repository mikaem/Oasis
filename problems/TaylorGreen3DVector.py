__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-10-05"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from TaylorGreen3D import *

initial_fields = dict(
        u=('sin(x[0])*cos(x[1])*cos(x[2])',  
           '-cos(x[0])*sin(x[1])*cos(x[2])', 
           '0'),
        p='0')

def velocity_tentative_hook(**NS_namespace):
    pass
