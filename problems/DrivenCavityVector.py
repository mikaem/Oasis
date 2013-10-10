__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from DrivenCavity import *

# Specify boundary conditions
u_top = Constant((1.0, 0.0))
def create_bcs(Vv, sys_comp, **NS_namespace):
    bcs = dict((ui, []) for ui in sys_comp)    
    bc0  = DirichletBC(Vv, (0., 0.), stationary_walls)
    bc1 = DirichletBC(Vv, u_top, lid)
    bcs['u'] = [bc1, bc0]
    return bcs
    
#def initialize(q_, **NS_namespace):
#    q_['u'].vector()[:] = 1e-12 # To help Krylov solver on first timestep
    
def velocity_tentative_hook(**NS_namespace):
    pass    
