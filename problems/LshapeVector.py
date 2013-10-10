__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from Lshape import *

# Override some problem specific parameters
NS_parameters.update(dict(
    folder = "LshapeVector_results",
  )
)

p_in = Expression("sin(pi*t)", t=0.)
create_bcs0 = create_bcs
def create_bcs(Vv, V, Q, sys_comp, **NS_namespace):
    bc = create_bcs0(V, Q, sys_comp, **NS_namespace)
    bcs = {'u': [DirichletBC(Vv, (0., 0.), walls)],
           'p': bc['p']}
    return bcs

def pre_solve_hook(Vv, **NS_namespace):
    return {}

def velocity_tentative_hook(**NS_namespace):
    pass    
    
def temporal_hook(tstep, p_, u_, plot_interval, **NS_namespace):
    if tstep % plot_interval == 0:
        plot(p_)
        plot(u_)
