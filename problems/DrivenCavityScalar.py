__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from DrivenCavity import *
from numpy import exp

# Override some problem specific parameters 
NS_parameters.update(dict(
    nu = 0.001,
    T = 5,
    dt = 0.01,
    max_iter = 1,
    folder = "drivencavityscalar_results",
    #convection = 'Skew',
    plot_interval = 10,
    velocity_degree = 1,
    use_krylov_solvers = False
  )
)

# Declare two scalar fields with different diffusivities
scalar_components = ['c', 'k']
Schmidt['c'] = 1.
Schmidt['k'] = 1.

class C0(Expression):
    def eval(self, values, x):
        values[0] = exp(-10*pow((pow(x[0]-0.5, 4) + pow(x[1]-0.5, 4)), 0.25))

# Specify boundary conditions
create_bcs_0 = create_bcs
def create_bcs(V, sys_comp, **NS_namespace):
    bcs = create_bcs_0(V, sys_comp, **NS_namespace)
    bcs['c'] = [DirichletBC(V, 0., DomainBoundary())]
    bcs['k'] = [DirichletBC(V, 2., lid)]
    return bcs

initialize_0 = initialize
def initialize(x_, x_1, x_2, bcs, **NS_namespace):
    initialize_0(x_, x_1, x_2, bcs, **NS_namespace)
    for ci in scalar_components:
        [bc.apply(x_[ci]) for bc in bcs[ci]]
        [bc.apply(x_1[ci]) for bc in bcs[ci]]
    #q_['c'].vector()[:] = 1e-12 # To help Krylov solver on first timestep
    #q_['k'].vector()[:] = 1e-12
    
def scalar_source(v, **NS_namespace):
    fs = {"c": C0(), "k": Constant(0)}
    return fs    
    
temporal_hook0 = temporal_hook    
def temporal_hook(tstep, u_, Vv, uv, p_, c_, k_, plot_interval, **NS_namespace):
    temporal_hook0(tstep, u_, Vv, uv, p_, plot_interval, **NS_namespace)
    if tstep % plot_interval == 0:
        plot(c_, title="First scalar ['c']")
        plot(k_, title="Second scalar ['k']")
