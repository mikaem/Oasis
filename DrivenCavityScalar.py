__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from DrivenCavity import *

# Override some problem specific parameters 
NS_parameters.update(dict(
    nu = 0.01,
    T = 2.5,
    dt = 0.01,
    folder = "drivencavityscalar_results",
    convection = 'Skew',
    plot_interval = 10,
    velocity_degree = 2,
    use_lumping_of_mass_matrix = True,
    use_krylov_solvers = True
  )
)

# Declare two scalar fields with different diffusivities
scalar_components = ['c', 'k']
Schmidt['c'] = 10.
Schmidt['k'] = 1.

# Specify boundary conditions
create_bcs_0 = create_bcs
def create_bcs(V, sys_comp, **NS_namespace):
    bcs = create_bcs_0(V, sys_comp, **NS_namespace)
    bcs['c'] = [DirichletBC(V, 1., lid)]
    bcs['k'] = [DirichletBC(V, 2., lid)]
    return bcs

initialize_0 = initialize
def initialize(q_, **NS_namespace):
    initialize_0(q_, **NS_namespace)
    q_['c'].vector()[:] = 1e-12 # To help Krylov solver on first timestep
    q_['k'].vector()[:] = 1e-12
    
temporal_hook0 = temporal_hook    
def temporal_hook(tstep, u_, Vv, uv, p_, c_, k_, plot_interval, **NS_namespace):
    temporal_hook0(tstep, u_, Vv, uv, p_, plot_interval, **NS_namespace)
    if tstep % plot_interval == 0:
        plot(c_, title="First scalar ['c']")
        plot(k_, title="Second scalar ['k']")
