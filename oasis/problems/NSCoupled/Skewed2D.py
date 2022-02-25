__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from oasis.problems import (
    constrained_domain,
    scalar_components,
    Schmidt,
    Schmidt_T,
    body_force,
    initialize,
    scalar_hook,
    scalar_source,
    pre_solve_hook,
    theend_hook,
    get_problem_parameters,
    post_import_problem,
    create_bcs,
)
from oasis.problems import create_bcs
from oasis.problems.NSCoupled import (
    NS_hook,
    start_iter_hook,
    end_iter_hook,
    default_parameters,
)
from oasis.problems.Skewed2D import mesh, walls, inlet, outlet, tol, L
import dolfin as df

# Override some problem specific parameters
def get_problem_parameters(**kwargs):
    NS_parameters = dict(
        nu=0.1, omega=1.0, plot_interval=10, max_iter=100, max_error=1e-12
    )
    NS_parameters["scalar_components"] = scalar_components
    NS_parameters["Schmidt"] = Schmidt
    NS_parameters["Schmidt_T"] = Schmidt_T
    # set default parameters
    for key, val in default_parameters.items():
        if key not in NS_parameters.keys():
            NS_parameters[key] = val
    return NS_parameters


def create_bcs(VQ, mesh, **NS_namespace):
    u_inlet = df.Expression(
        ("10*x[1]*(0.2-x[1])", "0"), element=VQ.sub(0).ufl_element()
    )
    bc0 = df.DirichletBC(VQ.sub(0), (0, 0), walls)
    bc1 = df.DirichletBC(VQ.sub(0), u_inlet, inlet)
    return dict(up=[bc1, bc0])


def theend_hook(u_, p_, **NS_namespace):
    df.plot(u_, title="Velocity")
    df.plot(p_, title="Pressure")
