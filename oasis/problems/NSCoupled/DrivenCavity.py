__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-08"
__copyright__ = "Copyright (C) 2014 " + __author__
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
from oasis.problems.DrivenCavity import noslip, top, bottom, mesh
import dolfin as df


def get_problem_parameters(**kwargs):
    NS_parameters = dict(nu=0.002, max_iter=100)
    NS_parameters["scalar_components"] = scalar_components
    NS_parameters["Schmidt"] = Schmidt
    NS_parameters["Schmidt_T"] = Schmidt_T
    # set default parameters
    for key, val in default_parameters.items():
        if key not in NS_parameters.keys():
            NS_parameters[key] = val
    return NS_parameters


# Specify boundary conditions
def create_bcs(VQ, **NS_namespace):
    bc0 = df.DirichletBC(VQ.sub(0), (0, 0), noslip)
    bc1 = df.DirichletBC(VQ.sub(0), (1, 0), top)
    return dict(up=[bc0, bc1])


def theend_hook(u_, p_, mesh, **NS_namespace):
    df.plot(u_, title="Velocity")
    df.plot(p_, title="Pressure")

    try:
        from fenicstools import StreamFunction
        import matplotlib.pyplot as plt

        psi = StreamFunction(u_, [], mesh, use_strong_bc=True)
        df.plot(psi, title="Streamfunction")
        plt.show()
    except:
        pass
