__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

# lots of unused imports, that are all imported here and used by the main script only
# from ..NSfracStep import *
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
from oasis.problems.NSfracStep import (
    velocity_tentative_hook,
    pressure_hook,
    start_timestep_hook,
    temporal_hook,
    default_parameters,
)
from oasis.problems.DrivenCavity import noslip, top, bottom, mesh
import dolfin as df

# set_log_active(False)


def get_problem_parameters(**kwargs):
    Schmidt["alfa"] = 1.0
    Schmidt["beta"] = 10.0
    NS_parameters = dict(
        Schmidt=Schmidt,
        scalar_components=scalar_components + ["alfa", "beta"],
        Schmidt_T=Schmidt_T,
        nu=0.001,
        T=1.0,
        dt=0.001,
        folder="drivencavity_results",
        plot_interval=20,
        save_step=10000,
        checkpoint=10000,
        print_intermediate_info=100,
        use_krylov_solvers=True,
    )
    # set default parameters
    for key, val in default_parameters.items():
        if key not in NS_parameters.keys():
            NS_parameters[key] = val
    NS_expressions = {}
    return NS_parameters, NS_expressions

    # NS_parameters['krylov_solvers'] = {'monitor_convergence': False,
    #                                   'report': False,
    #                                   'relative_tolerance': 1e-10,
    #                                   'absolute_tolerance': 1e-10}


# Specify boundary conditions
def create_bcs(V, **NS_namespace):
    bc0 = df.DirichletBC(V, 0, noslip)
    bc00 = df.DirichletBC(V, 1, top)
    bc01 = df.DirichletBC(V, 0, top)
    return dict(
        u0=[bc00, bc0],
        u1=[bc01, bc0],
        p=[],
        alfa=[bc00],
        beta=[df.DirichletBC(V, 1, bottom)],
    )


def initialize(x_1, x_2, bcs, **NS_namespace):
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(mesh, velocity_degree, **NS_namespace):
    Vv = df.VectorFunctionSpace(mesh, "CG", velocity_degree)
    return dict(uv=df.Function(Vv))


def temporal_hook(q_, tstep, u_, uv, p_, plot_interval, testing, **NS_namespace):
    if tstep % plot_interval == 0 and not testing:
        df.assign(uv.sub(0), u_[0])
        df.assign(uv.sub(1), u_[1])
        df.plot(uv, title="Velocity")
        df.plot(p_, title="Pressure")
        df.plot(q_["alfa"], title="alfa")
        df.plot(q_["beta"], title="beta")


def theend_hook(u_, p_, uv, mesh, testing, **NS_namespace):
    if not testing:
        df.assign(uv.sub(0), u_[0])
        df.assign(uv.sub(1), u_[1])
        df.plot(uv, title="Velocity")
        df.plot(p_, title="Pressure")

    u_norm = df.norm(u_[0].vector())
    if df.MPI.rank(df.MPI.comm_world) == 0 and testing:
        print("Velocity norm = {0:2.6e}".format(u_norm))

    if not testing:
        try:
            from fenicstools import StreamFunction

            psi = StreamFunction(uv, [], mesh, use_strong_bc=True)
            df.plot(psi, title="Streamfunction")
            import matplotlib.pyplot as plt

            plt.show()
        except:
            pass
