from __future__ import print_function

__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

# from ..Skewed2D import *
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
from oasis.problems.NSfracStep import (
    velocity_tentative_hook,
    pressure_hook,
    start_timestep_hook,
    temporal_hook,
)
from oasis.problems.Skewed2D import mesh, walls, inlet, outlet
import dolfin as df


def get_problem_parameters(**kwargs):
    NS_parameters = dict(
        scalar_components=scalar_components,
        Schmidt=Schmidt,
        Schmidt_T=Schmidt_T,
        nu=0.1,
        T=10.0,
        dt=0.05,
        use_krylov_solvers=True,
        print_velocity_pressure_convergence=True,
    )
    NS_expressions = {}
    return NS_parameters, NS_expressions


def create_bcs(V, Q, mesh, **NS_namespace):
    u_inlet = df.Expression("10*x[1]*(0.2-x[1])", element=V.ufl_element())
    bc0 = df.DirichletBC(V, 0, walls)
    bc1 = df.DirichletBC(V, u_inlet, inlet)
    bc2 = df.DirichletBC(V, 0, inlet)
    return dict(u0=[bc1, bc0], u1=[bc2, bc0], p=[df.DirichletBC(Q, 0, outlet)])


def pre_solve_hook(mesh, u_, AssignedVectorFunction, **NS_namespace):
    return dict(uv=AssignedVectorFunction(u_, "Velocity"), n=df.FacetNormal(mesh))


def temporal_hook(
    u_, p_, mesh, tstep, print_intermediate_info, uv, n, plot_interval, **NS_namespace
):
    if tstep % print_intermediate_info == 0:
        print("Continuity ", df.assemble(df.dot(u_, n) * df.ds()))

    if tstep % plot_interval == 0:
        uv()
        df.plot(uv, title="Velocity")
        df.plot(p_, title="Pressure")


def theend_hook(uv, p_, **NS_namespace):
    uv()
    df.plot(uv, title="Velocity")
    df.plot(p_, title="Pressure")
