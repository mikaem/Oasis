from __future__ import print_function

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
from oasis.problems.NSfracStep import (
    velocity_tentative_hook,
    pressure_hook,
    start_timestep_hook,
    temporal_hook,
)
from oasis.problems.SkewedFlow import mesh, inlet, walls, outlet
import dolfin as df

# from ..SkewedFlow import *
from numpy import cos, pi, cosh

print(
    """
This problem does not work well with IPCS since the outflow
boundary condition

    grad(u)*n=0, p=0

here is a poor representation of actual physics.

Need to use coupled solver with pseudo-traction

    (grad(u)-p)*n = 0

or extrude outlet such that the outflow boundary condition
becomes more realistic.
"""
)


def get_problem_parameters(**kwargs):
    NS_parameters = dict(
        scalar_components=scalar_components,
        Schmidt=Schmidt,
        Schmidt_T=Schmidt_T,
        nu=0.001,
        T=0.05,
        dt=0.01,
        use_krylov_solvers=True,
        print_velocity_pressure_convergence=True,
    )
    NS_expressions = {}
    return NS_parameters, NS_expressions


def create_bcs(V, Q, mesh, **NS_namespace):
    # Create inlet profile by solving Poisson equation on boundary
    bmesh = df.BoundaryMesh(mesh, "exterior")
    cc = df.MeshFunction("size_t", bmesh, bmesh.topology().dim(), 0)
    ii = df.AutoSubDomain(inlet)
    ii.mark(cc, 1)
    smesh = df.SubMesh(bmesh, cc, 1)
    Vu = df.FunctionSpace(smesh, "CG", 1)
    su = df.Function(Vu)
    us = df.TrialFunction(Vu)
    vs = df.TestFunction(Vu)
    df.solve(
        df.inner(df.grad(us), df.grad(vs)) * df.dx == df.Constant(10.0) * vs * df.dx,
        su,
        bcs=[df.DirichletBC(Vu, df.Constant(0), df.DomainBoundary())],
    )

    # Wrap the boundary function in an Expression to avoid the need to interpolate it back to V
    class MyExp(df.UserExpression):
        def eval(self, values, x):
            try:
                values[0] = su(x)
            except:
                values[0] = 0

    bc0 = df.DirichletBC(V, 0, walls)
    bc1 = df.DirichletBC(V, MyExp(element=V.ufl_element()), inlet)
    bc2 = df.DirichletBC(V, 0, inlet)
    return dict(
        u0=[bc0, bc1], u1=[bc0, bc2], u2=[bc0, bc2], p=[df.DirichletBC(Q, 0, outlet)]
    )


def temporal_hook(
    u_, p_, mesh, tstep, print_intermediate_info, plot_interval, **NS_namespace
):

    if tstep % print_intermediate_info == 0:
        print("Continuity ", df.assemble(df.dot(u_, df.FacetNormal(mesh)) * df.ds()))

    if tstep % plot_interval == 0:
        df.plot(u_, title="Velocity")
        df.plot(p_, title="Pressure")


def theend_hook(u_, p_, **NS_namespace):
    df.plot(u_, title="Velocity")
    df.plot(p_, title="Pressure")
