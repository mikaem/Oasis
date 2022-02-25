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
from oasis.problems.SkewedFlow import mesh, walls, inlet, outlet, tol, L, h
import dolfin as df
from dolfin import inner, grad, dx

# set_log_active(False)


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


def create_bcs(V, VQ, mesh, **NS_namespace):
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
        inner(grad(us), grad(vs)) * dx == df.Constant(10.0) * vs * dx,
        su,
        bcs=[df.DirichletBC(Vu, df.Constant(0), df.DomainBoundary())],
    )

    # Wrap the boundary function in an Expression to avoid the need to interpolate it back to V
    class MyExp(df.UserExpression):
        def eval(self, values, x):
            try:
                values[0] = su(x)
                values[1] = 0
                values[2] = 0
            except:
                values[:] = 0

        def value_shape(self):
            return (3,)

    bc0 = df.DirichletBC(VQ.sub(0), (0, 0, 0), walls)
    bc1 = df.DirichletBC(VQ.sub(0), MyExp(element=VQ.sub(0).ufl_element()), inlet)
    return dict(up=[bc0, bc1])


def theend_hook(u_, p_, **NS_namespace):
    df.plot(u_, title="Velocity")
    df.plot(p_, title="Pressure")
