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
    default_parameters,
)

# from oasis.problems.FlowPastSphere3D import mesh
import dolfin as df
from numpy import cos, pi, cosh
from os import getcwd, path
import pickle


# Create a mesh
def mesh(**params):
    m = df.Mesh("/home/mikael/MySoftware/Oasis/mymesh/boxwithsphererefined.xml")
    return m


def get_problem_parameters(**kwargs):
    if "restart_folder" in kwargs.keys():
        restart_folder = kwargs["restart_folder"]
        restart_folder = path.join(getcwd(), restart_folder)
        f = open(path.join(restart_folder, "params.dat"), "r")
        NS_parameters = pickle.load(f)
        NS_parameters["T"] = NS_parameters["T"] + 10 * NS_parameters["dt"]
        NS_parameters["restart_folder"] = restart_folder
        globals().update(NS_parameters)

    else:
        # Override some problem specific parameters
        NS_parameters = dict(
            scalar_components=scalar_components,
            Schmidt=Schmidt,
            Schmidt_T=Schmidt_T,
            nu=0.1,
            T=5.0,
            dt=0.01,
            h=0.75,
            sol=40,
            dpdx=0.05,
            velocity_degree=2,
            plot_interval=10,
            print_intermediate_info=10,
            use_krylov_solvers=True,
        )
        NS_parameters["krylov_solvers"]["monitor_convergence"] = True
        # set default parameters
        for key, val in default_parameters.items():
            if key not in NS_parameters.keys():
                NS_parameters[key] = val
        NS_expressions = {}
        return NS_parameters, NS_expressions


def create_bcs(V, Q, mesh, h, **NS_namespace):
    # Specify boundary conditions
    walls = "on_boundary && std::abs((x[1]-3)*(x[1]+3)*(x[2]-3)*(x[2]+3))<1e-8"
    inners = "on_boundary && std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]) < 1.5*{}".format(
        h
    )
    inlet = "x[0] < -3+1e-8 && on_boundary"
    outlet = "x[0] > 6-1e-8 && on_boundary"

    bmesh = df.BoundaryMesh(mesh, "exterior")
    cc = df.MeshFunction("size_t", bmesh, bmesh.topology().dim(), 0)
    ii = df.AutoSubDomain(lambda x, on_bnd: df.near(x[0], -3))
    ii.mark(cc, 1)
    smesh = df.SubMesh(bmesh, cc, 1)
    Vu = df.FunctionSpace(smesh, "CG", 1)
    su = df.Function(Vu)
    us = df.TrialFunction(Vu)
    vs = df.TestFunction(Vu)
    df.solve(
        df.inner(df.grad(us), df.grad(vs)) * df.dx == df.Constant(0.1) * vs * df.dx,
        su,
        bcs=[df.DirichletBC(Vu, df.Constant(0), df.DomainBoundary())],
    )

    lp = df.LagrangeInterpolator()
    sv = df.Function(V)
    lp.interpolate(sv, su)

    bc0 = df.DirichletBC(V, 0, walls)
    bc1 = df.DirichletBC(V, 0, inners)
    bcp1 = df.DirichletBC(Q, 0, outlet)
    bc2 = df.DirichletBC(V, 0, inlet)
    bc3 = df.DirichletBC(V, sv, inlet)
    return dict(u0=[bc0, bc1, bc3], u1=[bc0, bc1, bc2], u2=[bc0, bc1, bc2], p=[bcp1])


def pre_solve_hook(mesh, velocity_degree, u_, AssignedVectorFunction, **NS_namespace):
    return dict(uv=AssignedVectorFunction(u_))


def temporal_hook(tstep, uv, p_, plot_interval, **NS_namespace):
    if tstep % plot_interval == 0:
        uv()
        df.plot(uv, title="Velocity")
        df.plot(p_, title="Pressure")


def theend_hook(p_, uv, **NS_namespace):
    uv()
    df.plot(uv, title="Velocity")
    df.plot(p_, title="Pressure")
