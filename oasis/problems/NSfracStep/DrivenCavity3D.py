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

# from oasis.problems.DrivenCavity3D import mesh
import dolfin as df
from numpy import cos, pi


def get_problem_parameters(**kwargs):
    NS_parameters = dict(
        scalar_components=scalar_components,
        Schmidt=Schmidt,
        Schmidt_T=Schmidt_T,
        nu=0.01,
        T=1.0,
        dt=0.01,
        Nx=15,
        Ny=15,
        Nz=15,
        plot_interval=20,
        print_intermediate_info=100,
        use_krylov_solvers=True,
    )
    # set default parameters
    for key, val in default_parameters.items():
        if key not in NS_parameters.keys():
            NS_parameters[key] = val
    NS_expressions = dict(constrained_domain=PeriodicDomain())
    return NS_parameters, NS_expressions


# Create a mesh
def mesh(Nx, Ny, Nz, **params):
    m = df.UnitCubeMesh(Nx, Ny, Nz)
    x = m.coordinates()
    x[:, :2] = (x[:, :2] - 0.5) * 2
    x[:, :2] = 0.5 * (cos(pi * (x[:, :2] - 1.0) / 2.0) + 1.0)
    return m


class PeriodicDomain(df.SubDomain):
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool(df.near(x[2], 0) and on_boundary)

    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1]
        y[2] = x[2] - 1.0


def create_bcs(V, **NS_namespace):
    # Specify boundary conditions
    noslip = "std::abs(x[0]*x[1]*(1-x[0]))<1e-8"
    top = "std::abs(x[1]-1) < 1e-8"

    bc0 = df.DirichletBC(V, 0, noslip)
    bc00 = df.DirichletBC(V, 1, top)
    bc01 = df.DirichletBC(V, 0, top)

    return dict(u0=[bc00, bc0], u1=[bc01, bc0], u2=[bc01, bc0], p=[])


def initialize(x_1, x_2, bcs, **NS_namespace):
    for ui in x_2:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(
    mesh,
    velocity_degree,
    constrained_domain,
    u_,
    AssignedVectorFunction,
    **NS_namespace
):
    return dict(uv=AssignedVectorFunction(u_))


def temporal_hook(tstep, u_, uv, p_, plot_interval, **NS_namespace):
    if tstep % plot_interval == 0:
        uv()
        df.plot(uv, title="Velocity")
        df.plot(p_, title="Pressure")


def theend_hook(p_, uv, **NS_namespace):
    uv()
    df.plot(uv, title="Velocity")
    df.plot(p_, title="Pressure")
