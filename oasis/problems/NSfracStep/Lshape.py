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

# from oasis.problems.Lshape import mesh
import dolfin as df
import matplotlib.pyplot as plt


def get_problem_parameters(**kwargs):
    Re = 500.0
    nu = 1.0 / Re
    NS_parameters = dict(
        scalar_components=scalar_components,
        Schmidt=Schmidt,
        Schmidt_T=Schmidt_T,
        nu=nu,
        T=10,
        dt=0.01,
        Re=Re,
        Nx=40,
        Ny=40,
        folder="Lshape_results",
        max_iter=1,
        plot_interval=1,
        velocity_degree=2,
        use_krylov_solvers=True,
    )

    if "pressure_degree" in kwargs.keys():
        degree = kwargs["pressure_degree"]
    else:
        degree = NS_parameters["pressure_degree"]
    # set default parameters
    for key, val in default_parameters.items():
        if key not in NS_parameters.keys():
            NS_parameters[key] = val

    NS_expressions = dict(p_in=df.Expression("sin(pi*t)", t=0.0, degree=degree))
    return NS_parameters, NS_expressions


# Create a mesh here
class Submesh(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 0.25 - df.DOLFIN_EPS and x[1] > 0.25 - df.DOLFIN_EPS


def mesh(Nx, Ny, **params):
    mesh_ = df.UnitSquareMesh(Nx, Ny)
    subm = Submesh()
    mf1 = df.MeshFunction("size_t", mesh_, 2)
    mf1.set_all(0)
    subm.mark(mf1, 1)
    return df.SubMesh(mesh_, mf1, 0)


def inlet(x, on_boundary):
    return df.near(x[1] - 1.0, 0.0) and on_boundary


def outlet(x, on_boundary):
    return df.near(x[0] - 1.0, 0.0) and on_boundary


def walls(x, on_boundary):
    return (
        df.near(x[0], 0.0)
        or df.near(x[1], 0.0)
        or (x[0] > 0.25 - 5 * df.DOLFIN_EPS and x[1] > 0.25 - 5 * df.DOLFIN_EPS)
        and on_boundary
    )


def create_bcs(V, Q, sys_comp, p_in, **NS_namespace):
    bcs = dict((ui, []) for ui in sys_comp)
    bc0 = df.DirichletBC(V, 0.0, walls)
    pc0 = df.DirichletBC(Q, p_in, inlet)
    pc1 = df.DirichletBC(Q, 0.0, outlet)
    bcs["u0"] = [bc0]
    bcs["u1"] = [bc0]
    bcs["p"] = [pc0, pc1]
    return bcs


def pre_solve_hook(mesh, OasisFunction, u_, **NS_namespace):
    Vv = df.VectorFunctionSpace(mesh, "CG", 1)
    return dict(Vv=Vv, uv=OasisFunction(u_, Vv))


def start_timestep_hook(t, p_in, **NS_namespace):
    p_in.t = t


def temporal_hook(tstep, q_, u_, uv, Vv, plot_interval, **NS_namespace):
    if tstep % plot_interval == 0:
        df.plot(q_["p"], title="Pressure")
        uv()  # uv = project(u_, Vv)
        df.plot(uv, title="Velocity")
        plt.show()
