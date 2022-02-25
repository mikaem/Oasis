from __future__ import print_function

__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-03-21"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"


from oasis.problems import (
    add_function_to_tstepfiles,
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
import oasis.common.utilities as ut
from oasis.problems.NSfracStep import (
    velocity_tentative_hook,
    pressure_hook,
    start_timestep_hook,
    temporal_hook,
    default_parameters,
)
from oasis.problems.Cylinder import (
    mesh,
    Inlet,
    Cyl,
    Wall,
    Outlet,
    center,
    cases,
    H,
    L,
    D,
)
from dolfin import (
    Expression,
    DirichletBC,
    Function,
    MeshFunction,
    FacetNormal,
    plot,
    TestFunction,
    Identity,
    VectorFunctionSpace,
    grad,
    dot,
    assemble,
    project,
    DirichletBC,
    curl,
    DomainBoundary,
    Point,
    ds,
)
from os import getcwd, path
import pickle
import matplotlib.pyplot as plt

Schmidt["alfa"] = 0.1
scalar_components.append("alfa")


def get_problem_parameters(**kwargs):
    # Example: python NSfracstep.py [...] restart_folder="results/data/8/Checkpoint"
    if "restart_folder" in kwargs.keys():
        restart_folder = kwargs["restart_folder"]
        restart_folder = path.join(getcwd(), restart_folder)
        f = open(path.join(restart_folder, "params.dat"), "rb")
        NS_parameters = pickle.load(f)
        NS_parameters["restart_folder"] = restart_folder
        globals().update(NS_parameters)

    else:
        # Override some problem specific parameters
        case = kwargs["case"] if "case" in kwargs else 1
        Um = cases[case]["Um"]
        Re = cases[case]["Re"]
        Umean = 2.0 / 3.0 * Um

        NS_parameters = dict(
            scalar_components=scalar_components,
            Schmidt=Schmidt,
            Schmidt_T=Schmidt_T,
            Um=Um,
            Re=Re,
            Umean=Umean,
            nu=Umean * D / Re,
            H=H,
            L=L,
            D=D,
            T=100,
            dt=0.01,
            checkpoint=50,
            save_step=50,
            plot_interval=10,
            velocity_degree=2,
            print_intermediate_info=100,
            use_krylov_solvers=True,
        )
        NS_parameters["krylov_solvers"] = dict(monitor_convergence=True)

        NS_parameters["velocity_krylov_solver"] = dict(
            preconditioner_type="jacobi", solver_type="bicgstab"
        )
        # set default parameters
        for key, val in default_parameters.items():
            if key not in NS_parameters.keys():
                NS_parameters[key] = val

    NS_expressions = {}
    return NS_parameters, NS_expressions


def create_bcs(V, Q, Um, H, **NS_namespace):
    inlet = Expression("4.*{0}*x[1]*({1}-x[1])/pow({1}, 2)".format(Um, H), degree=2)
    ux = Expression("0.00*x[1]", degree=1)
    uy = Expression("-0.00*(x[0]-{})".format(center), degree=1)
    bc00 = DirichletBC(V, inlet, Inlet)
    bc01 = DirichletBC(V, 0, Inlet)
    bc10 = DirichletBC(V, ux, Cyl)
    bc11 = DirichletBC(V, uy, Cyl)
    bc2 = DirichletBC(V, 0, Wall)
    bcp = DirichletBC(Q, 0, Outlet)
    bca = DirichletBC(V, 1, Cyl)
    return dict(u0=[bc00, bc10, bc2], u1=[bc01, bc11, bc2], p=[bcp], alfa=[bca])


def initialize(x_1, x_2, bcs, **NS_namespace):
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(
    mesh,
    V,
    newfolder,
    tstepfiles,
    tstep,
    # ds,  # defined in dolfin -> import
    u_,
    # AssignedVectorFunction,  # defined in utilities -> import
    **NS_namespace
):
    uv = ut.AssignedVectorFunction(u_, name="Velocity")
    omega = Function(V, name="omega")
    # Store omega each save_step
    add_function_to_tstepfiles(omega, newfolder, tstepfiles, tstep)
    ff = MeshFunction("size_t", mesh, mesh.ufl_cell().geometric_dimension() - 1)
    Cyl.mark(ff, 1)
    n = FacetNormal(mesh)
    ds_ = ds[ff]

    return dict(uv=uv, omega=omega, ds=ds_, ff=ff, n=n)


def temporal_hook(
    q_,
    u_,
    tstep,
    V,
    uv,
    p_,
    plot_interval,
    omega,
    ds,
    save_step,
    mesh,
    nu,
    Umean,
    D,
    n,
    **NS_namespace
):
    if tstep % plot_interval == 0:
        uv()
        plt.figure(1)
        plot(uv, title="Velocity")
        plt.figure(2)
        plot(p_, title="Pressure")
        plt.figure(3)
        plot(q_["alfa"], title="alfa")
        plt.show()

    R = VectorFunctionSpace(mesh, "R", 0)
    c = TestFunction(R)
    tau = -p_ * Identity(2) + nu * (grad(u_) + grad(u_).T)
    forces = assemble(dot(dot(tau, n), c) * ds(1)).get_local() * 2 / Umean ** 2 / D

    print("Cd = {}, CL = {}".format(*forces))

    if tstep % save_step == 0:
        try:
            from fenicstools import StreamFunction

            omega.assign(StreamFunction(u_, []))
        except:
            omega.assign(
                project(
                    curl(u_),
                    V,
                    solver_type="cg",
                    bcs=[DirichletBC(V, 0, DomainBoundary())],
                )
            )


def theend_hook(q_, u_, p_, uv, mesh, ds, V, nu, Umean, D, L, **NS_namespace):
    uv()
    plot(uv, title="Velocity")
    plot(p_, title="Pressure")
    plot(q_["alfa"], title="alfa")
    R = VectorFunctionSpace(mesh, "R", 0)
    c = TestFunction(R)
    tau = -p_ * Identity(2) + nu * (grad(u_) + grad(u_).T)
    ff = MeshFunction("size_t", mesh, mesh.ufl_cell().geometric_dimension() - 1)
    Cyl.mark(ff, 1)
    n = FacetNormal(mesh)
    ds = ds[ff]
    forces = assemble(dot(dot(tau, n), c) * ds(1)).get_local() * 2 / Umean ** 2 / D

    print("Cd = {}, CL = {}".format(*forces))

    from fenicstools import Probes
    from numpy import linspace, repeat, where, resize

    xx = linspace(0, L, 10000)
    x = resize(repeat(xx, 2), (10000, 2))
    x[:, 1] = 0.2
    probes = Probes(x.flatten(), V)
    probes(u_[0])
    nmax = where(probes.array() < 0)[0][-1]
    print("L = ", x[nmax, 0] - 0.25)
    print("dP = ", p_(Point(0.15, 0.2)) - p_(Point(0.25, 0.2)))
    print("dP = ", p_(Point(0.15, 0.2)) - p_(Point(0.25, 0.2)))
