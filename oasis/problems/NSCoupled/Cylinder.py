from __future__ import print_function

__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-04"
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
from oasis.problems.NSCoupled import (
    NS_hook,
    start_iter_hook,
    end_iter_hook,
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
    Constant,
    div,
    dx,
)
from os import getcwd, path
import pickle
import matplotlib.pyplot as plt


def get_problem_parameters(**kwargs):
    case = kwargs["case"] if "case" in kwargs else 1
    Um = cases[case]["Um"]
    Re = cases[case]["Re"]
    Umean = 2.0 / 3.0 * Um
    NS_parameters = dict(
        scalar_components=scalar_components + ["c", "d"],
        Schmidt=Schmidt,
        Schmidt_T=Schmidt_T,
        Um=Um,
        Re=Re,
        Umean=Umean,
        H=H,
        L=L,
        D=D,
        nu=Umean * D / Re,
        omega=1.0,
        max_iter=100,
        plot_interval=10,
        velocity_degree=2,
    )

    # set default parameters
    for key, val in default_parameters.items():
        if key not in NS_parameters.keys():
            NS_parameters[key] = val
    return NS_parameters


def scalar_source(c_, d_, **NS_namespace):
    return {"c": -Constant(0.1) * c_ * c_, "d": -Constant(0.25) * c_ * d_ * d_}


def create_bcs(VQ, Um, CG, V, element, **NS_namespace):
    inlet = Expression(
        ("4.*{0}*x[1]*({1}-x[1])/pow({1}, 2)".format(Um, H), "0"), element=V
    )
    ux = Expression(("0.00*x[1]", "-0.00*(x[0]-{})".format(center)), element=V)
    if element == "MINI":
        # This is an inefficient solution due to FFC issue #69, solfin issue #489
        inlet0 = project(inlet, VQ.sub(0).collapse())
        ux0 = project(ux, VQ.sub(0).collapse(), solver_type="cg")
        wall = project(Constant((0, 0)), VQ.sub(0).collapse(), solver_type="cg")
        bc0 = DirichletBC(VQ.sub(0), inlet0, Inlet)
        bc1 = DirichletBC(VQ.sub(0), ux0, Cyl)
        bc2 = DirichletBC(VQ.sub(0), wall, Wall)

    else:
        bc0 = DirichletBC(VQ.sub(0), inlet, Inlet)
        bc1 = DirichletBC(VQ.sub(0), ux, Cyl)
        bc2 = DirichletBC(VQ.sub(0), (0, 0), Wall)
    return dict(
        up=[bc0, bc1, bc2],
        c=[DirichletBC(CG, 1, Cyl), DirichletBC(CG, 0, Inlet)],
        d=[DirichletBC(CG, 2, Cyl), DirichletBC(CG, 0, Inlet)],
    )


def theend_hook(u_, p_, up_, mesh, ds, VQ, nu, Umean, c_, testing, **NS_namespace):
    if not testing:
        plot(u_, title="Velocity")
        plot(p_, title="Pressure")
        plot(c_, title="Scalar")

    R = VectorFunctionSpace(mesh, "R", 0)
    c = TestFunction(R)
    tau = -p_ * Identity(2) + nu * (grad(u_) + grad(u_).T)
    ff = MeshFunction("size_t", mesh, 1, 0)
    Cyl.mark(ff, 1)
    n = FacetNormal(mesh)
    ds = ds(subdomain_data=ff)
    forces = assemble(dot(dot(tau, n), c) * ds(1)).get_local() * 2 / Umean ** 2 / D

    try:
        print("Cd = {0:2.6e}, CL = {1:2.6e}".format(*forces))

    except IndexError:
        pass

    if not testing:
        from fenicstools import Probes
        from numpy import linspace, repeat, where, resize

        xx = linspace(0, L, 10000)
        x = resize(repeat(xx, 2), (10000, 2))
        x[:, 1] = 0.2
        probes = Probes(x.flatten(), VQ)
        probes(up_)
        nmax = where(probes.array()[:, 0] < 0)[0][-1]
        print("L = ", x[nmax, 0] - 0.25)
        print("dP = ", up_(Point(0.15, 0.2))[2] - up_(Point(0.25, 0.2))[2])
        print(
            "Global divergence error ",
            assemble(dot(u_, n) * ds()),
            assemble(div(u_) * div(u_) * dx()),
        )
