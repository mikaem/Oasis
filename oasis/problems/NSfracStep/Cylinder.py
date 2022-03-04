#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:01:22 2022

@author: florianma
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-03-21"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"


from oasis.problems import (
    add_function_to_tstepfiles,
    constrained_domain,  # might get overloaded
    scalar_components,
    Schmidt,
    Schmidt_T,
    body_force,  # might get overloaded
    initialize,
    scalar_hook,  # might get overloaded
    scalar_source,  # might get overloaded
    pre_solve_hook,
    theend_hook,
    # get_problem_parameters,  # alwas overloaded
    # post_import_problem,  # never overloaded
    create_bcs,
)
import oasis.common.utilities as ut
from oasis.problems.NSfracStep import (
    velocity_tentative_hook,
    pressure_hook,
    start_timestep_hook,
    temporal_hook,
    default_parameters,
    FracDomain,
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
    curl,
    DomainBoundary,
    Point,
    ds,
    Mesh,
    XDMFFile,
    MeshValueCollection,
    Constant,
    cpp,
)
from os import getcwd, path
import pickle
import matplotlib.pyplot as plt


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
        case = kwargs["case"] if "case" in kwargs else 1
        Um = cases[case]["Um"]  # 0.3 or 1.5
        # Re = cases[case]["Re"]  # 20 or 100
        Umean = 2.0 / 3.0 * Um
        rho = 1.0
        mu = 0.001
        Re = rho * Umean * D / mu
        nu = mu / rho
        NS_parameters = default_parameters
        NS_parameters["Schmidt"] = Schmidt
        NS_parameters["Schmidt_T"] = Schmidt_T
        NS_parameters["H"] = H
        NS_parameters["Um"] = Um
        NS_parameters["Re"] = Re
        NS_parameters["nu"] = nu
        NS_parameters["Umean"] = Umean
        NS_parameters["T"] = 100
        NS_parameters["dt"] = 0.01
        NS_parameters["checkpoint"] = 50
        NS_parameters["save_step"] = 50
        NS_parameters["plot_interval"] = 10
        NS_parameters["velocity_degree"] = 2
        NS_parameters["print_intermediate_info"] = 100
        NS_parameters["use_krylov_solvers"] = True

        NS_parameters["scalar_components"] = scalar_components + ["alfa"]
        NS_parameters["Schmidt"]["alfa"] = 0.1
        NS_parameters["krylov_solvers"]["monitor_convergence"] = True
        NS_parameters["velocity_krylov_solver"]["preconditioner_type"] = "jacobi"
        NS_parameters["velocity_krylov_solver"]["solver_type"] = "bicgstab"
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


class Cylinder(FracDomain):
    # run for Um in [0.2, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0]
    # or  for Re in [20., 50., 60., 75.0, 100, 150, 200]
    def __init__(self, case=1):
        """
        Create the required function spaces, functions and boundary conditions
        for a channel flow problem
        """
        super().__init__()
        # problem parameters
        # case = parameters["case"] if "case" in parameters else 1
        Umax = cases[case]["Um"]  # 0.3 or 1.5
        # Re = cases[case]["Re"]  # 20 or 100
        Umean = 2.0 / 3.0 * Umax
        rho = 1.0
        mu = 0.001
        self.H = 0.41
        Re = rho * Umean * D / mu
        print("Re", Re)
        nu = mu / rho
        self.Umean = Umean
        self.Umax = Umax
        self.Schmidt = {}
        self.Schmidt_T = {}
        self.nu = nu
        self.T = 10
        self.dt = 0.01
        self.checkpoint = 50
        self.save_step = 50
        self.plot_interval = 10
        self.velocity_degree = 2
        self.print_intermediate_info = 100
        self.use_krylov_solvers = True
        # self.krylov_solvers["monitor_convergence"] = True
        self.velocity_krylov_solver["preconditioner_type"] = "jacobi"
        self.velocity_krylov_solver["solver_type"] = "bicgstab"
        self.NS_expressions = {}
        self.scalar_components = []
        return

    def mesh_from_file(self, mesh_name, facet_name):
        self.mesh = Mesh()
        with XDMFFile(mesh_name) as infile:
            infile.read(self.mesh)

        mvc = MeshValueCollection("size_t", self.mesh, self.mesh.topology().dim() - 1)
        with XDMFFile(facet_name) as infile:
            infile.read(mvc, "name_to_read")
        mf = self.mf = cpp.mesh.MeshFunctionSizet(self.mesh, mvc)
        self.bc_dict = {
            "fluid": 0,
            "channel_walls": 1,
            "cylinderwall": 2,
            "inlet": 3,
            "outlet": 4,
        }
        return

    def create_bcs(self):
        mf, bc_dict = self.mf, self.bc_dict
        V, Q, Umax, H = self.V, self.Q, self.Umax, self.H
        # U0_str = "4.0*x[1]*(0.41-x[1])/0.1681"
        U0_str = "4.*{0}*x[1]*({1}-x[1])/pow({1}, 2)".format(Umax, H)
        inlet = Expression(U0_str, degree=2)
        bc00 = DirichletBC(V, inlet, mf, bc_dict["inlet"])
        bc01 = DirichletBC(V, 0.0, mf, bc_dict["inlet"])
        bc10 = bc11 = DirichletBC(V, 0.0, mf, bc_dict["cylinderwall"])
        bc2 = DirichletBC(V, 0.0, mf, bc_dict["channel_walls"])
        bcp = DirichletBC(Q, 0.0, mf, bc_dict["outlet"])
        self.bcs = {
            "u0": [bc00, bc10, bc2],
            "u1": [bc01, bc11, bc2],
            "p": [bcp],
        }
        return

    def temporal_hook(self, t, tstep, **kvargs):
        if tstep % 100 == 0:
            fig, (ax1, ax2) = self.plot()
            plt.savefig("../results/frame_{:06d}.png".format(tstep))
            plt.close()
        return

    def theend_hook(self):
        print("finished :)")

    def plot(self):
        # u, p = self.u_, self.p_
        mesh = self.mesh
        u = self.q_["u0"].compute_vertex_values(mesh)
        v = self.q_["u1"].compute_vertex_values(mesh)
        p = self.q_["p"].compute_vertex_values(mesh)
        # print(u.shape, v.shape, p.shape)
        magnitude = (u ** 2 + v ** 2) ** 0.5
        # print(u.shape, v.shape, p.shape, magnitude.shape)

        # velocity = u.compute_vertex_values(mesh)
        # velocity.shape = (2, -1)
        # magnitude = np.linalg.norm(velocity, axis=0)
        x, y = mesh.coordinates().T
        # u, v = velocity
        tri = mesh.cells()
        # pressure = p.compute_vertex_values(mesh)
        # print(x.shape, y.shape, u.shape, v.shape)

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, figsize=(12, 6))
        ax1.quiver(x, y, u, v, magnitude)
        ax2.tricontourf(x, y, tri, p, levels=40)
        ax1.set_aspect("equal")
        ax2.set_aspect("equal")
        ax1.set_title("velocity")
        ax2.set_title("pressure")
        return fig, (ax1, ax2)
