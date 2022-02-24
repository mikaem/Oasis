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
    info_red,
    create_bcs,
)
from oasis.problems.NSfracStep import (
    velocity_tentative_hook,
    pressure_hook,
    start_timestep_hook,
    temporal_hook,
)

# from oasis.problems.Channel import mesh
import dolfin as df
from fenicstools import StructuredGrid, Probes
from numpy import arctan, array, cos, pi
from os import getcwd, makedirs, path
import pickle
import random


def get_problem_parameters(**kwargs):
    if "restart_folder" in kwargs.keys():
        # FIXME: this cant work, two different paths are joined.
        restart_folder = kwargs["restart_folder"]
        restart_folder = "my_restart_folder"
        restart_folder = path.join(getcwd(), restart_folder)
        f = open(
            path.join(
                path.dirname(path.abspath(__file__)), restart_folder, "params.dat"
            ),
            "r",
        )
        NS_parameters = pickle.load(f)
        NS_parameters["restart_folder"] = restart_folder
        # globals().update(NS_parameters)
        NS_expressions = {}  # should be loaded as well?
    else:
        nu = 2.0e-5
        Re_tau = 178.12
        Lx = (4.0 * pi,)
        Lz = (4.0 * pi / 3.0,)
        NS_parameters = dict(
            scalar_components=scalar_components,
            Schmidt=Schmidt,
            Schmidt_T=Schmidt_T,
            Lx=Lx,
            Ly=2.0,
            Lz=Lz,
            Nx=16,
            Ny=16,
            Nz=16,
            T=1.0,
            dt=0.2,
            update_statistics=10,
            save_statistics=100,
            check_flux=10,
            checkpoint=100,
            utau=nu * Re_tau,
            save_step=100,
            nu=nu,
            Re_tau=Re_tau,
            velocity_degree=1,
            folder="channel_results",
            use_krylov_solvers=True,
        )

        NS_expressions = dict(constrained_domain=PeriodicDomain(Lx, Lz))
    return NS_parameters, NS_expressions


class ChannelGrid(StructuredGrid):
    """Grid for computing statistics"""

    def modify_mesh(self, dx, dy, dz):
        """Create grid skewed towards the walls located at y = 1 and y = -1"""
        dy[1][:] = (
            arctan(pi * (dy[1][:] + self.origin[1])) / arctan(pi) - self.origin[1]
        )
        return dx, dy, dz


def mesh(Nx, Ny, Nz, Lx, Ly, Lz, **params):
    # Function for creating stretched mesh in y-direction
    m = df.BoxMesh(
        df.Point(0.0, -Ly / 2.0, -Lz / 2.0),
        df.Point(Lx, Ly / 2.0, Lz / 2.0),
        Nx,
        Ny,
        Nz,
    )
    x = m.coordinates()
    x[:, 1] = arctan(1.0 * pi * (x[:, 1])) / arctan(1.0 * pi)
    return m


class PeriodicDomain(df.SubDomain):
    def __init__(self, Lx, Lz):
        self.Lx = Lx
        self.Lz = Lz
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool(
            (df.near(x[0], 0) or df.near(x[2], -self.Lz / 2.0))
            and (not (df.near(x[0], self.Lx) or df.near(x[2], self.Lz / 2.0)))
            and on_boundary
        )

    def map(self, x, y):
        if df.near(x[0], self.Lx) and df.near(x[2], self.Lz / 2.0):
            y[0] = x[0] - self.Lx
            y[1] = x[1]
            y[2] = x[2] - self.Lz
        elif df.near(x[0], self.Lx):
            y[0] = x[0] - self.Lx
            y[1] = x[1]
            y[2] = x[2]
        else:  # near(x[2], Lz/2.):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - self.Lz


def inlet(x, on_bnd):
    return on_bnd and df.near(x[0], 0)


# Specify body force
def body_force(nu, Re_tau, utau, **NS_namespace):
    return df.Constant((utau ** 2, 0.0, 0.0))


def pre_solve_hook(
    V,
    u_,
    mesh,
    AssignedVectorFunction,
    newfolder,
    MPI,
    Nx,
    Ny,
    Nz,
    Lx,
    Ly,
    Lz,
    **NS_namespace
):
    """Called prior to time loop"""
    if MPI.rank(MPI.comm_world) == 0:
        makedirs(path.join(newfolder, "Stats"))

    uv = AssignedVectorFunction(u_)
    tol = 5e-8

    # It's periodic so don't pick the same location twice for sampling statistics:
    stats = ChannelGrid(
        V,
        [Nx, Ny + 1, Nz],
        [tol, -Ly / 2.0, -Lz / 2.0 + tol],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [Lx - Lx / Nx, Ly, Lz - Lz / Nz],
        statistics=True,
    )

    # Create MeshFunction to compute flux
    Inlet = df.AutoSubDomain(inlet)
    facets = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    Inlet.mark(facets, 1)
    normal = df.FacetNormal(mesh)

    return dict(uv=uv, stats=stats, facets=facets, normal=normal)


def create_bcs(V, q_, q_1, q_2, sys_comp, u_components, Ly, **NS_namespace):
    def walls(x, on_bnd):
        return df.near(x[1], -Ly / 2.0) or df.near(x[1], Ly / 2.0)

    info_red("Creating boundary conditions")
    bcs = dict((ui, []) for ui in sys_comp)
    bc = [df.DirichletBC(V, df.Constant(0), walls)]
    bcs["u0"] = bc
    bcs["u1"] = bc
    bcs["u2"] = bc
    return bcs


class RandomStreamVector(df.UserExpression):
    random.seed(2 + df.MPI.rank(df.MPI.comm_world))

    def eval(self, values, x):
        values[0] = 0.0005 * random.random()
        values[1] = 0.0005 * random.random()
        values[2] = 0.0005 * random.random()

    def value_shape(self):
        return (3,)


def initialize(V, q_, q_1, q_2, bcs, restart_folder, utau, nu, **NS_namespace):
    if restart_folder is None:
        # Initialize using a perturbed flow. Create random streamfunction
        Vv = df.VectorFunctionSpace(
            V.mesh(), V.ufl_element().family(), V.ufl_element().degree()
        )
        psi = df.interpolate(RandomStreamVector(element=Vv.ufl_element()), Vv)
        u0 = df.project(df.curl(psi), Vv, solver_type="cg")
        u0x = df.project(u0[0], V, bcs=bcs["u0"], solver_type="cg")
        u1x = df.project(u0[1], V, bcs=bcs["u0"], solver_type="cg")
        u2x = df.project(u0[2], V, bcs=bcs["u0"], solver_type="cg")

        # Create base flow
        y = df.interpolate(
            df.Expression("x[1] > 0 ? 1-x[1] : 1+x[1]", element=V.ufl_element()), V
        )
        uu = df.project(
            (
                1.25
                * (
                    utau
                    / 0.41
                    * df.ln(df.conditional(y < 1e-12, 1.0e-12, y) * utau / nu)
                    + 5.0 * utau
                )
            ),
            V,
            bcs=bcs["u0"],
            solver_type="cg",
        )

        # initialize vectors at two timesteps
        q_1["u0"].vector()[:] = uu.vector()[:]
        q_1["u0"].vector().axpy(1.0, u0x.vector())
        q_1["u1"].vector()[:] = u1x.vector()[:]
        q_1["u2"].vector()[:] = u2x.vector()[:]
        q_2["u0"].vector()[:] = q_1["u0"].vector()[:]
        q_2["u1"].vector()[:] = q_1["u1"].vector()[:]
        q_2["u2"].vector()[:] = q_1["u2"].vector()[:]


def temporal_hook(
    q_,
    u_,
    V,
    tstep,
    uv,
    stats,
    update_statistics,
    newfolder,
    folder,
    check_flux,
    save_statistics,
    mesh,
    facets,
    normal,
    check_if_reset_statistics,
    **NS_namespace
):
    # print timestep
    info_red("tstep = {}".format(tstep))
    if check_if_reset_statistics(folder):
        info_red("Resetting statistics")
        stats.probes.clear()

    if tstep % update_statistics == 0:
        stats(q_["u0"], q_["u1"], q_["u2"])

    if tstep % save_statistics == 0:
        statsfolder = path.join(newfolder, "Stats")
        # stats.toh5(0, tstep, filename=statsfolder +
        #           "/dump_mean_{}.h5".format(tstep))
        stats.tovtk(0, statsfolder + "/dump_mean_{}.vtk".format(tstep))

    if tstep % check_flux == 0:
        u1 = df.assemble(
            df.dot(u_, normal) * df.ds(1, domain=mesh, subdomain_data=facets)
        )
        u1 = df.assemble(
            df.dot(u_, normal) * df.ds(1, domain=mesh, subdomain_data=facets)
        )
        normv = df.norm(q_["u1"].vector())
        normw = df.norm(q_["u2"].vector())
        if df.MPI.rank(df.MPI.comm_world) == 0:
            print("Flux = ", u1, " tstep = ", tstep, " norm = ", normv, normw)


def theend(newfolder, tstep, stats, **NS_namespace):
    """Store statistics before exiting"""
    statsfolder = path.join(newfolder, "Stats")
    stats.toh5(0, tstep, filename=statsfolder + "/dump_mean_{}.h5".format(tstep))
