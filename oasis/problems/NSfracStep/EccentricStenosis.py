"""
This is the Oasis problem file for the simulations described in "On Delayed
Transition to Turbulence in an Eccentric Stenosis Model for Clean vs. Noisy
High-Fidelity CFD", currently in revision with the Journal of Biomechanics.

The mesh used in the publication is provided in xml format suitable for direct
use with Oasis <http://dx.doi.org/10.6084/m9.figshare.14710932>, and is loaded
by the problem file.

The output data files for the "clean" and "noisy" simulations are also
available at <http://dx.doi.org/10.6084/m9.figshare.14597820> and
<http://dx.doi.org/10.6084/m9.figshare.14597562>, respectively.  They contain
volumetric velocity data in x,y,z directions averaged over 45000 timesteps (4.5
real-time seconds) in .h5 format, readable with an included .xdmf and mesh file
(~2.5M tet elements), also in .h5 format. 18 Reynolds numbers from 100-600 are
included in 'noisy.zip' for both Newtonian and non-Newtonian (not all of which
were presented in the paper), and 7 Reynolds numbers from 600-800 are included
in 'clean zip' for both Newtonian and non-Newtonian. The Noisy non-Newtonian
data also includes time-averaged volumetric viscosity data for each Reynolds
number in .h5 format, along with a readable .xdmf (not presented in paper).
"""

from __future__ import print_function
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

# from oasis.problems.EccentricStenosis import mesh
import dolfin as df
import numpy as np
import random
import os
import platform


def get_problem_parameters(**kwargs):
    Re = float(kwargs.get("Re", 600))
    NS_parameters = dict(
        scalar_components=scalar_components,
        Schmidt=Schmidt,
        Schmidt_T=Schmidt_T,
        nu=0.0031078341013824886,  # mm^2/ms #3.1078341E-6 m^2/s, #0.003372 Pa-s/1085 kg/m^3 this is nu_inf (m^2/s)
        D=6.35,  # 0.00635,
        T=15e3,  # ms
        dt=0.1,  # ms
        Re=Re,
        nn_model="ModifiedCross",
        ModifiedCross=dict(
            lam=3.736e3,  # ms
            m_param=2.406,  # for Non-Newtonian model
            a_param=0.34,  # for Non-Newtonian model
            mu_inf=3.372e-6,  # g/(mm*ms) for non-Newtonian model
            mu_o=9e-5,  # g/(mm*ms) for non-Newtonian model
            rho=1085e-6,  # g/mm^3
        ),
        nu_nn_krylov_solver=dict(
            method="default",
            solver_type="cg",
            preconditioner_type="jacobi",
        ),
        mesh_file="eccentric_stenosis.xml.gz",
        save_step=10,
        velocity_degree=1,
        folder="eccentric_stenosis_results",
        krylov_solvers=dict(
            monitor_convergence=False,
            error_on_nonconvergence=True,
            relative_tolerance=1e-10,
            absolute_tolerance=1e-10,
        ),
        checkpoint=1000,
        print_intermediate_info=100,
        print_velocity_pressure_convergence=False,
    )
    # set default parameters
    for key, val in default_parameters.items():
        if key not in NS_parameters.keys():
            NS_parameters[key] = val

    average_inlet_velocity = get_ave_inlet_velocity(
        NS_parameters["Re"], NS_parameters["nu"], NS_parameters["D"]
    )
    NS_parameters.update(ave_inlet_velocity=average_inlet_velocity)
    inflow_prof = get_inflow_prof(average_inlet_velocity, NS_parameters["D"])
    NS_expressions = dict(u_in=inflow_prof, noise=Noise())
    return NS_parameters, NS_expressions


def mesh(mesh_file, **NS_namespace):
    if not os.path.isfile(mesh_file):
        if platform.system() == "Linux":
            os.system(
                f"wget -O {mesh_file} https://ndownloader.figshare.com/files/28254414"
            )
        elif platform.system() == "Darwin":
            os.system(
                f"curl -L https://ndownloader.figshare.com/files/28254414 -o {mesh_file}"
            )
        else:
            raise ImportError("Could not determine platform")
        print(f"Downloaded mesh {mesh_file}")

    mesh = df.Mesh(mesh_file)
    return mesh


class Noise(df.UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval(self, value, x):
        value[0] = np.random.normal(0, 0.001)


def create_bcs(V, Q, mesh, mesh_file, NS_expressions, **NS_namespace):
    if df.MPI.rank(df.MPI.comm_world) == 0:
        print("Create bcs")

    boundaries = df.MeshFunction("size_t", mesh, 2, mesh.domains())
    boundaries.set_values(boundaries.array() + 1)

    wallId = 1
    inletId = 2
    outletId = 3

    bc0 = df.DirichletBC(V, 0, boundaries, wallId)
    bc1 = df.DirichletBC(V, NS_expressions["u_in"], boundaries, inletId)
    bc2 = df.DirichletBC(V, NS_expressions["noise"], boundaries, inletId)
    bc3 = df.DirichletBC(V, NS_expressions["noise"], boundaries, inletId)
    bc4 = df.DirichletBC(Q, 0, boundaries, outletId)
    return dict(
        u0=[bc0, bc1],  # 0 on the sides, u_in on inlet, zero gradient outlet
        u1=[bc0, bc2],  # 0 on sides and perturbed inlet, zero gradient outlet
        u2=[bc0, bc3],  # 0 on sides and perturbed inlet, zero gradient outlet
        p=[bc4],
    )  # 0 outlet


def initialize(V, q_, q_1, q_2, x_1, x_2, bcs, restart_folder, **NS_namespace):
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(u_, tstep, AssignedVectorFunction, folder, **NS_namespace):
    visfile = df.XDMFFile(
        df.MPI.comm_world,
        os.path.join(folder, "viscosity_from_tstep_{}.xdmf".format(tstep)),
    )
    visfile.parameters["rewrite_function_mesh"] = False
    visfile.parameters["flush_output"] = True
    return dict(uv=AssignedVectorFunction(u_), visfile=visfile)


def velocity_tentative_hook(**NS_namespace):
    pass


def pressure_hook(**NS_namespace):
    pass


def start_timestep_hook(**NS_namespace):
    pass


def temporal_hook(tstep, save_step, visfile, nunn_, folder, **NS_namespace):
    if tstep % save_step == 0:
        visfile.write(nunn_, float(tstep))


def theend_hook(**NS_namespace):
    pass


def get_ave_inlet_velocity(Re, nu, D, **NS_namespace):
    average_inlet_velocity = Re * nu / D
    return average_inlet_velocity


def get_inflow_prof(average_inlet_velocity, D, **NS_namespace):
    u_inflow = df.Expression(
        "A*2*(1-((x[1]*x[1])+(x[2]*x[2]))*4/(D*D))",
        degree=2,
        A=average_inlet_velocity,
        D=D,
    )
    return u_inflow
