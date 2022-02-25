#!/usr/bin/env python

__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-06"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

"""
This module implements a generic form of the fractional step method for
solving the incompressible Navier-Stokes equations. There are several
possible implementations of the pressure correction and the more low-level
details are chosen at run-time and imported from any one of:

  solvers/NSfracStep/IPCS_ABCN.py    # Implicit convection
  solvers/NSfracStep/IPCS_ABE.py     # Explicit convection
  solvers/NSfracStep/IPCS.py         # Naive implict convection
  solvers/NSfracStep/BDFPC.py        # Naive Backwards Differencing IPCS in rotational form
  solvers/NSfracStep/BDFPC_Fast.py   # Fast Backwards Differencing IPCS in rotational form
  solvers/NSfracStep/Chorin.py       # Naive

The naive solvers are very simple and not optimized. They are intended
for validation of the other optimized versions. The fractional step method
can be used both non-iteratively or with iterations over the pressure-
velocity system.

The velocity vector is segregated, and we use three (in 3D) scalar
velocity components.

Each new problem needs to implement a new problem module to be placed in
the problems/NSfracStep folder. From the problems module one needs to import
a mesh and a control dictionary called NS_parameters. See
problems/NSfracStep/__init__.py for all possible parameters.

"""
import importlib
import oasis.common.io as io
import pickle
from os import path
import copy
import dolfin as df
import numpy as np
from oasis.common import parse_command_line  # ,convert
import oasis.common.utilities as ut
from ufl import Coefficient

from oasis.problems import (
    info_blue,
    info_green,
    info_red,
    OasisTimer,
    initial_memory_use,
    oasis_memory,
    post_import_problem,
)

commandline_kwargs = parse_command_line()

# Find the problem module
default_problem = "DrivenCavity"
problemname = commandline_kwargs.get("problem", default_problem)
# Import the problem module
print("Importing problem module " + problemname)
if problemname == "Channel":
    import oasis.problems.NSfracStep.Channel as pblm
elif problemname == "Cylinder":
    import oasis.problems.NSfracStep.Cylinder as pblm
if problemname == "DrivenCavity":
    import oasis.problems.NSfracStep.DrivenCavity as pblm
elif problemname == "DrivenCavity3D":
    import oasis.problems.NSfracStep.DrivenCavity3D as pblm
elif problemname == "EccentricStenosis":
    import oasis.problems.NSfracStep.EccentricStenosis as pblm
elif problemname == "FlowPastSphere3D":
    import oasis.problems.NSfracStep.FlowPastSphere3D as pblm
elif problemname == "LaminarChannel":
    import oasis.problems.NSfracStep.LaminarChannel as pblm
elif problemname == "Lshape":
    import oasis.problems.NSfracStep.Lshape as pblm
elif problemname == "Skewed2D":
    import oasis.problems.NSfracStep.Skewed2D as pblm
elif problemname == "SkewedFlow":
    import oasis.problems.NSfracStep.SkewedFlow as pblm
elif problemname == "TaylorGreen2D":
    import oasis.problems.NSfracStep.TaylorGreen2D as pblm
elif problemname == "TaylorGreen3D":
    import oasis.problems.NSfracStep.TaylorGreen3D as pblm

NS_namespace, NS_expressions = pblm.get_problem_parameters()

# problem_parameters used to be NS_parameters
NS_namespace, problem_parameters = post_import_problem(
    NS_namespace, NS_expressions, pblm.mesh, commandline_kwargs
)
scalar_components = problem_parameters["scalar_components"]
mesh = NS_namespace["mesh"]

# Use t and tstep from stored paramteres if restarting
if problem_parameters["restart_folder"] is not None:
    folder = path.abspath(NS_namespace["restart_folder"])
    f = open(path.join(folder, "params.dat"), "rb")
    params = pickle.load(f)
    f.close()
    t = params["t"]
    tstep = params["tstep"]

# Import chosen functionality from solvers
solvername = problem_parameters["solver"]
if solvername == "BDFPC":
    import oasis.solvers.NSfracStep.BDFPC as solver
elif solvername == "BDFPC_Fast":
    import oasis.solvers.NSfracStep.BDFPC_Fast as solver
elif solvername == "Chorin":
    import oasis.solvers.NSfracStep.Chorin as solver
elif solvername == "IPCS":
    import oasis.solvers.NSfracStep.IPCS as solver
elif solvername == "IPCS_ABCN":
    import oasis.solvers.NSfracStep.IPCS_ABCN as solver
elif solvername == "IPCS_ABE":
    import oasis.solvers.NSfracStep.IPCS_ABE as solver

# Create lists of components solved for
dim = mesh.geometry().dim()
u_components = ["u" + str(x) for x in range(dim)]
sys_comp = u_components + ["p"] + scalar_components
uc_comp = u_components + scalar_components

# Set up initial folders for storing results
newfolder, tstepfiles = io.create_initial_folders(
    sys_comp=sys_comp,
    **problem_parameters,
)
NS_namespace["newfolder"] = newfolder
NS_namespace["tstepfiles"] = tstepfiles

# Declare FunctionSpaces and arguments
velocity_degree = problem_parameters["velocity_degree"]
pressure_degree = problem_parameters["pressure_degree"]
V = Q = df.FunctionSpace(
    mesh, "CG", velocity_degree, constrained_domain=pblm.constrained_domain
)
if velocity_degree != pressure_degree:
    Q = df.FunctionSpace(
        mesh, "CG", pressure_degree, constrained_domain=pblm.constrained_domain
    )
NS_namespace["V"] = V
NS_namespace["Q"] = Q
NS_namespace["u"] = u = df.TrialFunction(V)
NS_namespace["v"] = v = df.TestFunction(V)
NS_namespace["p"] = p = df.TrialFunction(Q)
NS_namespace["q"] = q = df.TestFunction(Q)

# Use dictionary to hold all FunctionSpaces
VV = dict((ui, V) for ui in uc_comp)
VV["p"] = Q

# Create dictionaries for the solutions at three timesteps
q_ = dict((ui, df.Function(VV[ui], name=ui)) for ui in sys_comp)
q_1 = dict((ui, df.Function(VV[ui], name=ui + "_1")) for ui in sys_comp)
q_2 = dict((ui, df.Function(V, name=ui + "_2")) for ui in u_components)
NS_namespace["q_"], NS_namespace["q_1"], NS_namespace["q_2"] = q_, q_1, q_2

# Read in previous solution if restarting
io.init_from_restart(
    sys_comp=sys_comp,
    uc_comp=uc_comp,
    u_components=u_components,
    **problem_parameters,
    **NS_namespace,
)

# Create vectors of the segregated velocity components
u_ = df.as_vector([q_[ui] for ui in u_components])  # Velocity vector at t
u_1 = df.as_vector([q_1[ui] for ui in u_components])  # Velocity vector at t - dt
u_2 = df.as_vector([q_2[ui] for ui in u_components])  # Velocity vector at t - 2*dt
NS_namespace["u_"], NS_namespace["u_1"], NS_namespace["u_2"] = u_, u_1, u_2

# Adams Bashforth projection of velocity at t - dt/2
U_AB = 1.5 * u_1 - 0.5 * u_2

# Create short forms for accessing the solution vectors
x_ = dict((ui, q_[ui].vector()) for ui in sys_comp)  # Solution vectors t
x_1 = dict((ui, q_1[ui].vector()) for ui in sys_comp)  # Solution vectors t - dt
x_2 = dict((ui, q_2[ui].vector()) for ui in u_components)  # Solution vectors t - 2*dt
NS_namespace["x_"], NS_namespace["x_1"], NS_namespace["x_2"] = x_, x_1, x_2

# Create vectors to hold rhs of equations
b = dict((ui, df.Vector(x_[ui])) for ui in sys_comp)  # rhs vectors (final)
b_tmp = dict((ui, df.Vector(x_[ui])) for ui in sys_comp)  # rhs temp storage vectors
NS_namespace["b"], NS_namespace["b_tmp"] = b, b_tmp

# Short forms pressure and scalars
NS_namespace["p_"] = q_["p"]  # pressure at t
NS_namespace["p_1"] = q_1["p"]  # pressure at t - dt
NS_namespace["dp_"] = dp_ = df.Function(Q)  # pressure correction
for ci in scalar_components:
    exec("{}_   = q_ ['{}']".format(ci, ci))
    exec("{}_1  = q_1['{}']".format(ci, ci))

krylov_solvers = problem_parameters["krylov_solvers"]
use_krylov_solvers = problem_parameters["use_krylov_solvers"]
print_solve_info = use_krylov_solvers and krylov_solvers["monitor_convergence"]

# Boundary conditions
NS_namespace["bcs"] = bcs = pblm.create_bcs(**problem_parameters, **NS_namespace)

# LES setup
les_modelname = problem_parameters["les_model"]
if les_modelname == "DynamicLagrangian":
    import oasis.solvers.NSfracStep.LES.DynamicLagrangian as lesmodel
elif les_modelname == "DynamicModules":
    import oasis.solvers.NSfracStep.LES.DynamicModules as lesmodel
elif les_modelname == "KineticEnergySGS":
    import oasis.solvers.NSfracStep.LES.KineticEnergySGS as lesmodel
elif les_modelname == "NoModel":
    import oasis.solvers.NSfracStep.LES.NoModel as lesmodel
elif les_modelname == "ScaleDepDynamicLagrangian":
    import oasis.solvers.NSfracStep.LES.ScaleDepDynamicLagrangian as lesmodel
elif les_modelname == "Smagorinsky":
    import oasis.solvers.NSfracStep.LES.Smagorinsky as lesmodel
elif les_modelname == "Wale":
    import oasis.solvers.NSfracStep.LES.Wale as lesmodel

les_dict = lesmodel.les_setup()  # FIXME: which dicts have to be passed?
NS_namespace.update(les_dict)

# Non-Newtonian setup
nn_modelname = problem_parameters["nn_model"]
if nn_modelname == "NoModel":
    import oasis.solvers.NSfracStep.NNModel.NoModel as nnmodel
elif nn_modelname == "ModifiedCross":
    import oasis.solvers.NSfracStep.NNModel.ModifiedCross as nnmodel

nn_dict = nnmodel.nn_setup()  # FIXME: which dicts have to be passed?
NS_namespace.update(nn_dict)

# Initialize solution
pblm.initialize(**NS_namespace)

#  Fetch linear algebra solvers
u_sol, p_sol, c_sol = solver.get_solvers(**problem_parameters, **NS_namespace)

# Get constant body forces
f = pblm.body_force(**NS_namespace)
assert isinstance(f, Coefficient)
b0 = dict((ui, df.assemble(v * f[i] * df.dx)) for i, ui in enumerate(u_components))
NS_namespace["f"], NS_namespace["b0"] = f, b0

# Get scalar sources
fs = pblm.scalar_source(**problem_parameters)
for ci in scalar_components:
    assert isinstance(fs[ci], Coefficient)
    b0[ci] = df.assemble(v * fs[ci] * df.dx)
NS_namespace["fs"] = fs

# Preassemble and allocate
# TODO: ut.XXX should not be passed but raather imported in solver.py
F_dict = solver.setup(
    A_cache=ut.A_cache,
    homogenize=ut.homogenize,
    GradFunction=ut.GradFunction,
    DivFunction=ut.DivFunction,
    LESsource=ut.LESsource,
    NNsource=ut.NNsource,
    assemble_matrix=ut.assemble_matrix,
    u_components=u_components,
    **problem_parameters,
    **NS_namespace,
)
NS_namespace.update(F_dict)

t = problem_parameters["t"]
tstep = problem_parameters["tstep"]
T = problem_parameters["T"]
max_iter = problem_parameters["max_iter"]
it0 = problem_parameters["iters_on_first_timestep"]
max_error = problem_parameters["max_error"]
print_intermediate_info = problem_parameters["print_intermediate_info"]
AB_projection_pressure = problem_parameters["AB_projection_pressure"]

# Anything problem specific
psh_dict = pblm.pre_solve_hook(**problem_parameters, **NS_namespace)
NS_namespace.update(psh_dict)

tx = OasisTimer("Timestep timer")
tx.start()
stop = False
total_timer = OasisTimer("Start simulations", True)
while t < (T - tstep * df.DOLFIN_EPS) and not stop:
    t += problem_parameters["dt"]
    tstep += 1
    # TODO: maybe t and tstep should not be in any dictionary, since it changes
    problem_parameters["t"] = t
    problem_parameters["tstep"] = tstep
    inner_iter = 0
    udiff = np.array([1e8])  # Norm of velocity change over last inner iter
    num_iter = max(it0, max_iter) if tstep <= 10 else max_iter
    pblm.start_timestep_hook()  # FIXME: what do we need to pass here?

    while udiff[0] > max_error and inner_iter < num_iter:
        inner_iter += 1

        t0 = OasisTimer("Tentative velocity")
        if inner_iter == 1:
            lesmodel.les_update(**NS_namespace)
            nnmodel.nn_update(**NS_namespace)
            solver.assemble_first_inner_iter(
                u_components=u_components,
                **problem_parameters,
                **NS_namespace,
            )
        udiff[0] = 0.0
        for i, ui in enumerate(u_components):
            t1 = OasisTimer("Solving tentative velocity " + ui, print_solve_info)
            solver.velocity_tentative_assemble(ui=ui, **NS_namespace)
            pblm.velocity_tentative_hook(ui=ui, **NS_namespace)
            solver.velocity_tentative_solve(
                udiff=udiff,
                ui=ui,
                u_sol=u_sol,
                **problem_parameters,
                **NS_namespace,
            )
            t1.stop()
        t0 = OasisTimer("Pressure solve", print_solve_info)
        solver.pressure_assemble(**problem_parameters, **NS_namespace)
        pblm.pressure_hook(**NS_namespace)
        solver.pressure_solve(p_sol=p_sol, **NS_namespace)
        t0.stop()

        solver.print_velocity_pressure_info(
            num_iter=num_iter,
            norm=df.norm,
            info_blue=info_blue,
            inner_iter=inner_iter,
            udiff=udiff,
            **problem_parameters,
            **NS_namespace,
        )

    # Update velocity
    t0 = OasisTimer("Velocity update")
    solver.velocity_update(
        u_components=u_components,
        **problem_parameters,
        **NS_namespace,
    )
    t0.stop()

    # Solve for scalars
    if len(scalar_components) > 0:
        solver.scalar_assemble(**problem_parameters, **NS_namespace)
        for ci in scalar_components:
            t1 = OasisTimer("Solving scalar {}".format(ci), print_solve_info)
            pblm.scalar_hook()  # FIXME: what do we need to pass here?
            solver.scalar_solve(
                c_sol=c_sol, ci=ci, **problem_parameters, **NS_namespace
            )
            t1.stop()
    pblm.temporal_hook(**problem_parameters, **NS_namespace)

    # Save solution if required and check for killoasis file
    stop = io.save_solution(
        u_components=u_components,
        NS_parameters=problem_parameters,
        constrained_domain=pblm.constrained_domain,
        AssignedVectorFunction=ut.AssignedVectorFunction,
        total_timer=total_timer,
        **problem_parameters,
        **NS_namespace,
    )
    # Update to a new timestep
    for ui in u_components:
        x_2[ui].zero()
        x_2[ui].axpy(1.0, x_1[ui])
        x_1[ui].zero()
        x_1[ui].axpy(1.0, x_[ui])

    for ci in scalar_components:
        x_1[ci].zero()
        x_1[ci].axpy(1.0, x_[ci])

    # Print some information
    if tstep % print_intermediate_info == 0:
        toc = tx.stop()
        info_green(
            "Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}".format(
                t, tstep, T
            )
        )
        info_red(
            "Total computing time on previous {0:d} timesteps = {1:f}".format(
                print_intermediate_info, toc
            )
        )
        solver.list_timings(df.TimingClear.clear, [df.TimingType.wall])
        tx.start()

    # AB projection for pressure on next timestep
    if AB_projection_pressure and t < (T - tstep * df.DOLFIN_EPS) and not stop:
        x_["p"].axpy(0.5, dp_.vector())

total_timer.stop()
solver.list_timings(df.TimingClear.keep, [df.TimingType.wall])
info_red("Total computing time = {0:f}".format(total_timer.elapsed()[0]))
oasis_memory("Final memory use ")
total_initial_dolfin_memory = df.MPI.sum(df.MPI.comm_world, initial_memory_use)
info_red(
    "Memory use for importing dolfin = {} MB (RSS)".format(total_initial_dolfin_memory)
)
info_red(
    "Total memory use of solver = "
    + str(oasis_memory.memory - total_initial_dolfin_memory)
    + " MB (RSS)"
)

if problem_parameters["restart_folder"] is not None:
    io.merge_visualization_files(newfolder=newfolder)

# Final hook
pblm.theend_hook(**problem_parameters, **NS_namespace)

# F_dict.keys() = dict_keys(['A', 'M', 'K', 'Ap', 'divu', 'gradp', 'Ta', 'Tb', 'bb', 'bx', 'u_ab', 'a_conv', 'a_scalar', 'LT', 'KT', 'NT'])
# nn_dict.keys() = dict_keys(['nunn_'])
# les_dict.keys() = dict_keys(['nut_'])
# quantities.keys() = dict_keys(['u', 'u_', 'u_1', 'u_2', 'x_', 'x_1', 'x_2', 'b', 'b_tmp', 'p', 'p_', 'p_1', 'dp_', 'q', 'q_', 'q_1', 'q_2', 'v', 'V', 'Q', 'f', 'fs', 'bcs', 'b0'])
# NS_parameters.keys() = dict_keys(['nu', 'folder', 'velocity_degree', 'pressure_degree', 't', 'tstep', 'T', 'dt', 'AB_projection_pressure', 'solver', 'max_iter', 'max_error', 'iters_on_first_timestep', 'use_krylov_solvers', 'print_intermediate_info', 'print_velocity_pressure_convergence', 'plot_interval', 'checkpoint', 'save_step', 'restart_folder', 'output_timeseries_as_vector', 'killtime', 'les_model', 'Smagorinsky', 'Wale', 'DynamicSmagorinsky', 'KineticEnergySGS', 'nn_model', 'ModifiedCross', 'testing', 'krylov_solvers', 'velocity_update_solver', 'velocity_krylov_solver', 'pressure_krylov_solver', 'scalar_krylov_solver', 'nut_krylov_solver', 'nu_nn_krylov_solver'])
# psh_dict.keys() = dict_keys(['uv'])
# NS_parameters.keys() = dict_keys(['nu', 'folder', 'velocity_degree', 'pressure_degree', 't', 'tstep', 'T', 'dt', 'AB_projection_pressure', 'solver', 'max_iter', 'max_error', 'iters_on_first_timestep', 'use_krylov_solvers', 'print_intermediate_info', 'print_velocity_pressure_convergence', 'plot_interval', 'checkpoint', 'save_step', 'restart_folder', 'output_timeseries_as_vector', 'killtime', 'les_model', 'Smagorinsky', 'Wale', 'DynamicSmagorinsky', 'KineticEnergySGS', 'nn_model', 'ModifiedCross', 'testing', 'krylov_solvers', 'velocity_update_solver', 'velocity_krylov_solver', 'pressure_krylov_solver', 'scalar_krylov_solver', 'nut_krylov_solver', 'nu_nn_krylov_solver'])
# psh_dict.keys() = dict_keys(['uv'])
