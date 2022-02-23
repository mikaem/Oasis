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
)

commandline_kwargs = parse_command_line()

# Find the problem module
default_problem = "DrivenCavity"  # Cylinder, DrivenCavity
problemname = commandline_kwargs.get("problem", default_problem)
pth = ".".join(("oasis.problems.NSfracStep", problemname))
problemspec = importlib.util.find_spec(pth)
if problemspec is None:
    problemspec = importlib.util.find_spec(problemname)
if problemspec is None:
    raise RuntimeError(problemname + " not found")

# Import the problem module
print("Importing problem module " + problemname + ":\n" + problemspec.origin)
pblm = importlib.util.module_from_spec(problemspec)
problemspec.loader.exec_module(pblm)

# what may be overwritten and what may not?
NS_parameters = pblm.NS_parameters
scalar_components = pblm.scalar_components  # TODO: can that be in NS_parameters?
temporal_hook = pblm.temporal_hook
theend_hook = pblm.theend_hook

# updates NS_parameters, scalar_components, Schmidt
pblm.problem_parameters(NS_parameters, scalar_components, pblm.Schmidt)

# Update current namespace with NS_parameters and commandline_kwargs ++
# updates NS_parameters!
mesh = pblm.post_import_problem(
    NS_parameters,
    pblm.mesh,
    commandline_kwargs,
    pblm.NS_expressions,
)["mesh"]

# Use t and tstep from stored paramteres if restarting
if NS_parameters["restart_folder"] is not None:
    folder = path.abspath(NS_parameters["restart_folder"])
    f = open(path.join(folder, "params.dat"), "rb")
    params = pickle.load(f)
    f.close()
    t = params["t"]
    tstep = params["tstep"]

# Import chosen functionality from solvers
pth = ".".join(("oasis.solvers.NSfracStep", NS_parameters["solver"]))
solver = importlib.import_module(pth)

# Create lists of components solved for
dim = mesh.geometry().dim()
u_components = ["u" + str(x) for x in range(dim)]
sys_comp = u_components + ["p"] + scalar_components
uc_comp = u_components + scalar_components

# Set up initial folders for storing results
newfolder, tstepfiles = io.create_initial_folders(
    scalar_components=scalar_components,
    sys_comp=sys_comp,
    info_red=info_red,
    **NS_parameters,
)

quantities = {}
# Declare FunctionSpaces and arguments
velocity_degree = NS_parameters["velocity_degree"]
pressure_degree = NS_parameters["pressure_degree"]
V = Q = df.FunctionSpace(
    mesh, "CG", velocity_degree, constrained_domain=pblm.constrained_domain
)
if velocity_degree != pressure_degree:
    Q = df.FunctionSpace(
        mesh, "CG", pressure_degree, constrained_domain=pblm.constrained_domain
    )
quantities["V"] = V
quantities["Q"] = Q
quantities["u"] = u = df.TrialFunction(V)
quantities["v"] = v = df.TestFunction(V)
quantities["p"] = p = df.TrialFunction(Q)
quantities["q"] = q = df.TestFunction(Q)

# Use dictionary to hold all FunctionSpaces
VV = dict((ui, V) for ui in uc_comp)
VV["p"] = Q

# Create dictionaries for the solutions at three timesteps
q_ = dict((ui, df.Function(VV[ui], name=ui)) for ui in sys_comp)
q_1 = dict((ui, df.Function(VV[ui], name=ui + "_1")) for ui in sys_comp)
q_2 = dict((ui, df.Function(V, name=ui + "_2")) for ui in u_components)
quantities["q_"], quantities["q_1"], quantities["q_2"] = q_, q_1, q_2

# Read in previous solution if restarting
io.init_from_restart(
    sys_comp=sys_comp,
    uc_comp=uc_comp,
    u_components=u_components,
    **quantities,
    **NS_parameters,
)

# Create vectors of the segregated velocity components
u_ = df.as_vector([q_[ui] for ui in u_components])  # Velocity vector at t
u_1 = df.as_vector([q_1[ui] for ui in u_components])  # Velocity vector at t - dt
u_2 = df.as_vector([q_2[ui] for ui in u_components])  # Velocity vector at t - 2*dt
quantities["u_"], quantities["u_1"], quantities["u_2"] = u_, u_1, u_2

# Adams Bashforth projection of velocity at t - dt/2
U_AB = 1.5 * u_1 - 0.5 * u_2

# Create short forms for accessing the solution vectors
x_ = dict((ui, q_[ui].vector()) for ui in sys_comp)  # Solution vectors t
x_1 = dict((ui, q_1[ui].vector()) for ui in sys_comp)  # Solution vectors t - dt
x_2 = dict((ui, q_2[ui].vector()) for ui in u_components)  # Solution vectors t - 2*dt
quantities["x_"], quantities["x_1"], quantities["x_2"] = x_, x_1, x_2

# Create vectors to hold rhs of equations
b = dict((ui, df.Vector(x_[ui])) for ui in sys_comp)  # rhs vectors (final)
b_tmp = dict((ui, df.Vector(x_[ui])) for ui in sys_comp)  # rhs temp storage vectors
quantities["b"], quantities["b_tmp"] = b, b_tmp

# Short forms pressure and scalars
quantities["p_"] = q_["p"]  # pressure at t
quantities["p_1"] = q_1["p"]  # pressure at t - dt
quantities["dp_"] = dp_ = df.Function(Q)  # pressure correction
for ci in scalar_components:
    exec("{}_   = q_ ['{}']".format(ci, ci))
    exec("{}_1  = q_1['{}']".format(ci, ci))

krylov_solvers = NS_parameters["krylov_solvers"]
use_krylov_solvers = NS_parameters["use_krylov_solvers"]
print_solve_info = use_krylov_solvers and krylov_solvers["monitor_convergence"]

# Boundary conditions
quantities["bcs"] = bcs = pblm.create_bcs(V=V)

# LES setup
pth = ".".join(("oasis.solvers.NSfracStep.LES", NS_parameters["les_model"]))
lesmodel = importlib.import_module(pth)
les_dict = lesmodel.les_setup()  # FIXME: which dicts have to be passed?

# Non-Newtonian setup
# exec("from oasis.solvers.NSfracStep.NNModel.{} import *".format(nn_model))
pth = ".".join(("oasis.solvers.NSfracStep.NNModel", NS_parameters["nn_model"]))
nnmodel = importlib.import_module(pth)
nn_dict = nnmodel.nn_setup()  # FIXME: which dicts have to be passed?

# Initialize solution
pblm.initialize(**quantities, **NS_parameters)

#  Fetch linear algebra solvers
u_sol, p_sol, c_sol = solver.get_solvers(
    **quantities,
    scalar_components=scalar_components,
    **NS_parameters,
)

# Get constant body forces
f = pblm.body_force(mesh=mesh, **NS_parameters)
assert isinstance(f, Coefficient)
b0 = dict((ui, df.assemble(v * f[i] * df.dx)) for i, ui in enumerate(u_components))
quantities["f"], quantities["b0"] = f, b0

# Get scalar sources
fs = pblm.scalar_source(scalar_components=scalar_components, **NS_parameters)
for ci in scalar_components:
    assert isinstance(fs[ci], Coefficient)
    b0[ci] = df.assemble(v * fs[ci] * df.dx)
quantities["fs"] = fs

# Preassemble and allocate
# TODO: ut.XXX should not be passed but raather imported in solver.py
F_dict = solver.setup(
    scalar_components=scalar_components,
    A_cache=ut.A_cache,
    homogenize=ut.homogenize,
    GradFunction=ut.GradFunction,
    DivFunction=ut.DivFunction,
    LESsource=ut.LESsource,
    NNsource=ut.NNsource,
    assemble_matrix=ut.assemble_matrix,
    u_components=u_components,
    **nn_dict,
    **les_dict,
    **quantities,
    **NS_parameters,
)

t = NS_parameters["t"]
tstep = NS_parameters["tstep"]
T = NS_parameters["T"]
max_iter = NS_parameters["max_iter"]
it0 = NS_parameters["iters_on_first_timestep"]
max_error = NS_parameters["max_error"]
print_intermediate_info = NS_parameters["print_intermediate_info"]
AB_projection_pressure = NS_parameters["AB_projection_pressure"]

# Anything problem specific
psh_dict = pblm.pre_solve_hook(mesh=mesh, velocity_degree=velocity_degree)

tx = OasisTimer("Timestep timer")
tx.start()
stop = False
total_timer = OasisTimer("Start simulations", True)
while t < (T - tstep * df.DOLFIN_EPS) and not stop:
    t += NS_parameters["dt"]
    tstep += 1
    NS_parameters["t"] = t
    NS_parameters["tstep"] = tstep
    inner_iter = 0
    udiff = np.array([1e8])  # Norm of velocity change over last inner iter
    num_iter = max(it0, max_iter) if tstep <= 10 else max_iter
    pblm.start_timestep_hook()  # FIXME: what do we need to pass here?

    while udiff[0] > max_error and inner_iter < num_iter:
        inner_iter += 1

        t0 = OasisTimer("Tentative velocity")
        if inner_iter == 1:
            lesmodel.les_update(**les_dict, **NS_parameters)
            nnmodel.nn_update(**nn_dict, **NS_parameters)
            solver.assemble_first_inner_iter(
                scalar_components=scalar_components,
                u_components=u_components,
                **nn_dict,
                **les_dict,
                **F_dict,
                **quantities,
                **NS_parameters,
            )
        udiff[0] = 0.0
        for i, ui in enumerate(u_components):
            t1 = OasisTimer("Solving tentative velocity " + ui, print_solve_info)
            solver.velocity_tentative_assemble(ui=ui, **F_dict, **quantities)
            pblm.velocity_tentative_hook(ui=ui, **quantities, **NS_parameters)
            solver.velocity_tentative_solve(
                udiff=udiff,
                ui=ui,
                u_sol=u_sol,
                **F_dict,
                **quantities,
                **NS_parameters,
            )
            t1.stop()
        t0 = OasisTimer("Pressure solve", print_solve_info)
        solver.pressure_assemble(**quantities, **F_dict, **NS_parameters)
        pblm.pressure_hook(**quantities)
        solver.pressure_solve(p_sol=p_sol, **quantities, **F_dict)
        t0.stop()

        solver.print_velocity_pressure_info(
            num_iter=num_iter,
            norm=df.norm,
            info_blue=info_blue,
            inner_iter=inner_iter,
            udiff=udiff,
            **quantities,
            **NS_parameters,
        )

    # Update velocity
    t0 = OasisTimer("Velocity update")
    solver.velocity_update(
        u_components=u_components,
        **quantities,
        **F_dict,
        **NS_parameters,
    )
    t0.stop()

    # Solve for scalars
    if len(scalar_components) > 0:
        solver.scalar_assemble(
            scalar_components=scalar_components,
            Schmidt_T=pblm.Schmidt_T,
            Schmidt=pblm.Schmidt,
            **quantities,
            **F_dict,
            **nn_dict,
            **les_dict,
            **NS_parameters,
        )
        for ci in scalar_components:
            t1 = OasisTimer("Solving scalar {}".format(ci), print_solve_info)
            pblm.scalar_hook()  # FIXME: what do we need to pass here?
            solver.scalar_solve(
                scalar_components=scalar_components,
                Schmidt=pblm.Schmidt,
                c_sol=c_sol,
                ci=ci,
                **quantities,
                **F_dict,
                **NS_parameters,
            )
            t1.stop()
    pblm.temporal_hook(**psh_dict, **quantities, **NS_parameters)

    # Save solution if required and check for killoasis file
    stop = io.save_solution(
        newfolder=newfolder,
        tstepfiles=tstepfiles,
        u_components=u_components,
        scalar_components=scalar_components,
        NS_parameters=NS_parameters,
        constrained_domain=pblm.constrained_domain,
        AssignedVectorFunction=ut.AssignedVectorFunction,
        total_timer=total_timer,
        **NS_parameters,
        **quantities,
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

if NS_parameters["restart_folder"] is not None:
    io.merge_visualization_files(newfolder=newfolder)

# Final hook
theend_hook(mesh=mesh, **psh_dict, **quantities, **NS_parameters)

# F_dict.keys() = dict_keys(['A', 'M', 'K', 'Ap', 'divu', 'gradp', 'Ta', 'Tb', 'bb', 'bx', 'u_ab', 'a_conv', 'a_scalar', 'LT', 'KT', 'NT'])
# nn_dict.keys() = dict_keys(['nunn_'])
# les_dict.keys() = dict_keys(['nut_'])
# quantities.keys() = dict_keys(['u', 'u_', 'u_1', 'u_2', 'x_', 'x_1', 'x_2', 'b', 'b_tmp', 'p', 'p_', 'p_1', 'dp_', 'q', 'q_', 'q_1', 'q_2', 'v', 'V', 'Q', 'f', 'fs', 'bcs', 'b0'])
# NS_parameters.keys() = dict_keys(['nu', 'folder', 'velocity_degree', 'pressure_degree', 't', 'tstep', 'T', 'dt', 'AB_projection_pressure', 'solver', 'max_iter', 'max_error', 'iters_on_first_timestep', 'use_krylov_solvers', 'print_intermediate_info', 'print_velocity_pressure_convergence', 'plot_interval', 'checkpoint', 'save_step', 'restart_folder', 'output_timeseries_as_vector', 'killtime', 'les_model', 'Smagorinsky', 'Wale', 'DynamicSmagorinsky', 'KineticEnergySGS', 'nn_model', 'ModifiedCross', 'testing', 'krylov_solvers', 'velocity_update_solver', 'velocity_krylov_solver', 'pressure_krylov_solver', 'scalar_krylov_solver', 'nut_krylov_solver', 'nu_nn_krylov_solver'])
# psh_dict.keys() = dict_keys(['uv'])
