__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-04"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

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
    post_import_problem,
)

"""
This module implements a generic steady state coupled solver for the
incompressible Navier-Stokes equations. Several mixed function spaces
are supported. The spaces are chosen at run-time through the parameter
elements, that may be

    "TaylorHood" Pq continuous Lagrange elements for velocity and Pq-1 for pressure
    "CR"         Crouzeix-Raviart for velocity - discontinuous Galerkin (DG0) for pressure
    "MINI"       P1 velocity with bubble - P1 for pressure

Each new problem needs to implement a new problem module to be placed in
the problems/NSCoupled folder. From the problems module one needs to import
a mesh and a control dictionary called NS_parameters. See
problems/NSCoupled/__init__.py for all possible parameters.

"""

commandline_kwargs = parse_command_line()

# Find the problem module
default_problem = "Cylinder"
problemname = commandline_kwargs.get("problem", default_problem)

# Import the problem module
print("Importing problem module " + problemname)
if problemname == "Cylinder":
    import oasis.problems.NSCoupled.Cylinder as pblm
if problemname == "DrivenCavity":
    import oasis.problems.NSCoupled.DrivenCavity as pblm
elif problemname == "Nozzle2D":
    import oasis.problems.NSCoupled.Nozzle2D as pblm
elif problemname == "Skewed2D":
    import oasis.problems.NSCoupled.Skewed2D as pblm
elif problemname == "SkewedFlow":
    import oasis.problems.NSCoupled.SkewedFlow as pblm

NS_namespace = pblm.get_problem_parameters()

# vars().update(**vars(pblm))

# Update problem spesific parameters
# pblm.problem_parameters(**vars())

# Update current namespace with NS_parameters and commandline_kwargs ++
# vars().update(pblm.post_import_problem(**vars()))

NS_namespace, problem_parameters = post_import_problem(
    NS_namespace, {}, pblm.mesh, commandline_kwargs
)
scalar_components = problem_parameters["scalar_components"]
mesh = NS_namespace["mesh"]

# Import chosen functionality from solvers
solvername = problem_parameters["solver"]
if solvername == "cylindrical":
    import oasis.solvers.NSCoupled.cylindrical as solver
elif solvername == "default":
    import oasis.solvers.NSCoupled.default as solver
elif solvername == "naive":
    import oasis.solvers.NSCoupled.naive as solver

# Create lists of components solved for
u_components = ["u"]
sys_comp = ["up"] + scalar_components

# Get the chosen mixed elment
element = commandline_kwargs.get("element", "TaylorHood")
degree = solver.elements[element]["degree"]
family = solver.elements[element]["family"]
bubble = solver.elements[element]["bubble"]

# TaylorHood may overload degree of elements
if element == "TaylorHood":
    degree["u"] = commandline_kwargs.get("velocity_degree", degree["u"])
    degree["p"] = commandline_kwargs.get("pressure_degree", degree["p"])
    # Should assert that degree['p'] = degree['u']-1 ??

# Declare Elements
V = df.VectorElement(family["u"], mesh.ufl_cell(), degree["u"])
Q = df.FiniteElement(family["p"], mesh.ufl_cell(), degree["p"])
NS_namespace["V"] = V
NS_namespace["Q"] = Q


# Create Mixed space
# MINI element has bubble, add to V
if bubble:
    B = df.VectorElement("Bubble", mesh.ufl_cell(), mesh.geometry().dim() + 1)
    VQ = df.FunctionSpace(
        mesh, df.MixedElement(V + B, Q), constrained_domain=pblm.constrained_domain
    )

else:
    VQ = df.FunctionSpace(mesh, V * Q, constrained_domain=pblm.constrained_domain)
NS_namespace["VQ"] = VQ
# Create trial and test functions
NS_namespace["up"] = up = df.TrialFunction(VQ)
NS_namespace["u"], NS_namespace["p"] = u, p = df.split(up)
NS_namespace["v"], NS_namespace["q"] = v, q = df.TestFunctions(VQ)

# For scalars use CG space
CG = df.FunctionSpace(mesh, "CG", 1, constrained_domain=pblm.constrained_domain)
NS_namespace["CG"] = CG
NS_namespace["c"] = c = df.TrialFunction(CG)
NS_namespace["ct"] = ct = df.TestFunction(CG)

VV = dict(up=VQ)
VV.update(dict((ui, CG) for ui in scalar_components))

# Create dictionaries for the solutions at two timesteps
q_ = dict((ui, df.Function(VV[ui], name=ui)) for ui in sys_comp)
q_1 = dict((ui, df.Function(VV[ui], name=ui + "_1")) for ui in sys_comp)
NS_namespace["q_"], NS_namespace["q_1"] = q_, q_1

# Short forms
NS_namespace["up_"] = up_ = q_["up"]  # Solution at next iteration
NS_namespace["up_1"] = up_1 = q_1["up"]  # Solution at previous iteration
NS_namespace["u_"], NS_namespace["p_"] = u_, p_ = df.split(up_)
NS_namespace["u_1"], NS_namespace["p_1"] = u_1, p_1 = df.split(up_1)

# Create short forms for accessing the solution vectors
x_ = dict((ui, q_[ui].vector()) for ui in sys_comp)  # Solution vectors
x_1 = dict(
    (ui, q_1[ui].vector()) for ui in sys_comp
)  # Solution vectors previous iteration
NS_namespace["x_"], NS_namespace["x_1"] = x_, x_1

# Create vectors to hold rhs of equations
b = dict((ui, df.Vector(x_[ui])) for ui in sys_comp)
NS_namespace["b"] = b

# Short form scalars
for ci in scalar_components:
    print("{}_   = q_ ['{}']".format(ci, ci))
    print("{}_1   = q_1 ['{}']".format(ci, ci))
    exec("{}_   = q_ ['{}']".format(ci, ci))
    exec("{}_1  = q_1['{}']".format(ci, ci))

# Boundary conditions
bcs = pblm.create_bcs(element=element, **problem_parameters, **NS_namespace)
NS_namespace["bcs"] = bcs

# Initialize solution
pblm.initialize(**problem_parameters, **NS_namespace)

#  Fetch linear algebra solvers
up_sol, c_sol = solver.get_solvers(**problem_parameters, **NS_namespace)
NS_namespace["up_sol"], NS_namespace["c_sol"] = up_sol, c_sol

# Get constant body forces
f = pblm.body_force(**problem_parameters, **NS_namespace)

# Get scalar sources
fs = pblm.scalar_source(**vars())
NS_namespace["f"], NS_namespace["fs"] = f, fs

# Preassemble and allocate
F_dict = solver.setup(**problem_parameters, **NS_namespace)
NS_namespace.update(F_dict)
Fs = F_dict["Fs"]

# Anything problem specific
psh_dict = pblm.pre_solve_hook(**problem_parameters, **NS_namespace)
NS_namespace.update(psh_dict)

max_iter = problem_parameters["max_iter"]
max_error = problem_parameters["max_error"]


def iterate(iters=max_iter):
    # Newton iterations for steady flow
    iter = 0
    error = 1

    while iter < iters and error > max_error:
        pblm.start_iter_hook(**problem_parameters, **NS_namespace)
        solver.NS_assemble(**problem_parameters, **NS_namespace)
        pblm.NS_hook(**problem_parameters, **NS_namespace)
        solver.NS_solve(**problem_parameters, **NS_namespace)
        pblm.end_iter_hook(**problem_parameters, **NS_namespace)

        # Update to next iteration
        for ui in sys_comp:
            x_1[ui].zero()
            x_1[ui].axpy(1.0, x_[ui])

        error = b["up"].norm("l2")
        solver.print_velocity_pressure_info(
            iter=iter, error=error, **problem_parameters, **NS_namespace
        )

        iter += 1


def iterate_scalar(iters=max_iter, errors=max_error):
    # Newton iterations for scalars
    if len(scalar_components) > 0:
        err = {ci: 1 for ci in scalar_components}
        for ci in scalar_components:
            globals().update(ci=ci)
            citer = 0
            while citer < iters and err[ci] > errors:
                solver.scalar_assemble(**problem_parameters, **NS_namespace)
                pblm.scalar_hook(**problem_parameters, **NS_namespace)
                solver.scalar_solve(**problem_parameters, **NS_namespace)
                err[ci] = b[ci].norm("l2")
                if df.MPI.rank(df.MPI.comm_world) == 0:
                    print("Iter {}, Error {} = {}".format(citer, ci, err[ci]))
                citer += 1


timer = OasisTimer("Start Newton iterations flow", True)
# Assemble rhs once, before entering iterations (velocity components)
b["up"] = df.assemble(Fs["up"], tensor=b["up"])
for bc in bcs["up"]:
    bc.apply(b["up"], x_["up"])

iterate(max_iter)
timer.stop()

# Assuming there is no feedback to the flow solver from the scalar field,
# we solve the scalar only after converging the flow
if len(scalar_components) > 0:
    scalar_timer = OasisTimer("Start Newton iterations scalars", True)
    # Assemble rhs once, before entering iterations (velocity components)
    for scalar in scalar_components:
        b[scalar] = df.assemble(Fs[scalar], tensor=b[scalar])
        for bc in bcs[scalar]:
            bc.apply(b[scalar], x_[scalar])

    iterate_scalar()
    scalar_timer.stop()

solver.list_timings(df.TimingClear.keep, [df.TimingType.wall])
info_red("Total computing time = {0:f}".format(timer.elapsed()[0]))
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

# Final hook
pblm.theend_hook(**vars())
pblm.theend_hook(**vars())
