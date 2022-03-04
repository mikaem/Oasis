#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:45:42 2022

@author: florianma
"""
import dolfin as df
import numpy as np
from oasis.common import parse_command_line
from oasis.problems import (
    info_green,
    info_red,
    OasisTimer,
    initial_memory_use,
    oasis_memory,
)
from oasis.problems.NSfracStep.Cylinder import Cylinder as ProblemDomain
import oasis.solvers.NSfracStep.IPCS_ABCN as solver

# TODO: implement ScalarSolver
commandline_kwargs = parse_command_line()
default_problem = "Cylinder"
problemname = commandline_kwargs.get("problem", default_problem)
# Import the problem module
print("Importing problem module " + problemname)
# TODO: import the right ProblemDomain based on problemname:
# solver =  NSfracStep.get_domain(problemname)

commandline_kwargs["dt"] = 0.001
my_domain = ProblemDomain(case=2)
my_domain.set_parameters_from_commandline(commandline_kwargs)
my_domain.mesh_from_file(mesh_name="../mesh.xdmf", facet_name="../mf.xdmf")
df.plot(my_domain.mesh)
my_domain.recommend_dt()
# Create lists of components solved for
my_domain.initialize_problem_components()
# Declare FunctionSpaces and arguments
my_domain.dolfin_variable_declaration()
# Boundary conditions
my_domain.create_bcs()
# TODO: Read in previous solution if restarting
# TODO: LES setup
# TODO: Non-Newtonian setup.
# Initialize solution
my_domain.apply_bcs()
solvername = my_domain.solver
# TODO: import the right solver based on solvername:
# solver =  NSfracStep.get_solver(solvername)

cond = my_domain.krylov_solvers["monitor_convergence"]
psi = my_domain.use_krylov_solvers and cond  # = print_solve_info

tx = OasisTimer("Timestep timer")
tx.start()
total_timer = OasisTimer("Start simulations", True)

it0 = my_domain.iters_on_first_timestep
max_iter = my_domain.max_iter

fit = solver.FirstInner(my_domain)
tvs = solver.TentativeVelocityStep(my_domain)
ps = solver.PressureStep(my_domain)

stop = False
t = 0.0
tstep = 0
while t < (my_domain.T - tstep * df.DOLFIN_EPS) and not stop:
    t += my_domain.dt
    tstep += 1

    inner_iter = 0
    udiff = np.array([1e8])  # Norm of velocity change over last inner iter
    num_iter = max(it0, max_iter) if tstep <= 10 else max_iter
    my_domain.start_timestep_hook()  # FIXME: what do we need to pass here?

    while udiff[0] > my_domain.max_error and inner_iter < num_iter:
        inner_iter += 1

        t0 = OasisTimer("Tentative velocity")
        if inner_iter == 1:
            # lesmodel.les_update(**NS_namespace)
            # nnmodel.nn_update(**NS_namespace)
            fit.assemble_first_inner_iter()
            tvs.A = fit.A
        udiff[0] = 0.0
        for i, ui in enumerate(my_domain.u_components):
            t1 = OasisTimer("Solving tentative velocity " + ui, psi)
            tvs.assemble(ui=ui)
            my_domain.velocity_tentative_hook(ui=ui)
            tvs.solve(ui=ui, udiff=udiff)
            t1.stop()
        t0.stop()
        t2 = OasisTimer("Pressure solve", psi)
        ps.assemble()
        my_domain.pressure_hook()
        ps.solve()
        t2.stop()

        my_domain.print_velocity_pressure_info(num_iter, inner_iter, udiff)

    # Update velocity
    t3 = OasisTimer("Velocity update")
    for i, ui in enumerate(my_domain.u_components):
        tvs.velocity_update(ui=ui)
    t3.stop()

    # TODO: Solve for scalars
    # if len(scalar_components) > 0:
    #     solver.scalar_assemble()
    #     for ci in scalar_components:
    #         t1 = OasisTimer("Solving scalar {}".format(ci), print_solve_info)
    #         pblm.scalar_hook()
    #         solver.scalar_solve()
    #         t1.stop()
    my_domain.temporal_hook(t=t, tstep=tstep)

    # TODO: Save solution if required and check for killoasis file
    # stop = io.save_solution()
    my_domain.advance()

    # Print some information
    if tstep % my_domain.print_intermediate_info == 0:
        toc = tx.stop()
        my_domain.show_info(t, tstep, toc)
        df.list_timings(df.TimingClear.clear, [df.TimingType.wall])
        tx.start()

    # AB projection for pressure on next timestep
    if (
        my_domain.AB_projection_pressure
        and t < (my_domain.T - tstep * df.DOLFIN_EPS)
        and not stop
    ):
        my_domain.q_["p"].vector().axpy(0.5, my_domain.dp_.vector())

total_timer.stop()
df.list_timings(df.TimingClear.keep, [df.TimingType.wall])
info_red("Total computing time = {0:f}".format(total_timer.elapsed()[0]))
oasis_memory("Final memory use ")
# total_initial_dolfin_memory
m = df.MPI.sum(df.MPI.comm_world, initial_memory_use)
info_red("Memory use for importing dolfin = {} MB (RSS)".format(m))
info_red("Total memory use of solver = " + str(oasis_memory.memory - m) + " MB (RSS)")

# Final hook
my_domain.theend_hook()
