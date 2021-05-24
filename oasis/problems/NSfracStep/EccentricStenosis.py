from __future__ import print_function
from oasis import *
from oasis.problems import *
#from mshr import *
#from Probes import *
import numpy as np
import random
import sys
import os
import platform

def problem_parameters(NS_parameters, NS_expressions, commandline_kwargs, **NS_namespace):
    Re = float(commandline_kwargs.get("Re", 700))
    NS_parameters.update(
        nu = 0.0031078341013824886, #mm^2/ms #3.1078341E-6 m^2/s, #0.003372 Pa-s/1085 kg/m^3 this is nu_inf (m^2/s)
        D = 6.35,#0.00635,
        T = 15e3, #ms
        dt=0.1, #ms
        Re=Re,
        #nn_model = 'ModifiedCross',
        mesh_file = 'eccentric_stenosis.xml.gz',
        save_step=10,
        velocity_degree=1,
        krylov_solvers=dict(
            monitor_convergence=False,
            error_on_nonconvergence=True,
            relative_tolerance=1e-10,
            absolute_tolerance=1e-10),
        checkpoint=10,
        print_intermediate_info = 100,
        print_velocity_pressure_convergence=False
        )

    average_inlet_velocity = get_ave_inlet_velocity(NS_parameters['Re'], NS_parameters['nu'], NS_parameters['D'])
    NS_parameters.update(ave_inlet_velocity=average_inlet_velocity)
    inflow_prof = get_inflow_prof(average_inlet_velocity, NS_parameters['D'])
    NS_expressions.update(dict(u_in=inflow_prof))

def mesh(mesh_file, **NS_namespace):
    if not os.path.isfile(mesh_file):
        if platform.system() == "Linux":
            os.system(f"wget -O {mesh_file} https://ndownloader.figshare.com/files/28021383")
        elif platform.system() == "Darwin":
            os.system(f"curl -L https://ndownloader.figshare.com/files/28021383 -o {mesh_file}")
        else:
            raise ImportError("Could not determine platform")
        print(f"Downloaded mesh {mesh_file}")

    mesh = Mesh(mesh_file)
    return mesh

def create_bcs(V, Q, mesh, mesh_file, **NS_namespace):
    if MPI.rank(MPI.comm_world) == 0:
        print("Create bcs")

    boundaries = MeshFunction("size_t", mesh, 2, mesh.domains())
    boundaries.set_values(boundaries.array()+1)

    wallId = 1
    inletId = 2
    outletId = 3

    bc0 = DirichletBC(V, 0, boundaries, wallId)
    bc1 = DirichletBC(V, NS_expressions['u_in'], boundaries, inletId)
    bc2 = DirichletBC(V, 0, boundaries, inletId)
    bc3 = DirichletBC(Q, 0, boundaries, outletId)
    return dict(u0=[bc0, bc1], #0 on the sides, u_in on inlet, zero gradient outlet
                u1=[bc0, bc2], #0 on sides and inlet, zero gradient outlet
                u2=[bc0, bc2], #0 on sides and inlet, zero gradient outlet
                p=[bc3]) #0 outlet

def initialize(V, q_, q_1, q_2, x_1, x_2, bcs, restart_folder, **NS_namespace):
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(u_, V, D, AssignedVectorFunction, **NS_namespace):
    #x = array([10*D,0,0])
    #p = Probe(x, V, 0)
    return dict(uv=AssignedVectorFunction(u_))#, p=p)

def velocity_tentative_hook(**NS_namespace):
    pass

def pressure_hook(**NS_namespace):
    pass

def start_timestep_hook(**NS_namespace):
    pass

def temporal_hook(t, x_, folder, p, V, **NS_namespace):
    pass #p(x_, t=t, newfolder=folder, V=V)

def theend_hook(**NS_namespace):
    pass

def get_ave_inlet_velocity(Re, nu, D,**NS_namespace):
    average_inlet_velocity = Re*nu/D
    return average_inlet_velocity

def get_inflow_prof(average_inlet_velocity, D, **NS_namespace):
    u_inflow = Expression('A*2*(1-((x[1]*x[1])+(x[2]*x[2]))*4/(D*D))',degree=2, A=average_inlet_velocity, D=D)
    return u_inflow
