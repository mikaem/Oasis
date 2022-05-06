from __future__ import print_function
__author__ = "Kei Yamamoto <keiyamamo@math.uio.no>"
__copyright__ = "Copyright (C) 2021 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..NSfracStep import *
from ..StraightPipe import *
from os import getcwd, makedirs
import pickle
import random
from pprint import pprint
from fenicstools import interpolate_nonmatching_mesh_any

"""
This script is written for simulating turbulent flow in a straight pipe.
We assume that the axial directon is x direction in this script. 
"""

def problem_parameters(commandline_kwargs, NS_parameters, NS_expressions, **NS_namespace):
    if "restart_folder" in commandline_kwargs.keys():
        restart_folder = commandline_kwargs["restart_folder"]
        restart_folder = path.join(getcwd(), restart_folder)
        f = open(path.join(path.dirname(path.abspath(__file__)), restart_folder, 'params.dat'), 'r')
        NS_parameters.update(pickle.load(f))
        NS_parameters['restart_folder'] = restart_folder
        globals().update(NS_parameters)
    else:
        Lx = 10 # length of pipe in x direction
        D = 2 # diameter
        NS_parameters.update(Lx=Lx, D=D)

        # Override some problem specific parameters
        T = 2000    # use over 2000 for dt=0.01
        dt = 0.01
        nu = 1/7*1.e-3
        Re_tau = 180
        NS_parameters.update(
            save_solution_after_tstep = 150000,
            save_solution_step = 5,
            checkpoint=1000,
            utau = nu * Re_tau / (D/2),
            save_step=100,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            nu=nu,
            Re_tau=Re_tau,
            T=T,
            dt=dt,
            velocity_degree=1,
            folder="straightpipe_results",
            use_krylov_solvers=True)
    

    NS_expressions.update(dict(constrained_domain=PeriodicDomain(Lx)))
    if MPI.rank(MPI.comm_world)==0:
        print("Running with the following parameters:")
        pprint(NS_parameters)

class PeriodicDomain(SubDomain):
    def __init__(self, Lx):
        self.Lx = Lx
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(near(x[0], 0) and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - self.Lx
        y[1] = x[1]
        y[2] = x[2]

# Specify body force
def body_force(utau, D, **NS_namespace):
    return Constant((4*(utau**2)/D, 0., 0.))

def pre_solve_hook(mesh, restart_folder, velocity_degree, newfolder, MPI, **NS_namespace):
    """Called prior to time loop"""
    if MPI.rank(MPI.comm_world) == 0:
        makedirs(path.join(newfolder, "Outlet"))

    if MPI.rank(MPI.comm_world) == 0:
        makedirs(path.join(newfolder, "Solutions"))

    # create path for saving velocity, pressure, mesh inside Solution folder
    solution_path = path.join(newfolder, "Solutions")
    solution_mesh_path = path.join(solution_path, "mesh.h5")
    solution_velocity_path = path.join(solution_path, "u.h5")
    solution_pressure_path = path.join(solution_path, "p.h5")
    solution_files = {"solution_mesh" : solution_mesh_path, "solution_v" : solution_velocity_path, "solution_p" : solution_pressure_path}
   
    # Save mesh as HDF5 file for post processing
    with HDF5File(MPI.comm_world, solution_mesh_path, "w") as mesh_file:
        mesh_file.write(mesh, "mesh")

    Vv = VectorFunctionSpace(mesh, "CG", velocity_degree)
    U = Function(Vv)

    # create path for saving velocioties at the outlet 
    out_let_path = path.join(newfolder, "Outlet")
    u0_outlet_path = path.join(out_let_path, "u0_outlet.h5")
    u1_outlet_path = path.join(out_let_path, "u1_outlet.h5")
    u2_outlet_path = path.join(out_let_path, "u2_outlet.h5")
    outlet_files = {"u0_outlet": u0_outlet_path, "u1_outlet": u1_outlet_path, "u2_outlet": u2_outlet_path}
   
    Vu = FunctionSpace(outlet_mesh, "CG", velocity_degree)
    
    return dict(outlet_files=outlet_files, solution_files=solution_files, U=U, Vu=Vu)

def create_bcs(V, sys_comp, **NS_namespace):

    info_red("Creating boundary conditions")
    bcs = dict((ui, []) for ui in sys_comp)
    bc_dict = {"inlet":1, "outlet":2, "wall":3}
    bc = [DirichletBC(V, Constant(0), mf, bc_dict["wall"])]
    bcs['u0'] = bc
    bcs['u1'] = bc
    bcs['u2'] = bc
    return bcs

class RandomStreamVector(UserExpression):
    random.seed(2 + MPI.rank(MPI.comm_world))

    def eval(self, values, x):
        values[0] = 0.01 * random.random() 
        values[1] = 0.01 * random.random()
        values[2] = 0.01 * random.random()

    def value_shape(self):
        return (3,)

def initialize(V, q_1, q_2, utau, D, Lx, bcs, restart_folder, nu, **NS_namespace):
    if restart_folder is None:
        #  Initialize using a perturbed flow. Create random streamfunction
         Vv = VectorFunctionSpace(V.mesh(), V.ufl_element().family(),
                                  V.ufl_element().degree())
         psi = interpolate(RandomStreamVector(element=Vv.ufl_element()), Vv)
         u0 = project(curl(psi), Vv, solver_type='cg')
         u0x = project(u0[0], V, bcs=bcs['u0'], solver_type='cg')
         u1x = project(u0[1], V, bcs=bcs['u0'], solver_type='cg')
         u2x = project(u0[2], V, bcs=bcs['u0'], solver_type='cg')
         # Create base flow / paraboric flow
         r = interpolate(Expression("(x[1])*(x[1]) + (x[2])*(x[2])", element=V.ufl_element()), V)
         uu = project((utau*utau*(D*D-r*r)/D/Lx), V, bcs=bcs['u0'], solver_type='cg')
         # initialize vectors at two timesteps
         q_1['u0'].vector()[:] = uu.vector()[:]
         q_1['u0'].vector().axpy(1.0, u0x.vector())
         q_1['u1'].vector()[:] = u1x.vector()[:]
         q_1['u2'].vector()[:] = u2x.vector()[:]
         q_2['u0'].vector()[:] = q_1['u0'].vector()[:]
         q_2['u1'].vector()[:] = q_1['u1'].vector()[:]
         q_2['u2'].vector()[:] = q_1['u2'].vector()[:]

def temporal_hook(u_, p_, Vu, tstep, outlet_files, save_solution_after_tstep, save_solution_step, U, solution_files, u_mean0, u_mean1, u_mean2,
                 **NS_namespace): 

    if tstep >= save_solution_after_tstep:
         # save velocity at outlet  / first we interpolate u_ into Vu which is defined on the bmesh (outlet)
        u0_outlet = interpolate_nonmatching_mesh_any(u_[0], Vu)
        u1_outlet = interpolate_nonmatching_mesh_any(u_[1], Vu)
        u2_outlet = interpolate_nonmatching_mesh_any(u_[2], Vu) 

        if MPI.rank(MPI.comm_world) == 0:
            print("u1 outlet = ", u1_outlet.vector().get_local()[100])
        
        # Save velocity at outlet
        u0_path = outlet_files["u0_outlet"]
        u1_path = outlet_files["u1_outlet"]
        u2_path = outlet_files["u2_outlet"]

        # Save velocity at the outlet of the mesh. They will be used as inlet profile of coilped pipe
        file_mode = "w" if tstep == save_solution_after_tstep else "a"

        u0_hdf5_file = HDF5File(MPI.comm_world, u0_path, file_mode=file_mode)
        u0_hdf5_file.write(u0_outlet,"/velocity", tstep)
        u0_hdf5_file.close()
        u1_hdf5_file = HDF5File(MPI.comm_world, u1_path, file_mode=file_mode)
        u1_hdf5_file.write(u1_outlet,"/velocity", tstep)
        u1_hdf5_file.close()
        u2_hdf5_file = HDF5File(MPI.comm_world, u2_path, file_mode=file_mode)
        u2_hdf5_file.write(u2_outlet,"/velocity", tstep)
        u2_hdf5_file.close()

        
     # save velocity and pressure in the entire domain
    if tstep % save_solution_step == 0 and tstep >= save_solution_after_tstep:
        # Assign velocity components to vector solution
        assign(U.sub(0), u_[0])
        assign(U.sub(1), u_[1])
        assign(U.sub(2), u_[2])
        
        # Save velocity
        viz_u = HDF5File(MPI.comm_world, solution_files["solution_v"], file_mode=file_mode)
        viz_u.write(U, "/velocity", tstep)
        viz_u.close()

        # Save pressure
        viz_p = HDF5File(MPI.comm_world, solution_files["solution_p"], file_mode=file_mode)
        viz_p.write(p_, "/pressure", tstep)
        viz_p.close()
    
        # Accumulate velocity to compute mean at the end 
        u_mean0.vector().axpy(1, u_[0].vector())
        u_mean1.vector().axpy(1, u_[1].vector())
        u_mean2.vector().axpy(1, u_[2].vector())

def theend_hook(newfolder, u_mean, u_mean0, u_mean1, u_mean2, T, dt, save_solution_after_tstep, save_solution_step, **NS_namespace):
    path_to_u_mean = path.join(newfolder, "Solutions", "u_mean.h5")
    NumTStepForAverage = (T/dt - save_solution_after_tstep) / save_solution_step + 1
    u_mean0.vector()[:] = u_mean0.vector()[:] /  NumTStepForAverage 
    u_mean1.vector()[:] = u_mean1.vector()[:] /  NumTStepForAverage 
    u_mean2.vector()[:] = u_mean2.vector()[:] /  NumTStepForAverage 

    assign(u_mean.sub(0), u_mean0)
    assign(u_mean.sub(1), u_mean1)
    assign(u_mean.sub(2), u_mean2)

    # check flux at the outlet / Volume flow rate  is the same as mean bulk velocity
    normal = FacetNormal(mesh)
    ds_outlet =  Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)
    flux = assemble(dot(u_mean, normal)*ds_outlet)
    outlet_area = assemble(1*ds_outlet)
    if MPI.rank(MPI.comm_world) == 0:
        print("Flux = ", flux)
        print("Volume flow rate  = ", flux/outlet_area)
    # Save u_mean
    with HDF5File(MPI.comm_world, path_to_u_mean, "w") as u_mean_file:
        u_mean_file.write(u_mean, "u_mean")
