__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from ..NSfracStep import *

parameters["mesh_partitioner"] = "SCOTCH"

# Set up parameters
NS_parameters.update(
    nu = 5e-4,
    T  = 1,
    dt = .0001,
    plot_interval = 20,
    save_step=1,
    print_intermediate_info = 100,
    use_krylov_solvers = True)

scalar_components = ["temp"]

NS_parameters["DynamicSmagorinsky"].update(Cs_comp_step=1)
NS_parameters["boussinesq"].update(use=True, beta=100, g=0, T_ref=0)

mesh = RectangleMesh(0,0,0.2,1,250,500)

noslip = "on_boundary"
left = "on_boundary && x[0] < DOLFIN_EPS"
right = "on_boundary && std::abs(.2-x[0]) < DOLFIN_EPS"

Schmidt["temp"] = 10

# Specify boundary conditions
def create_bcs(V, Q, **NS_namespace):
    bc0  = DirichletBC(V, 0, noslip)
    bcT1 = DirichletBC(V, .1, left)
    bcT2 = DirichletBC(V, -.1, right)
    return dict(u0 = [bc0],
                u1 = [bc0],
                p = [],
                temp = [bcT1, bcT2])
                
def initialize(x_1, x_2, bcs, **NS_namespace):
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:    
        [bc.apply(x_2[ui]) for bc in bcs[ui]]

def pre_solve_hook(mesh, nut_, velocity_degree, **NS_namespace):
    Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree)
    nutfile = File("results/temp_box/nut.pvd")
    CSGSFile = File("results/temp_box/CSGS.pvd")
    v_file = File("results/temp_box/U.pvd")
    T_file = File("results/temp_box/T.pvd")
    set_log_active(False)
    return dict(uv=Function(Vv), nutfile=nutfile, 
            CSGSFile=CSGSFile, v_file=v_file, T_file=T_file)

def start_timestep_hook(t, **NS_namespace):
    print "t = ", t, "s"
    
def temporal_hook(tstep, save_step, nut_, u_, nutfile, v_file, uv, 
        CSGSFile, T_file, Temp, **NS_namespace):
    if tstep%save_step == 0:
        #nutfile << nut_
        assign(uv.sub(0), u_[0])
        assign(uv.sub(1), u_[1])
        v_file << uv
        #CSGSFile << Cs
        T_file << Temp
