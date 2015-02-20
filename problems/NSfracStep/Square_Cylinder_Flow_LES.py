__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from ..NSfracStep import *

parameters["mesh_partitioner"] = "SCOTCH"

# Set up parameters
NS_parameters.update(
    nu = 5.35e-7,
    T  = .01,
    dt = 5e-5,
    plot_interval = 20,
    les_model="Smagorinsky",
    save_step=1,
    print_intermediate_info = 100,
    use_krylov_solvers = True)

mesh = Mesh("mesh/square.xml")

noslip = "on_boundary && x[0] > DOLFIN_EPS && std::abs(0.1-x[0]) > \
        DOLFIN_EPS"
inlet = "on_boundary && x[0] < DOLFIN_EPS"
outlet = "on_boundary && std::abs(0.1-x[0]) < DOLFIN_EPS"

# Specify boundary conditions
def create_bcs(V, Q, **NS_namespace):
    bc0  = DirichletBC(V, 0, noslip)
    bc00 = DirichletBC(V, 0.535, inlet)
    bc01 = DirichletBC(V, 0, inlet)
    bcp = DirichletBC(Q, 0, outlet)
    return dict(u0 = [bc0, bc00],
                u1 = [bc0, bc01],
                p  = [bcp])
                
def initialize(x_1, x_2, bcs, **NS_namespace):
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:    
        [bc.apply(x_2[ui]) for bc in bcs[ui]]

def pre_solve_hook(mesh, nut_, velocity_degree, **NS_namespace):
    Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree)
    nutfile = File("results/square_flow/nut.pvd")
    CSGSFile = File("results/square_flow/CSGS.pvd")
    v_file = File("results/square_flow/U.pvd")
    set_log_active(False)
    return dict(uv=Function(Vv), nutfile=nutfile, 
            CSGSFile=CSGSFile, v_file=v_file)

def start_timestep_hook(t, **NS_namespace):
    print "t = ", t, "s"
    
def temporal_hook(tstep, save_step, nut_, u_, nutfile, v_file, uv, 
        CSGSFile, **NS_namespace):
    
    if tstep%save_step == 0:
        nutfile << nut_
        assign(uv.sub(0), u_[0])
        assign(uv.sub(1), u_[1])
        v_file << uv
        #CSGSFile << Cs
