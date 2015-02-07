__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from ..NSfracStep import *

parameters["mesh_partitioner"] = "SCOTCH"

# Set up parameters
NS_parameters.update(
    nu = 15.11E-7,
    T  = .1,
    dt = 1E-5,
    les_model="DynamicLagrangian",
    plot_interval = 20,
    save_step=1,
    print_intermediate_info = 100,
    use_krylov_solvers = True)

NS_parameters["DynamicSmagorinsky"].update(comp_step=1)

from mshr import *

r1 = Rectangle(Point(0, 0), Point(0.02, 0.005))
r2 = Rectangle(Point(0, 0), Point(0.1, 0.01))
r3 = Rectangle(Point(0.08, 0), Point(0.1, 0.005))

domain = r2-r3-r1

mesh = generate_mesh(domain, 1000)

noslip = "on_boundary && std::abs(0.1-x[0]) > DOLFIN_EPS ||\
                std::abs(0.01 - x[1]) < DOLFIN_EPS || \
                std::abs(0.005 - x[1]) < DOLFIN_EPS ||\
                x[1] < DOLFIN_EPS || \
                std::abs(0.02 - x[0]) < DOLFIN_EPS ||\
                std::abs(0.08 - x[0]) < DOLFIN_EPS"
inlet = "on_boundary && x[0] < DOLFIN_EPS"
outlet = "on_boundary && std::abs(0.1-x[0]) < DOLFIN_EPS"

# Specify boundary conditions
def create_bcs(V, Q, **NS_namespace):
    bc0  = DirichletBC(V, 0, noslip)
    bc00 = DirichletBC(V, 1, inlet)
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
    nutfile = File("results/bwfstep/nut.pvd")
    CSGSFile = File("results/bwfstep/CSGS.pvd")
    v_file = File("results/bwfstep/U.pvd")
    set_log_active(False)
    return dict(uv=Function(Vv), nutfile=nutfile, 
            CSGSFile=CSGSFile, v_file=v_file)

def start_timestep_hook(t, **NS_namespace):
    print "t = ", t, "s"
    
def temporal_hook(tstep, save_step, nut_, u_, nutfile, v_file, uv, 
        CSGSFile, Cs, **NS_namespace):
    
    if tstep%save_step == 0:
        nutfile << nut_
        assign(uv.sub(0), u_[0])
        assign(uv.sub(1), u_[1])
        v_file << uv
        CSGSFile << Cs
