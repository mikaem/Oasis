__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from ..NSfracStep import *
import time

parameters["mesh_partitioner"] = "SCOTCH"

# Set up parameters
NS_parameters.update(
    nu = 2E-7,
    T  = 0.1,
    dt = 5E-7,
    les_model="ScaleDepDynamicLagrangian",
    plot_interval = 20,
    save_step=20,
    print_intermediate_info = 100,
    velocity_degree=2,
    use_krylov_solvers = True)
NS_parameters["DynamicSmagorinsky"].update(JMM_init=1E3)

from mshr import *

r1 = Rectangle(Point(0, 0), Point(0.001, 0.01))
r2 = Rectangle(Point(0.001, 0.0048), Point(0.003, 0.0052))
r3 = Rectangle(Point(0.003, 0.0046), Point(0.014, 0.0054))
t1 = Polygon([Point(0.012, 0.0046), Point(0.014, 0.0040), Point(0.014,0.0046)])
t2 = Polygon([Point(0.012, 0.0054), Point(0.014, 0.0054), Point(0.014,0.0060)])
domain = r2 + r3 + r1

mesh = generate_mesh(domain,1200)

noslip = "on_boundary && std::abs(0.014-x[0]) > DOLFIN_EPS"
inlet = "on_boundary && x[0] < DOLFIN_EPS"
outlet = "on_boundary && std::abs(0.014-x[0]) < DOLFIN_EPS"

# Specify boundary conditions
def create_bcs(V, Q, **NS_namespace):
    bc0  = DirichletBC(V, 0, noslip)
    bc00 = DirichletBC(V, .4, inlet)
    bc01 = DirichletBC(V, 0, inlet)
    bcp = DirichletBC(Q, 0, outlet)
    return dict(u0 = [bc0, bc00],
                u1 = [bc0, bc01],
                u2 = [bc0, bc01],
                p  = [bcp])
                
def initialize(x_1, x_2, bcs, **NS_namespace):
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:    
        [bc.apply(x_2[ui]) for bc in bcs[ui]]

def pre_solve_hook(mesh, nut_, velocity_degree, CG1, **NS_namespace):
    Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree)
    nutfile = File("results/thinchannel/nut.pvd")
    CSGSFile = File("results/thinchannel/Cs.pvd")
    v_file = File("results/thinchannel/U.pvd")
    JLMfile = File("results/thinchannel/JLM.pvd")
    JMMfile = File("results/thinchannel/JMM.pvd")
    set_log_active(False)
    
    return dict(uv=Function(Vv), nutfile=nutfile, JLMfile=JLMfile,
            CSGSFile=CSGSFile, v_file=v_file, JMMfile=JMMfile)

def start_timestep_hook(t, **NS_namespace):
    info_red("t = {}s".format(t))
    
def temporal_hook(tstep, save_step, nut_, u_, nutfile, v_file, uv, 
        CSGSFile, Cs, JLMfile, JMMfile, JLM, JMM, **NS_namespace):

    if tstep%save_step == 0:
        nutfile << nut_
        assign(uv.sub(0), u_[0])
        assign(uv.sub(1), u_[1])
        v_file << uv
        CSGSFile << Cs
        JLMfile << JLM
        JMMfile << JMM
