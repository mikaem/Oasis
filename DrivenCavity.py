__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from Oasis import *
from numpy import ceil, cos, pi, arctan

# Create a mesh here
mesh = UnitSquareMesh(51, 51)
x = mesh.coordinates()
x[:, :] = (x[:, :] - 0.5)*2
x[:, :] = 0.5*(cos(pi*(x[:, :]-1.) / 2.) + 1.)
#x[:, :] = ( arctan(0.025*pi*x[:, :])/arctan(0.025*pi) +1. ) / 2.
del x

# Override some problem specific parameters
T = 0.5
dt = 0.01
folder = "drivencavity_results"
newfolder = create_initial_folders(folder, dt)
recursive_update(NS_parameters,
   dict(nu = 0.001,
        T = T,
        dt = dt,
        folder = folder,
        plot_interval = 10,
        save_step = 1000,
        checkpoint = 1000,
        newfolder = newfolder,
        velocity_degree = 1,
        use_lumping_of_mass_matrix = True,
        use_krylov_solvers = True,
        krylov_solvers = dict(monitor_convergence = True))
)

def pre_solve_hook(Vv, p_, **NS_namespace):    
    uv = Function(Vv)  # For plotting in temporal_hook
    return dict(uv=uv)

def lid(x, on_boundary):
    return (on_boundary and near(x[1], 1.0))
    
def stationary_walls(x, on_boundary):
    return on_boundary and (near(x[0], 0.) or near(x[0], 1.) or near(x[1], 0.))

# Specify boundary conditions
u_top = Constant(1.0)
def create_bcs(V, sys_comp, **NS_namespace):
    bcs = dict((ui, []) for ui in sys_comp)    
    bc0  = DirichletBC(V, 0., stationary_walls)
    bc00 = DirichletBC(V, u_top, lid)
    bc01 = DirichletBC(V, 0., lid)
    bcs['u0'] = [bc00, bc0]
    bcs['u1'] = [bc01, bc0]
    return bcs

def start_timestep_hook(t, **NS_namespace):
    pass
    #u_top.assign(cos(t))
    
def initialize(q_, **NS_namespace):
    q_['u0'].vector()[:] = 1e-12

def temporal_hook(tstep, u_, Vv, uv, p_, plot_interval, **NS_namespace):
    if tstep % plot_interval == 0:
        uv.assign(project(u_, Vv))
        plot(uv)
        plot(p_)

def theend(u_, **NS_namespace):
    try:
        from cbc.cfd.tools.Streamfunctions import StreamFunction
        psi = StreamFunction(u_, [], use_strong_bc=True)
        plot(psi)
        interactive()
    except:
        pass
