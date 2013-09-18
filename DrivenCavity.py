__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from Oasis import *
from numpy import ceil, cos, pi, arctan

# Create a mesh
Nx = 51
Ny = 51
mesh = UnitSquareMesh(Nx, Ny)
x = mesh.coordinates()
x[:, :] = (x[:, :] - 0.5)*2
x[:, :] = 0.5*(cos(pi*(x[:, :]-1.) / 2.) + 1.)
#x[:, :] = ( arctan(pi*x[:, :])/arctan(pi) +1. ) / 2.
#set_log_level(10)

T = 2.5
#dt = 0.25*T/ceil(T/0.2/mesh.hmin())
dt = 0.001
# Override some problem specific parameters
recursive_update(NS_parameters,
   dict(nu = 0.001,
        T = T,
        dt = dt,
        folder = "drivencavity_results",
        plot_interval = 100,
        save_step = 10000,
        checkpoint = 10000,
        velocity_degree = 1,
        max_iter = 1,
        use_lumping_of_mass_matrix = True,
        use_krylov_solvers = True,
        krylov_solvers = dict(monitor_convergence=True,
                              relative_tolerance = 1e-7))
)

def pre_solve_hook(Vv, **NS_namespace):    
    # Declare a Function used for plotting in temporal_hook
    uv = Function(Vv)  
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
    #bcs['u0'] = [bc0, bc00]
    #bcs['u1'] = [bc0, bc01]
    return bcs

def start_timestep_hook(t, **NS_namespace):
    pass
    #u_top.assign(cos(t))
    
def initialize(x_, x_1, x_2, bcs, **NS_namespace):
    for ui in x_2:
        [bc.apply(x_[ui]) for bc in bcs[ui]]
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
        [bc.apply(x_2[ui]) for bc in bcs[ui]]
    #q_['u0'].vector()[:] = 1e-12 # To help Krylov solver on first timestep
    
def temporal_hook(tstep, u_, Vv, uv, p_, plot_interval, **NS_namespace):
    if tstep % plot_interval == 0:
        uv.assign(project(u_, Vv))
        plot(uv, title='Velocity')
        plot(p_, title='Pressure')

def theend(u_, p_, uv, Vv, **NS_namespace):
    uv.assign(project(u_, Vv))
    plot(uv, title='Velocity')
    plot(p_, title='Pressure')

    try:
        from cbc.cfd.tools.Streamfunctions import StreamFunction
        psi = StreamFunction(u_, [], use_strong_bc=True)
        plot(psi, title='Streamfunction')
        interactive()
    except:
        pass
