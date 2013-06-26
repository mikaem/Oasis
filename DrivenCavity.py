__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from Oasis import *

from cbc.cfd.tools.Streamfunctions import StreamFunction
from numpy import ceil, cos, pi
import time

# Create a mesh here
mesh = UnitSquareMesh(91, 91)
x = mesh.coordinates()
x[:, :] = (x[:, :] - 0.5)*2
x[:, :] = 0.5*(cos(pi*(x[:, :]-1.) / 2.) + 1.)

dim = mesh.geometry().dim()
u_components = ['u0', 'u1']
sys_comp =  u_components + ['p']

# Override some problem specific parameters and put the variables in DC_dict
T = 2.5
#dt = 5*T/ceil(T/0.2/mesh.hmin())
dt = 0.01
folder = "drivencavity_results"
newfolder = create_initial_folders(folder, dt)
NS_parameters.update(dict(
    nu = 0.001,
    T = T,
    dt = dt,
    folder = folder,
    newfolder = newfolder,
    velocity_degree = 1,
    use_lumping_of_mass_matrix = False,
    use_krylov_solvers = False,
    sys_comp = sys_comp
  )
)
if NS_parameters['velocity_degree'] > 1:
    NS_parameters['use_lumping_of_mass_matrix'] = False

# Put all the NS_parameters in the global namespace of Problem
# These parameters are all imported by the Navier Stokes solver
globals().update(NS_parameters)

constrained_domain = None

# Specify body force
f = Constant((0,)*dim)

# Normalize pressure or not? 
#normalize = False

def pre_solve(NS_dict):    
    """Called prior to time loop"""
    globals().update(NS_dict)
    uv = Function(Vv) 
    velocity_plotter = VTKPlotter(uv)
    pressure_plotter = VTKPlotter(p_) 
    globals().update(uv=uv, 
                   velocity_plotter=velocity_plotter,
                   pressure_plotter=pressure_plotter)

# Specify boundary conditions
def create_bcs():
    bcs = dict((ui, []) for ui in sys_comp)
    
    # Driven cavity example:
    def lid(x, on_boundary):
        return (on_boundary and near(x[1], 1.0))
        
    def stationary_walls(x, on_boundary):
        return on_boundary and (near(x[0], 0.) or near(x[0], 1.) or near(x[1], 0.))

    bc0  = DirichletBC(V, 0., stationary_walls)
    bc00 = DirichletBC(V, 1., lid)
    bc01 = DirichletBC(V, 0., lid)
    bcs['u0'] = [bc00, bc0]
    bcs['u1'] = [bc01, bc0]

    return bcs
    
def initialize(NS_dict):
    globals().update(NS_dict)
    q_['u0'].vector()[:] = 1e-12
    
# Set up linear solvers
def get_solvers():
    """return three solvers, velocity, pressure and velocity update.
    In case of lumping return None for velocity update"""
    if use_krylov_solvers:
        u_sol = KrylovSolver('bicgstab', 'jacobi')
        u_sol.parameters.update(krylov_solvers)
        u_sol.parameters['preconditioner']['reuse'] = False
        u_sol.t = 0

        if use_lumping_of_mass_matrix:
            du_sol = None
        else:
            du_sol = KrylovSolver('bicgstab', 'hypre_euclid')
            du_sol.parameters.update(krylov_solvers)
            du_sol.parameters['preconditioner']['reuse'] = True
            du_sol.t = 0
            
        p_sol = KrylovSolver('gmres', 'hypre_amg')
        p_sol.parameters['preconditioner']['reuse'] = True
        p_sol.parameters.update(krylov_solvers)
        p_sol.t = 0
    else:
        u_sol = LUSolver()
        u_sol.t = 0

        if use_lumping_of_mass_matrix:
            du_sol = None
        else:
            du_sol = LUSolver()
            du_sol.parameters['reuse_factorization'] = True
            du_sol.t = 0

        p_sol = LUSolver()
        p_sol.parameters['reuse_factorization'] = True
        p_sol.t = 0
        
    return u_sol, p_sol, du_sol

def pre_pressure_solve():
    pass

def pre_velocity_tentative_solve(ui):
    if use_krylov_solvers:
        if ui == "u0":
            u_sol.parameters['preconditioner']['reuse'] = False
        else:
            u_sol.parameters['preconditioner']['reuse'] = True

def update_end_of_timestep(tstep):
    if tstep % 10 == 0:
        uv.assign(project(u_, Vv))
        pressure_plotter.plot()
        velocity_plotter.plot()

def theend():
    psi = StreamFunction(u_, [], use_strong_bc=True)
    plot(psi)
    interactive()
