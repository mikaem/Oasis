__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from Oasis import *

from numpy import ceil
import time

# Create a mesh here
mesh = BoxMesh(-pi, -pi, -pi, pi, pi, pi, 20, 20, 20)

dim = mesh.geometry().dim()
u_components = ['u0', 'u1', 'u2']
sys_comp =  u_components + ['p']

class PeriodicDomain(SubDomain):
    
    def inside(self, x, on_boundary):
        return bool((near(x[0], -pi) or near(x[1], -pi) or near(x[2], -pi)) and 
                (not (near(x[0], pi) or near(x[1], pi) or near(x[2], pi))) and on_boundary)

    def map(self, x, y):
        if near(x[0], pi) and near(x[1], pi) and near(x[2], pi):
            y[0] = x[0] - 2.0*pi
            y[1] = x[1] - 2.0*pi
            y[2] = x[2] - 2.0*pi
        elif near(x[0], pi) and near(x[1], pi):
            y[0] = x[0] - 2.0*pi
            y[1] = x[1] - 2.0*pi
            y[2] = x[2]
        elif near(x[1], pi) and near(x[2], pi):
            y[0] = x[0] 
            y[1] = x[1] - 2.0*pi
            y[2] = x[2] - 2.0*pi
        elif near(x[1], pi):
            y[0] = x[0] 
            y[1] = x[1] - 2.0*pi
            y[2] = x[2]
        elif near(x[0], pi) and near(x[2], pi):
            y[0] = x[0] - 2.0*pi
            y[1] = x[1] 
            y[2] = x[2] - 2.0*pi
        elif near(x[0], pi):
            y[0] = x[0] - 2.0*pi
            y[1] = x[1] 
            y[2] = x[2]            
        else: # near(x[2], pi):
            y[0] = x[0] 
            y[1] = x[1]
            y[2] = x[2] - 2.0*pi

constrained_domain = PeriodicDomain()

# Override some problem specific parameters and put the variables in DC_dict
T = 0.5
#dt = 0.25*T/ceil(T/0.2/mesh.hmin())
dt = 0.01
folder = "taylorgreen3D_results"
newfolder = create_initial_folders(folder, dt)
NS_parameters.update(dict(
    nu = 0.01,
    T = T,
    dt = dt,
    folder = folder,
    newfolder = newfolder,
    sys_comp = sys_comp,
    velocity_degree = 1,
    use_krylov_solvers = True,
    use_lumping_of_mass_matrix = False,
  )
)

if NS_parameters['velocity_degree'] > 1:
    NS_parameters['use_lumping_of_mass_matrix'] = False

# Put all the NS_parameters in the global namespace of Problem
# These parameters are all imported by the Navier Stokes solver
globals().update(NS_parameters)

# Specify body force
f = Constant((0,)*dim)

# Normalize pressure or not? 
#normalize = False

def pre_solve(NS_dict):    
    """Called prior to time loop"""
    globals().update(NS_dict)
    velocity_plotter0 = VTKPlotter(q_['u0'])
    velocity_plotter1 = VTKPlotter(q_['u1'])
    pressure_plotter = VTKPlotter(p_) 
    globals().update(
                   velocity_plotter0=velocity_plotter0,
                   velocity_plotter1=velocity_plotter1,
                   pressure_plotter=pressure_plotter)

# Specify boundary conditions
def create_bcs():
    return dict((ui, []) for ui in sys_comp)

initial_fields = dict(
        u0='sin(x[0])*cos(x[1])*cos(x[2])',
        u1='-cos(x[0])*sin(x[1])*cos(x[2])',
        u2='0',
        p='0')
    
def initialize(NS_dict):
    globals().update(NS_dict)
    for ui in sys_comp:
        vv = project(Expression((initial_fields[ui])), VV[ui])
        q_[ui].vector()[:] = vv.vector()[:]
        if not ui == 'p':
            q_1[ui].vector()[:] = q_[ui].vector()[:]
            q_2[ui].vector()[:] = q_[ui].vector()[:]

# Set up linear solvers
def get_solvers():
    """return three solvers, velocity, pressure and velocity update.
    In case of lumping return None for velocity update"""
    if use_krylov_solvers:
        u_sol = KrylovSolver('bicgstab', 'jacobi')
        u_sol.parameters['error_on_nonconvergence'] = False
        u_sol.parameters['nonzero_initial_guess'] = True
        u_sol.parameters['preconditioner']['reuse'] = False
        u_sol.parameters['monitor_convergence'] = True
        u_sol.parameters['maximum_iterations'] = 100
        u_sol.parameters['relative_tolerance'] = 1e-8
        u_sol.parameters['absolute_tolerance'] = 1e-8
        u_sol.t = 0

        if use_lumping_of_mass_matrix:
            du_sol = None
        else:
            du_sol = KrylovSolver('bicgstab', 'hypre_euclid')
            du_sol.parameters['error_on_nonconvergence'] = False
            du_sol.parameters['nonzero_initial_guess'] = True
            du_sol.parameters['preconditioner']['reuse'] = True
            du_sol.parameters['monitor_convergence'] = True
            du_sol.parameters['maximum_iterations'] = 50
            du_sol.parameters['relative_tolerance'] = 1e-9
            du_sol.parameters['absolute_tolerance'] = 1e-10
            du_sol.t = 0
            
        p_sol = KrylovSolver('gmres', 'hypre_amg')
        p_sol.parameters['error_on_nonconvergence'] = True
        p_sol.parameters['nonzero_initial_guess'] = True
        p_sol.parameters['preconditioner']['reuse'] = True
        p_sol.parameters['monitor_convergence'] = True
        p_sol.parameters['maximum_iterations'] = 100
        p_sol.parameters['relative_tolerance'] = 1e-8
        p_sol.parameters['absolute_tolerance'] = 1e-8
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
    #uv.assign(project(u_, Vv))
    if tstep % 10 == 0:
        pressure_plotter.plot()
        velocity_plotter0.plot()
        velocity_plotter1.plot()

def theend():
    pass
