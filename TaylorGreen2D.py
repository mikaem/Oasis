__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from Oasis import *

from numpy import ceil
import time

# Create a mesh here
mesh = UnitSquareMesh(50, 50)
scale = 2*(mesh.coordinates() - 0.5)*pi
mesh.coordinates()[:, :] = scale

dim = mesh.geometry().dim()
u_components = ['u0', 'u1']
sys_comp =  u_components + ['p']

class PeriodicDomain(SubDomain):
    
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], -pi) or near(x[1], -pi)) and 
                (not ((near(x[0], -pi) and near(x[1], pi)) or 
                        (near(x[0], pi) and near(x[1], -pi)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], pi) and near(x[1], pi):
            y[0] = x[0] - 2.0*pi
            y[1] = x[1] - 2.0*pi
        elif near(x[0], pi):
            y[0] = x[0] - 2.0*pi
            y[1] = x[1]
        else: #near(x[1], pi)
            y[0] = x[0]
            y[1] = x[1] - 2.0*pi

constrained_domain = PeriodicDomain()

# Override some problem specific parameters and put the variables in DC_dict
T = 10.
#dt = 0.25*T/ceil(T/0.2/mesh.hmin())
dt = 0.1
folder = "taylorgreen2D_results"
newfolder = create_initial_folders(folder, dt)
NS_parameters.update(dict(
    nu = 0.01,
    T = T,
    dt = dt,
    folder = folder,
    max_iter = 5,
    newfolder = newfolder,
    sys_comp = sys_comp,
    use_krylov_solvers = True,
    use_lumping_of_mass_matrix = False,
    monitor_convergence = False,
    krylov_report = False    
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
        u0='2./sqrt(3.)*sin(2.*pi/3.)*sin(x[0])*cos(x[1])',
        u1='2./sqrt(3.)*sin(-2.*pi/3.)*cos(x[0])*sin(x[1])', 
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
    #uv.assign(project(u_, Vv))
    pressure_plotter.plot()
    velocity_plotter0.plot()
    velocity_plotter1.plot()

def theend():
    pass
