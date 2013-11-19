__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
from commands import getoutput
from os import getpid, path, makedirs, getcwd, listdir, remove, system
import cPickle
import inspect
from collections import defaultdict
from numpy import array, maximum, zeros

parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"
#parameters["form_compiler"]["cache_dir"] = "/home/mikael/MySoftware/Oasis/instant"
parameters["mesh_partitioner"] = "ParMETIS"
parameters["form_compiler"].add("no_ferari", True)

# Default parameters
NS_parameters = dict(
  # Physical constants and solver parameters
  nu = 0.01,             # Kinematic viscosity
  t = 0,                 # Time
  tstep = 0,             # Timestep
  T = 1.0,               # End time
  dt = 0.01,             # Time interval on each timestep
  
  # Some discretization options
  AB_projection_pressure = False,  # Use Adams Bashforth projection as first estimate for pressure on new timestep
  velocity_degree = 2,
  pressure_degree = 1,  
  convection = "ABCN",  # "ABCN", "ABE" or "Naive"
  
  # Parameters used to tweek solver  
  max_iter = 1,          # Number of inner pressure velocity iterations on timestep
  max_error = 1e-6,      # Tolerance for inner iterations (pressure velocity iterations)
  iters_on_first_timestep = 2,  # Number of iterations on first timestep
  use_krylov_solvers = False,  # Otherwise use LU-solver
  low_memory_version = False,  # Use assembler and not preassembled matrices
  print_intermediate_info = 10,
  print_velocity_pressure_convergence = True,
  velocity_update_type = "default",
  
  # Parameters used to tweek output  
  plot_interval = 10,    
  checkpoint = 10,       # Overwrite solution in Checkpoint folder each checkpoint tstep
  save_step = 10,        # Store solution in new folder each save_step tstep
  folder = 'results',    # Relative folder for storing results 
  restart_folder = None, # If restarting solution, set the folder holding the solution to start from here
  output_timeseries_as_vector = True, # Store velocity as vector in Timeseries 
  
  # Solver parameters that will be transferred to dolfins parameters['krylov_solver']
  krylov_solvers = dict(
    monitor_convergence = False,
    report = False,
    error_on_nonconvergence = False,
    nonzero_initial_guess = True,
    maximum_iterations = 100,
    relative_tolerance = 1e-8,
    absolute_tolerance = 1e-8)
)

constrained_domain = None

# To solve for scalars provide a list like ['scalar1', 'scalar2']
scalar_components = []

# With diffusivities given as a Schmidt number defined by:
#   Schmidt = nu / D (= momentum diffusivity / mass diffusivity)
Schmidt = defaultdict(lambda: 1.)

# The following helper functions are available in dolfin
# They are redefined here for printing only on process 0. 
RED   = "\033[1;37;31m%s\033[0m"
BLUE  = "\033[1;37;34m%s\033[0m"
GREEN = "\033[1;37;32m%s\033[0m"

def info_blue(s, check=True):
    if MPI.process_number()==0 and check:
        print BLUE % s

def info_green(s, check=True):
    if MPI.process_number()==0 and check:
        print GREEN % s
    
def info_red(s, check=True):
    if MPI.process_number()==0 and check:
        print RED % s

Timer.__init__0 = Timer.__init__
def timer_init(self, task, verbose=False):
    info_blue(task, verbose)
    self.__init__0(task)
Timer.__init__ = timer_init

class OasisTimer(Timer):
    def __init__(self, task, verbose=False):
        Timer.__init__(self, task)
        info_blue(task, verbose)

def getMyMemoryUsage():
    mypid = getpid()
    mymemory = getoutput("ps -o rss %s" % mypid).split()[1]
    return mymemory

def dolfin_memory_usage(s):
    # Check how much memory is actually used by dolfin before we allocate anything
    dolfin_memory_use = getMyMemoryUsage()
    info_red('Memory use {} = '.format(s) + dolfin_memory_use)
    return dolfin_memory_use

# Print memory use up til now
initial_memory_use = dolfin_memory_usage('plain dolfin')

def body_force(mesh, **NS_namespace):
    """Specify body force"""
    return Constant((0,)*mesh.geometry().dim())

def scalar_source(scalar_components, **NS_namespace):
    fs = dict((ci, Constant(0)) for ci in scalar_components)
    return fs
    
def initialize(**NS_namespace):
    """Initialize solution. """
    pass

def create_bcs(sys_comp, **NS_namespace):
    """Return dictionary of Dirichlet boundary conditions."""
    return dict((ui, []) for ui in sys_comp)

def tentative_velocity_hook(ui, use_krylov_solvers, u_sol, **NS_namespace):
    """Called just prior to solving for tentative velocity."""
    pass

def pressure_hook(**NS_namespace):
    """Called prior to pressure solve."""
    pass

def scalar_hook(**NS_namespace):
    """Called prior to scalar solve."""
    pass

def start_timestep_hook(**NS_parameters):
    """Called at start of new timestep"""
    pass

def temporal_hook(**NS_namespace):
    """Called at end of a timestep."""
    pass

def pre_solve_hook(**NS_namespace):
    """Called just prior to entering time-loop. Must return a dictionary."""
    return {}

def theend(**NS_namespace):
    """Called at the very end."""
    pass
