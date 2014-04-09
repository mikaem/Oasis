__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-09"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from problems import *

# Default parameters NSfracStep solver
NS_parameters = dict(
  # Physical constants and solver parameters
  nu = 0.01,             # Kinematic viscosity
  t = 0.0,               # Time
  tstep = 0,             # Timestep
  T = 1.0,               # End time
  dt = 0.01,             # Time interval on each timestep
  
  # Some discretization options
  AB_projection_pressure = False,  # Use Adams Bashforth projection as first estimate for pressure on new timestep
  velocity_degree = 2,
  pressure_degree = 1,  
  solver = "IPCS_ABCN",  # "IPCS_ABCN", "IPCS_ABE", "IPCS", "Chorin"
  
  # Parameters used to tweek solver  
  max_iter = 1,          # Number of inner pressure velocity iterations on timestep
  max_error = 1e-6,      # Tolerance for inner iterations (pressure velocity iterations)
  iters_on_first_timestep = 2,  # Number of iterations on first timestep
  use_krylov_solvers = False,  # Otherwise use LU-solver
  low_memory_version = False,  # Use assembler and not preassembled matrices
  print_intermediate_info = 10,
  print_velocity_pressure_convergence = False,
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
    maximum_iterations = 200,
    relative_tolerance = 1e-8,
    absolute_tolerance = 1e-8)
)
  
def velocity_tentative_hook(**NS_namespace):
    """Called just prior to solving for tentative velocity."""
    pass

def pressure_hook(**NS_namespace):
    """Called prior to pressure solve."""
    pass

def start_timestep_hook(**NS_namespace):
    """Called at start of new timestep"""
    pass

def temporal_hook(**NS_namespace):
    """Called at end of a timestep."""
    pass