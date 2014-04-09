__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-09"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from problems import *

# Default parameters NSfracStep solver
NS_parameters.update(
  # Physical constants and solver parameters
  nu = 0.01,             # Kinematic viscosity
  omega = 1.0,           # Underrelaxation factor
  
  # Some discretization options
  solver = "default",    # "default", "naive" 
  
  # Parameters used to tweek solver  
  max_iter = 10,         # Number of inner pressure velocity iterations on timestep
  max_error = 1e-8,      # Tolerance for absolute error
  print_velocity_pressure_convergence = False,
  
  # Parameters used to tweek output  
  plot_interval = 10,    
  folder = 'results',    # Relative folder for storing results 
)

def NS_hook(**NS_namespace):
    pass

def start_new_iter_hook(**NS_namespace):
    pass
  
def end_iter_hook(**NS_namespace):
    pass
