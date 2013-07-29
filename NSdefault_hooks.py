__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
import cPickle
from os import getpid, path, makedirs, getcwd, listdir, remove, system

#parameters["linear_algebra_backend"] = "Epetra"
parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

# Default parameters
NS_parameters = dict(
  nu = 0.01,
  t = 0,
  tstep = 0,
  T = 1.0,
  max_iter = 1,
  max_error = 1e-6,
  iters_on_first_timestep = 2,
  dt = 0.01,
  plot_interval = 10,
  checkpoint = 10,       # Overwrite solution in Checkpoint folder each checkpoint tstep
  save_step = 10,        # Store solution in new folder each save_step tstep
  folder = 'results',    # Relative folder for storing results 
  restart_folder = None, # If restarting solution, set the folder holder the solution to start from here
  use_lumping_of_mass_matrix = False,
  use_krylov_solvers = False,
  velocity_degree = 2,
  pressure_degree = 1,  
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

def create_initial_folders(folder, dt):
    # To avoid writing over old data create a new folder for each run
    newfolder = path.join(folder, 'data', 'dt={0:2.4e}'.format(dt))
    if not path.exists(newfolder):
        newfolder = path.join(newfolder, '1')
    else:
        previous = listdir(newfolder)
        newfolder = path.join(newfolder, str(max(map(eval, previous)) + 1))

    MPI.barrier()
    h5folder = path.join(newfolder, "HDF5")
    statsfolder = path.join(newfolder, "Stats")
    checkpointfolder = path.join(newfolder, "Checkpoint")

    if MPI.process_number() == 0:
        makedirs(h5folder)
        makedirs(statsfolder)
        makedirs(checkpointfolder)
        
    return newfolder

def save_solution(tstep, t, q_, q_1, NS_parameters, save_step, checkpoint, 
                  **NS_namespace):
    NS_parameters.update(t=t, tstep=tstep)
    if tstep % save_step == 0: 
        save_tstep_solution(tstep, q_, NS_parameters)
    if tstep % checkpoint == 0:
        save_checkpoint_solution(tstep, q_, q_1, NS_parameters)

def save_tstep_solution(tstep, q_, params):        
    newfolder = path.join(params['newfolder'], 'timestep='+str(tstep))
    if MPI.process_number() == 0:
        try:
            makedirs(newfolder)
        except OSError:
            pass
    MPI.barrier()
    if MPI.process_number() == 0:
        f = open(path.join(newfolder, 'params.dat'), 'w')
        cPickle.dump(params,  f)

    for ui in q_.keys():
        newfile = File(path.join(newfolder, ui + '.xml.gz'))
        newfile << q_[ui]

def save_checkpoint_solution(tstep, q_, q_1, params):
    newfolder = path.join(params['newfolder'], 'timestep='+str(tstep))
    checkpointfolder = path.join(params['newfolder'], "Checkpoint")
    if MPI.process_number() == 0:
        f = open(path.join(checkpointfolder, 'params.dat'), 'w')
        cPickle.dump(params, f)
        
    for ui in q_.keys():
        # Check if solution has already been stored in timestep folder
        if 'timestep='+str(tstep) in listdir(params['newfolder']):
            system('cp {0} {1}'.format(path.join(newfolder, ui + '.xml.gz'), 
                                       path.join(checkpointfolder, ui + '.xml.gz')))
        else:
            cfile = File(path.join(checkpointfolder, ui + '.xml.gz'))
            cfile << q_[ui]   
        if not ui == 'p':
            cfile_1 = File(path.join(checkpointfolder, ui + '_1.xml.gz'))
            cfile_1 << q_1[ui]

def check_if_kill(tstep, t, q_, q_1, NS_parameters, folder, info_red, **NS_namespace):
    NS_parameters.update(t=t, tstep=tstep)
    if 'killoasis' in listdir(folder):
        info_red('killoasis Found! Stopping simulations cleanly...')
        save_checkpoint_solution(tstep, q_, q_1, NS_parameters)
        MPI.barrier()
        if MPI.process_number() == 0: 
            remove(path.join(folder, 'killoasis'))
        return True
    else:
        return False
        
def body_force(mesh, **NS_namespace):
    """Specify body force"""
    return Constant((0,)*mesh.geometry().dim())

def initialize(**NS_namespace):
    """Initialize solution. Could also be used to create new variables 
    or parameters that can be placed in the NS_namespace and then be
    used for, e.g., postprocessing"""
    return {}

def create_bcs(sys_comp, **NS_namespace):
    """Return dictionary of Dirichlet boundary conditions."""
    return dict((ui, []) for ui in sys_comp)

def get_solvers(use_krylov_solvers, use_lumping_of_mass_matrix, 
                krylov_solvers, sys_comp, bcs, x_, Q, **NS_namespace):
    """Return solvers for all fields we are solving for.
    In case of lumping return None for velocity update."""
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
        p_sol = KrylovSolver('gmres', 'petsc_amg')
        p_sol.parameters['preconditioner']['reuse'] = True
        p_sol.parameters.update(krylov_solvers)
        p_sol.t = 0
        if bcs['p'] == []:
            attach_pressure_nullspace(p_sol, x_, Q)
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
        if bcs['p'] == []:
            p_sol.normalize = True
            
    return [u_sol, p_sol, du_sol] 

def attach_pressure_nullspace(p_sol, x_, Q):
    # Create vector that spans the null space
    null_vec = Vector(x_['p'])
    Q.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0/null_vec.norm("l2")
    # Create null space basis object and attach to Krylov solver
    null_space = VectorSpaceBasis([null_vec])
    p_sol.set_nullspace(null_space)
    p_sol.null_space = null_space

def velocity_tentative_hook(ui, use_krylov_solvers, u_sol, 
                                 **NS_namespace):
    """Called just prior to solving tentative velocity"""
    if use_krylov_solvers:
        if ui == "u0":
            u_sol.parameters['preconditioner']['reuse'] = False
        else:
            u_sol.parameters['preconditioner']['reuse'] = True

def pressure_hook(**NS_namespace):
    """Called prior to pressure solve."""
    pass

def start_timestep_hook(**NS_parameters):
    """Called at start of new timestep"""
    pass

def velocity_update_hook(**NS_namespace):
    """Called prior to velocity update solve."""
    pass

def scalar_hook(**NS_namespace):
    """Called prior to scalar solve."""
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
