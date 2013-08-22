__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
import cPickle
from collections import defaultdict
from os import getpid, path, makedirs, getcwd, listdir, remove, system

#parameters["linear_algebra_backend"] = "Epetra"
parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters['mesh_partitioner'] = "ParMETIS"

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
  convection = "Standard", 
  print_intermediate_info = 10,
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
# With diffusivities given here. We use a Schmidt number defined as
# Schmidt = nu / D (= momentum diffusivity / mass diffusivity)
Schmidt = defaultdict(lambda: 1.)

def create_initial_folders(folder, restart_folder):
    # To avoid writing over old data create a new folder for each run
    if MPI.process_number() == 0:
        try:
            makedirs(folder)
        except OSError:
            pass

    newfolder = path.join(folder, 'data')
    if restart_folder:
        newfolder = path.join(newfolder, restart_folder.split('/')[-2])
    else:
        if not path.exists(newfolder):
            newfolder = path.join(newfolder, '1')
        else:
            previous = listdir(newfolder)
            newfolder = path.join(newfolder, str(max(map(eval, previous)) + 1))

    MPI.barrier()
    if MPI.process_number() == 0:
        try:
            makedirs(path.join(newfolder, "HDF5"))
        except OSError:
            pass
        try:
            makedirs(path.join(newfolder, "Stats"))
        except OSError:
            pass
        try:
            makedirs(path.join(newfolder, "Checkpoint"))
        except OSError:
            pass
        
    return newfolder

def save_solution(tstep, t, q_, q_1, folder, newfolder, save_step, checkpoint, 
                  NS_parameters, **NS_namespace):
    NS_parameters.update(t=t, tstep=tstep)
    if tstep % save_step == 0: 
        save_tstep_solution(tstep, q_, newfolder, NS_parameters)
    killoasis = check_if_kill(folder)    
    if tstep % checkpoint == 0 or killoasis:
        save_checkpoint_solution(tstep, q_, q_1, newfolder, NS_parameters)
    
    return killoasis

def save_tstep_solution(tstep, q_, newfolder, NS_parameters):  
    timefolder = path.join(newfolder, 'timestep='+str(tstep))
    if MPI.process_number() == 0:
        try:
            makedirs(timefolder)
        except OSError:
            pass
    MPI.barrier()
    if MPI.process_number() == 0:
        f = open(path.join(timefolder, 'params.dat'), 'w')
        cPickle.dump(NS_parameters,  f)

    for ui in q_.keys():
        newfile = File(path.join(timefolder, ui + '.xml.gz'))
        newfile << q_[ui]

def save_checkpoint_solution(tstep, q_, q_1, newfolder, NS_parameters):
    timefolder = path.join(newfolder, 'timestep='+str(tstep))
    checkpointfolder = path.join(newfolder, "Checkpoint")
    if MPI.process_number() == 0:
        f = open(path.join(checkpointfolder, 'params.dat'), 'w')
        cPickle.dump(NS_parameters, f)
        
    for ui in q_.keys():
        # Check if solution has already been stored in timestep folder
        if 'timestep='+str(tstep) in listdir(newfolder):
            system('cp {0} {1}'.format(path.join(timefolder, ui + '.xml.gz'), 
                                       path.join(checkpointfolder, ui + '.xml.gz')))
        else:
            cfile = File(path.join(checkpointfolder, ui + '.xml.gz'))
            cfile << q_[ui]   
        if not ui == 'p':
            cfile_1 = File(path.join(checkpointfolder, ui + '_1.xml.gz'))
            cfile_1 << q_1[ui]

def check_if_kill(folder):
    if 'killoasis' in listdir(folder):
        MPI.barrier()
        if MPI.process_number() == 0: 
            info_red('killoasis Found! Stopping simulations cleanly...')
            remove(path.join(folder, 'killoasis'))
        return True
    else:
        return False

def check_if_reset_statistics(folder):
    if 'resetoasis' in listdir(folder):
        MPI.barrier()
        if MPI.process_number() == 0: 
            info_red('resetoasis Found!')
            remove(path.join(folder, 'resetoasis'))
        return True
    else:
        return False
        
def body_force(mesh, **NS_namespace):
    """Specify body force"""
    return Constant((0,)*mesh.geometry().dim())

def convection_form(conv, u, v, U_, **NS_namespace):
    if conv == 'Standard':
        return inner(v, dot(U_, nabla_grad(u)))
        
    elif conv == 'Divergence':
        return inner(v, nabla_div(outer(U_, u)))
        
    elif conv == 'Divergence by parts':
        # Use with care. ds term could be important
        return -inner(grad(v), outer(U_, u))
        
    elif conv == 'Skew':
        return 0.5*(inner(v, dot(U_, nabla_grad(u))) + inner(v, nabla_div(outer(U_, u))))

    else:
        raise TypeError("Wrong convection form {}".format(conv))

def initialize(**NS_namespace):
    """Initialize solution. Could also be used to create new variables 
    or parameters that can be placed in the NS_namespace and then be
    used for, e.g., postprocessing"""
    return {}

def create_bcs(sys_comp, **NS_namespace):
    """Return dictionary of Dirichlet boundary conditions."""
    return dict((ui, []) for ui in sys_comp)

def get_solvers(use_krylov_solvers, use_lumping_of_mass_matrix, 
                krylov_solvers, sys_comp, bcs, x_, Q, 
                scalar_components, **NS_namespace):
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
        p_sol = KrylovSolver('gmres', 'hypre_amg')
        p_sol.parameters['preconditioner']['reuse'] = True
        p_sol.parameters.update(krylov_solvers)
        p_sol.t = 0
        if bcs['p'] == []:
            attach_pressure_nullspace(p_sol, x_, Q)
        sols = [u_sol, p_sol, du_sol]
        if len(scalar_components) > 0:
            c_sol = KrylovSolver('bicgstab', 'hypre_euclid')
            c_sol.parameters.update(krylov_solvers)
            c_sol.parameters['preconditioner']['reuse'] = False
            c_sol.t = 0
            sols.append(c_sol)
        else:
            sols.append(None)
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
        sols = [u_sol, p_sol, du_sol]    
        if len(scalar_components) > 0:
            c_sol = LUSolver()
            c_sol.t = 0
            sols.append(c_sol)
        else:
            sols.append(None)
        
    return sols

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
