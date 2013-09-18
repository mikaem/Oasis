__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
import cPickle
from collections import defaultdict
from os import getpid, path, makedirs, getcwd, listdir, remove, system
from numpy import array, maximum

#parameters["linear_algebra_backend"] = "Epetra"
parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["optimize"] = False
parameters["form_compiler"]["cpp_optimize"] = True
parameters['mesh_partitioner'] = "ParMETIS"

# Default parameters
NS_parameters = dict(
  # Physical constants and solver parameters
  nu = 0.01,             # Kinematic viscosity
  t = 0,                 # Time
  tstep = 0,             # Timestep
  T = 1.0,               # End time
  dt = 0.01,             # Time interval on each timestep
  
  # Some discretization options
  convection = "Standard",  # "Standard", "Skew", "Divergence", "Divergence by parts"
  AB_projection_pressure = False,  # Use Adams Bashforth projection as first estimate for pressure on new timestep
  velocity_degree = 2,
  pressure_degree = 1,  
  
  # Parameters used to tweek solver  
  max_iter = 1,          # Number of inner pressure velocity iterations on timestep
  max_error = 1e-6,      # Tolerance for inner iterations (pressure velocity iterations)
  iters_on_first_timestep = 2,  # Number of iterations on first timestep
  use_lumping_of_mass_matrix = False,  
  use_krylov_solvers = False,  # Otherwise use LU-solver
  low_memory_version = False,  # Use assembler and not preassembled matrices
  print_intermediate_info = 10,
  print_velocity_pressure_convergence = True,
  
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

def create_initial_folders(folder, restart_folder, sys_comp, tstep, 
                           scalar_components, output_timeseries_as_vector, **NS_namespace):
    """Create necessary folders."""
    
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
            previous = max(map(eval, previous)) if previous else 0
            newfolder = path.join(newfolder, str(previous + 1))

    MPI.barrier()
    if MPI.process_number() == 0:
        if not restart_folder:
            makedirs(path.join(newfolder, "Voluviz"))
            makedirs(path.join(newfolder, "Stats"))
            makedirs(path.join(newfolder, "VTK"))
            makedirs(path.join(newfolder, "Timeseries"))
            makedirs(path.join(newfolder, "Checkpoint"))
            
    tstepfolder = path.join(newfolder, "Timeseries")
    tstepfiles = {}
    comps = sys_comp
    if output_timeseries_as_vector:
        comps = ['p', 'u'] + scalar_components 
        
    for ui in comps:
        tstepfiles[ui] = XDMFFile(path.join(tstepfolder, ui+'_from_tstep_{}.xdmf'.format(tstep)))
        tstepfiles[ui].parameters["rewrite_function_mesh"] = False
    return newfolder, tstepfiles

def save_solution(tstep, t, q_, q_1, folder, newfolder, save_step, checkpoint, 
                  NS_parameters, tstepfiles, Vv, u_, u_components,
                  output_timeseries_as_vector, **NS_namespace):
    """Called at end of timestep. Check for kill and save solution if required."""
    NS_parameters.update(t=t, tstep=tstep)
    if tstep % save_step == 0: 
        save_tstep_solution_h5(tstep, q_, u_, newfolder, tstepfiles, Vv, 
                               output_timeseries_as_vector, u_components)
        
    killoasis = check_if_kill(folder)
    if tstep % checkpoint == 0 or killoasis:
        save_checkpoint_solution_h5(tstep, q_, q_1, newfolder, NS_parameters)
        
    return killoasis

def save_tstep_solution_h5(tstep, q_, u_, newfolder, tstepfiles, Vv,
                           output_timeseries_as_vector, u_components):
    """Store solution on current timestep to XDMF file."""
    timefolder = path.join(newfolder, 'Timeseries')
    if output_timeseries_as_vector:
        # project or store velocity to vector function space
        if not hasattr(tstepfiles['u'], 'uv'): # First time around only
            tstepfiles['u'].uv = Function(Vv)
            tstepfiles['u'].d = dict((ui, Vv.sub(i).dofmap().collapse(Vv.mesh())[1]) 
                                     for i, ui in enumerate(u_components))

        # The short but timeconsuming way:
        #tstepfiles['u'].uv.assign(project(u_, Vv))
        
        # Or the faster, but more comprehensive way:
        for ui in u_components:
            q_[ui].update()    
            vals = tstepfiles['u'].d[ui].values()
            keys = tstepfiles['u'].d[ui].keys()
            tstepfiles['u'].uv.vector()[vals] = q_[ui].vector()[keys]
        tstepfiles['u'] << (tstepfiles['u'].uv, float(tstep))
        
        # Store the rest of the solution functions
        for ui in ['p']+scalar_components:
            tstepfiles[ui] << (q_[ui], float(tstep))
    else:
        for ui in q_:
            tstepfiles[ui] << (q_[ui], float(tstep))
        
    if MPI.process_number() == 0:
        if not path.exists(path.join(timefolder, "params.dat")):
            f = open(path.join(timefolder, 'params.dat'), 'w')
            cPickle.dump(NS_parameters,  f)

def save_checkpoint_solution_h5(tstep, q_, q_1, newfolder, NS_parameters):
    """Overwrite solution in Checkpoint folder. 
    
    For safety reasons, in case the solver is interrupted, take backup of 
    solution first.
    
    """
    checkpointfolder = path.join(newfolder, "Checkpoint")
    if MPI.process_number() == 0:
        if path.exists(path.join(checkpointfolder, "params.dat")):
            system('cp {0} {1}'.format(path.join(checkpointfolder, "params.dat"),
                                        path.join(checkpointfolder, "params_old.dat")))
        f = open(path.join(checkpointfolder, "params.dat"), 'w')
        cPickle.dump(NS_parameters,  f)
        
    MPI.barrier()
    for ui in q_:
        h5file = path.join(checkpointfolder, ui+'.h5')
        oldfile = path.join(checkpointfolder, ui+'_old.h5')
        # For safety reasons...
        if path.exists(h5file):
            if MPI.process_number() == 0:
                system('cp {0} {1}'.format(h5file, oldfile))
        MPI.barrier()
        ###
        newfile = HDF5File(h5file, 'w')
        newfile.flush()
        newfile.write(q_[ui].vector(), '/current')
        if not ui == 'p':
            newfile.write(q_1[ui].vector(), '/previous')
        if path.exists(oldfile):
            if MPI.process_number() == 0:
                system('rm {0}'.format(oldfile))
        MPI.barrier()
    if MPI.process_number() == 0:
        system('rm {0}'.format(path.join(checkpointfolder, "params_old.dat")))
        
def check_if_kill(folder):
    """Check if user has put a file named killoasis in folder."""
    found = 0
    if 'killoasis' in listdir(folder):
        found = 1
    collective = MPI.sum(found)
    if collective > 0:
        if MPI.process_number() == 0:
            remove(path.join(folder, 'killoasis'))
        info_red('killoasis Found! Stopping simulations cleanly...')
        return True
    else:
        return False

def check_if_reset_statistics(folder):
    """Check if user has put a file named resetoasis in folder."""
    found = 0
    if 'resetoasis' in listdir(folder):
        found = 1
    collective = MPI.sum(found)    
    if collective > 0:        
        if MPI.process_number() == 0:
            remove(path.join(folder, 'resetoasis'))
        info_red('resetoasis Found!')
        return True
    else:
        return False

def body_force(mesh, **NS_namespace):
    """Specify body force"""
    return Constant((0,)*mesh.geometry().dim())

def scalar_source(scalar_components, **NS_namespace):
    fs = dict((ci, Constant(0)) for ci in scalar_components)
    return fs
    
def convection_form(conv, u, v, U_AB, **NS_namespace):
    if conv == 'Standard':
        return inner(v, dot(U_AB, nabla_grad(u)))
        
    elif conv == 'Divergence':
        return inner(v, nabla_div(outer(U_AB, u)))
        
    elif conv == 'Divergence by parts':
        # Use with care. ds term could be important
        return -inner(grad(v), outer(U_AB, u))
        
    elif conv == 'Skew':
        return 0.5*(inner(v, dot(U_AB, nabla_grad(u))) + inner(v, nabla_div(outer(U_AB, u))))

    else:
        raise TypeError("Wrong convection form {}".format(conv))

def assemble_lumped_P1_diagonal(V, M, **NS_namespace):
    ones = Function(V)
    ones.vector()[:] = 1.
    ML = M * ones.vector()
    ML.set_local(1. / ML.array())
    return ML
    
def initialize(**NS_namespace):
    """Initialize solution. """
    pass

def init_from_restart(restart_folder, sys_comp, uc_comp, u_components, 
               q_, q_1, q_2, **NS_namespace):
    """Initialize solution from checkpoint files """
    if restart_folder:
        for ui in sys_comp:
            filename = path.join(restart_folder, ui + '.h5')
            hdf5_file = HDF5File(filename, "r")
            hdf5_file.read(q_[ui].vector(), "/current")      
            q_[ui].vector().apply('insert')
            # Check for the solution at a previous timestep as well
            if ui in uc_comp:
                try:
                    hdf5_file.read(q_1[ui].vector(), "/previous")
                    q_1[ui].vector().apply('insert')
                except:
                    q_1[ui].vector()[:] = q_[ui].vector()[:]
                    q_1[ui].vector().apply('insert')
            if ui in u_components:
                q_2[ui].vector()[:] = q_1[ui].vector()[:]
                q_2[ui].vector().apply('insert')            

def create_bcs(sys_comp, **NS_namespace):
    """Return dictionary of Dirichlet boundary conditions."""
    return dict((ui, []) for ui in sys_comp)

def get_solvers(use_krylov_solvers, use_lumping_of_mass_matrix, 
                krylov_solvers, sys_comp, bcs, x_, Q, 
                scalar_components, **NS_namespace):
    """Return linear solvers. 
    
    We are solving for
       - tentative velocity
       - pressure correction
       - velocity update (unless lumping is switched on)
       
       and possibly:       
       - scalars
            
    """
    if use_krylov_solvers:
        ## tentative velocity solver ##
        u_sol = KrylovSolver('bicgstab', 'jacobi')
        u_sol.parameters.update(krylov_solvers)
        u_sol.parameters['preconditioner']['reuse'] = False
        u_sol.t = 0
        ## velocity correction solver
        if use_lumping_of_mass_matrix:
            du_sol = None
        else:
            du_sol = KrylovSolver('bicgstab', 'hypre_euclid')
            du_sol.parameters.update(krylov_solvers)
            du_sol.parameters['preconditioner']['reuse'] = True
            du_sol.t = 0            
        ## pressure solver ##
        p_sol = KrylovSolver('gmres', 'hypre_amg')
        p_sol.parameters['preconditioner']['reuse'] = True
        p_sol.parameters.update(krylov_solvers)
        p_sol.t = 0
        if bcs['p'] == []:
            attach_pressure_nullspace(p_sol, x_, Q)
        sols = [u_sol, p_sol, du_sol]
        ## scalar solver ##
        if len(scalar_components) > 0:
            #c_sol = KrylovSolver('bicgstab', 'hypre_euclid')
            c_sol = KrylovSolver('bicgstab', 'jacobi')
            c_sol.parameters.update(krylov_solvers)
            c_sol.parameters['preconditioner']['reuse'] = False
            c_sol.t = 0
            sols.append(c_sol)
        else:
            sols.append(None)
    else:
        ## tentative velocity solver ##
        u_sol = LUSolver()
        u_sol.t = 0
        ## velocity correction ##
        if use_lumping_of_mass_matrix:
            du_sol = None
        else:
            du_sol = LUSolver()
            du_sol.parameters['reuse_factorization'] = True
            du_sol.t = 0
        ## pressure solver ##
        p_sol = LUSolver()
        p_sol.parameters['reuse_factorization'] = True
        p_sol.t = 0  
        if bcs['p'] == []:
            p_sol.normalize = True
        sols = [u_sol, p_sol, du_sol]
        ## scalar solver ##
        if len(scalar_components) > 0:
            c_sol = LUSolver()
            c_sol.t = 0
            sols.append(c_sol)
        else:
            sols.append(None)
        
    return sols

def attach_pressure_nullspace(p_sol, x_, Q):
    """Create null space basis object and attach to Krylov solver."""
    null_vec = Vector(x_['p'])
    Q.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0/null_vec.norm("l2")
    null_space = VectorSpaceBasis([null_vec])
    p_sol.set_nullspace(null_space)
    p_sol.null_space = null_space

def add_pressure_gradient_rhs(b, x_, P, p_, v, i, ui, **NS_namespace):
    """Add pressure gradient on rhs of tentative velocity equation."""
    if P:
        b[ui].axpy(-1., P[ui]*x_['p'])
    else:
        b[ui].axpy(-1., assemble(v*p_.dx(i)*dx))

def add_pressure_gradient_rhs_update(b, dt, P, dp_, v, i, ui, **NS_namespace):
    """Add pressure gradient on rhs of velocity update equation."""
    if P:
        b[ui].axpy(-dt, P[ui]*dp_.vector())
    else:
        b[ui].axpy(-dt, assemble(v*dp_.dx(i)*dx))
        
def assemble_pressure_rhs(b, Rx, x_, dt, q, u_, Ap, **NS_namespace):
    """Assemble rhs of pressure equation."""
    b['p'][:] = 0.
    if Rx:
        for ui in Rx:
            b['p'].axpy(-1./dt, Rx[ui]*x_[ui])
    else:
        b['p'].axpy(-1./dt, assemble(div(u_)*q*dx))
    b['p'].axpy(1., Ap*x_['p'])
    
def solve_pressure(dp_, x_, Ap, b, p_sol, **NS_namespace):
    """Solve pressure equation."""
    dp_.vector()[:] = x_['p'][:]
    # KrylovSolvers use nullspace for normalization of pressure
    if hasattr(p_sol, 'null_space'):
        p_sol.null_space.orthogonalize(b['p']);

    p_sol.solve(Ap, x_['p'], b['p'])
    
    # LUSolver use normalize directly for normalization of pressure
    if hasattr(p_sol, 'normalize'):
        normalize(x_['p'])

    dp_.vector()[:] = x_['p'][:] - dp_.vector()[:]

def solve_scalar(ci, scalar_components, Ta, Tb, b, x_, bb, bx, bcs, c_sol, 
                  **NS_namespace):
    if len(scalar_components) > 1: 
        # Reuse solver for all scalars. This requires the same matrix and vectors to be used by c_sol.
        Tb._scale(0.)
        Tb.axpy(1., Ta, True)
        bb[:] = b[ci][:]
        bx[:] = x_[ci][:]
        [bc.apply(Tb, bb) for bc in bcs[ci]]
        c_sol.solve(Tb, bx, bb)
        x_[ci][:] = bx[:]
    else:
        [bc.apply(Ta, b[ci]) for bc in bcs[ci]]
        c_sol.solve(Ta, x_[ci], b[ci])    
    #x_[ci][x_[ci] < 0] = 0.               # Bounded solution
    #x_[ci].set_local(maximum(0., x_[ci].array()))
    #x_[ci].apply("insert")

def update_velocity_lumping(P, dp_, ML, dt, x_, v, u_components, **NS_namespace):
    for i, ui in enumerate(u_components):
        if P:
            x_[ui].axpy(-dt, (P[ui] * dp_.vector()) * ML)
        else:
            x_[ui].axpy(-dt, (assemble(v*dp_.dx(i)*dx)) * ML)
            
def velocity_tentative_hook(ui, use_krylov_solvers, u_sol, **NS_namespace):
    """Called just prior to solving for tentative velocity."""
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

