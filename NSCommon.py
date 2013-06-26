__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import MPI, File
import cPickle
from os import getpid, path, makedirs, getcwd, listdir, remove, system

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
  checkpoint = 10, # Overwrite solution in Checkpoint folder each checkpoint tstep
  save_step = 10,  # Store solution in new folder each save_step tstep
  folder = 'results',
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

def save_solution(tstep, t, q_, q_1, params):
    params.update(t=t, tstep=tstep)
    if tstep % params['save_step'] == 0: 
        save_tstep_solution(tstep, q_, params)
    if tstep % params['checkpoint'] == 0:
        save_checkpoint_solution(tstep, q_, q_1, params)

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

def check_if_kill(tstep, t, q_, q_1, params):
    params.update(t=t, tstep=tstep)
    if 'killoasis' in listdir(params['folder']):
        save_checkpoint_solution(tstep, q_, q_1, params)
        MPI.barrier()
        if MPI.process_number() == 0: 
            remove(path.join(params['folder'], 'killoasis'))        
        return True
    else:
        return False
