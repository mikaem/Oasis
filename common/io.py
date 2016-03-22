__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-26"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from os import makedirs, getcwd, listdir, remove, system, path
import cPickle
from dolfin import MPI, Function, XDMFFile, HDF5File, info_red, \
    VectorFunctionSpace, mpi_comm_world, FunctionAssigner

__all__ = ["create_initial_folders", "save_solution", "save_tstep_solution_h5",
           "save_checkpoint_solution_h5", "check_if_kill", "check_if_reset_statistics",
           "init_from_restart"]

def create_initial_folders(folder, restart_folder, sys_comp, tstep, info_red,
                           scalar_components, output_timeseries_as_vector, 
                           **NS_namespace):
    """Create necessary folders."""
    info_red("Creating initial folders")
    # To avoid writing over old data create a new folder for each run
    if MPI.rank(mpi_comm_world()) == 0:
        try:
            makedirs(folder)
        except OSError:
            pass

    MPI.barrier(mpi_comm_world())
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

    MPI.barrier(mpi_comm_world())
    if MPI.rank(mpi_comm_world()) == 0:
        if not restart_folder:
            #makedirs(path.join(newfolder, "Voluviz"))
            #makedirs(path.join(newfolder, "Stats"))
            #makedirs(path.join(newfolder, "VTK"))
            makedirs(path.join(newfolder, "Timeseries"))
            makedirs(path.join(newfolder, "Checkpoint"))
            
    tstepfolder = path.join(newfolder, "Timeseries")
    tstepfiles = {}
    comps = sys_comp
    if output_timeseries_as_vector:
        comps = ['p', 'u'] + scalar_components 
        
    for ui in comps:
        tstepfiles[ui] = XDMFFile(mpi_comm_world(), path.join(tstepfolder, ui+'_from_tstep_{}.xdmf'.format(tstep)))
        tstepfiles[ui].parameters["rewrite_function_mesh"] = False
        tstepfiles[ui].parameters["flush_output"] = True
    
    return newfolder, tstepfiles

def save_solution(tstep, t, q_, q_1, folder, newfolder, save_step, checkpoint, 
                  NS_parameters, tstepfiles, u_, u_components, scalar_components,
                  output_timeseries_as_vector, constrained_domain, 
                  AssignedVectorFunction, **NS_namespace):
    """Called at end of timestep. Check for kill and save solution if required."""
    NS_parameters.update(t=t, tstep=tstep)
    if tstep % save_step == 0: 
        save_tstep_solution_h5(tstep, q_, u_, newfolder, tstepfiles, constrained_domain,
                               output_timeseries_as_vector, u_components, AssignedVectorFunction,
                               scalar_components, NS_parameters)
        
    killoasis = check_if_kill(folder)
    if tstep % checkpoint == 0 or killoasis:
        save_checkpoint_solution_h5(tstep, q_, q_1, newfolder, u_components, 
                                    NS_parameters)
        
    return killoasis

def save_tstep_solution_h5(tstep, q_, u_, newfolder, tstepfiles, constrained_domain,
                           output_timeseries_as_vector, u_components, AssignedVectorFunction,
                           scalar_components, NS_parameters):
    """Store solution on current timestep to XDMF file."""
    timefolder = path.join(newfolder, 'Timeseries')
    if output_timeseries_as_vector:
        # project or store velocity to vector function space
        for comp, tstepfile in tstepfiles.iteritems():
            if comp == "u":
                V = q_['u0'].function_space()
                # First time around create vector function and assigners
                if not hasattr(tstepfile, 'uv'):
                    tstepfile.uv = AssignedVectorFunction(u_)

                # Assign solution to vector
                tstepfile.uv()
                    
                # Store solution vector
                tstepfile.write(tstepfile.uv, float(tstep))
            
            elif comp in q_:                
                tstepfile.write(q_[comp], float(tstep))
            
            else:
                tstepfile.write(tstepfile.function, float(tstep))
            
    else:
        for comp, tstepfile in tstepfiles.iteritems():
            tstepfile << (q_[comp], float(tstep))
        
    if MPI.rank(mpi_comm_world()) == 0:
        if not path.exists(path.join(timefolder, "params.dat")):
            f = open(path.join(timefolder, 'params.dat'), 'w')
            cPickle.dump(NS_parameters,  f)

def save_checkpoint_solution_h5(tstep, q_, q_1, newfolder, u_components, 
                                NS_parameters):
    """Overwrite solution in Checkpoint folder. 
    
    For safety reasons, in case the solver is interrupted, take backup of 
    solution first.
    
    Must be restarted using the same mesh-partitioning. This will be fixed
    soon. (MM)
    
    """
    checkpointfolder = path.join(newfolder, "Checkpoint")
    NS_parameters["num_processes"] = MPI.size(mpi_comm_world())
    if MPI.rank(mpi_comm_world()) == 0:
        if path.exists(path.join(checkpointfolder, "params.dat")):
            system('cp {0} {1}'.format(path.join(checkpointfolder, "params.dat"),
                                        path.join(checkpointfolder, "params_old.dat")))
        f = open(path.join(checkpointfolder, "params.dat"), 'w')
        cPickle.dump(NS_parameters,  f)
        
    MPI.barrier(mpi_comm_world())
    for ui in q_:
        h5file = path.join(checkpointfolder, ui+'.h5')
        oldfile = path.join(checkpointfolder, ui+'_old.h5')
        # For safety reasons...
        if path.exists(h5file):
            if MPI.rank(mpi_comm_world()) == 0:
                system('cp {0} {1}'.format(h5file, oldfile))
        MPI.barrier(mpi_comm_world())
        ###
        newfile = HDF5File(mpi_comm_world(), h5file, 'w')
        newfile.flush()
        newfile.write(q_[ui].vector(), '/current')
        if ui in u_components:
            newfile.write(q_1[ui].vector(), '/previous')
        if path.exists(oldfile):
            if MPI.rank(mpi_comm_world()) == 0:
                system('rm {0}'.format(oldfile))
        MPI.barrier(mpi_comm_world())
    if MPI.rank(mpi_comm_world()) == 0 and path.exists(path.join(checkpointfolder, "params_old.dat")):
        system('rm {0}'.format(path.join(checkpointfolder, "params_old.dat")))
        
def check_if_kill(folder):
    """Check if user has put a file named killoasis in folder."""
    found = 0
    if 'killoasis' in listdir(folder):
        found = 1
    collective = MPI.sum(mpi_comm_world(), found)
    if collective > 0:
        if MPI.rank(mpi_comm_world()) == 0:
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
    collective = MPI.sum(mpi_comm_world(), found)    
    if collective > 0:        
        if MPI.rank(mpi_comm_world()) == 0:
            remove(path.join(folder, 'resetoasis'))
            info_red('resetoasis Found!')
        return True
    else:
        return False

def init_from_restart(restart_folder, sys_comp, uc_comp, u_components, 
                      q_, q_1, q_2, **NS_namespace):
    """Initialize solution from checkpoint files """
    if restart_folder:
        for ui in sys_comp:
            filename = path.join(restart_folder, ui + '.h5')
            hdf5_file = HDF5File(mpi_comm_world(), filename, "r")
            hdf5_file.read(q_[ui].vector(), "/current", False)      
            q_[ui].vector().apply('insert')
            # Check for the solution at a previous timestep as well
            if ui in uc_comp:
                q_1[ui].vector().zero()
                q_1[ui].vector().axpy(1., q_[ui].vector())
                q_1[ui].vector().apply('insert')
                if ui in u_components:
                    hdf5_file.read(q_2[ui].vector(), "/previous", False)
                    q_2[ui].vector().apply('insert')
