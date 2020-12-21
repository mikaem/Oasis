__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-26"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from os import makedirs, getcwd, listdir, remove, system, path
from xml.etree import ElementTree as ET
import pickle
import time
import glob
from dolfin import (MPI, Function, XDMFFile, HDF5File,
    VectorFunctionSpace, FunctionAssigner)
from oasis.problems import info_red

__all__ = ["create_initial_folders", "save_solution", "save_tstep_solution_h5",
           "save_checkpoint_solution_h5", "check_if_kill", "check_if_reset_statistics",
           "init_from_restart", "merge_visualization_files", "merge_xml_files"]


def create_initial_folders(folder, restart_folder, sys_comp, tstep, info_red,
                           scalar_components, output_timeseries_as_vector,
                           **NS_namespace):
    """Create necessary folders."""
    info_red("Creating initial folders")
    # To avoid writing over old data create a new folder for each run
    if MPI.rank(MPI.comm_world) == 0:
        try:
            makedirs(folder)
        except OSError:
            pass

    MPI.barrier(MPI.comm_world)
    newfolder = path.join(folder, 'data')
    if restart_folder:
        newfolder = path.join(newfolder, restart_folder.split('/')[-2])
    else:
        if not path.exists(newfolder):
            newfolder = path.join(newfolder, '1')
        else:
            #previous = listdir(newfolder)
            previous = [f for f in listdir(newfolder) if not f.startswith('.')]
            previous = max(map(eval, previous)) if previous else 0
            newfolder = path.join(newfolder, str(previous + 1))

    MPI.barrier(MPI.comm_world)
    if MPI.rank(MPI.comm_world) == 0:
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
        tstepfiles[ui] = XDMFFile(MPI.comm_world, path.join(
            tstepfolder, ui + '_from_tstep_{}.xdmf'.format(tstep)))
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

    pauseoasis = check_if_pause(folder)
    while pauseoasis:
        time.sleep(5)
        pauseoasis = check_if_pause(folder)

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
        for comp, tstepfile in tstepfiles.items():
            if comp == "u":
                # Create vector function and assigners
                uv = AssignedVectorFunction(u_)

                # Assign solution to vector
                uv()

                # Store solution vector
                tstepfile.write(uv, float(tstep))

            elif comp in q_:
                tstepfile.write(q_[comp], float(tstep))

            else:
                tstepfile.write(tstepfile.function, float(tstep))

    else:
        for comp, tstepfile in tstepfiles.items():
            tstepfile << (q_[comp], float(tstep))

    if MPI.rank(MPI.comm_world) == 0:
        if not path.exists(path.join(timefolder, "params.dat")):
            f = open(path.join(timefolder, 'params.dat'), 'wb')
            pickle.dump(NS_parameters,  f)


def save_checkpoint_solution_h5(tstep, q_, q_1, newfolder, u_components,
                                NS_parameters):
    """Overwrite solution in Checkpoint folder.

    For safety reasons, in case the solver is interrupted, take backup of
    solution first.

    Must be restarted using the same mesh-partitioning. This will be fixed
    soon. (MM)

    """
    checkpointfolder = path.join(newfolder, "Checkpoint")
    NS_parameters["num_processes"] = MPI.size(MPI.comm_world)
    if MPI.rank(MPI.comm_world) == 0:
        if path.exists(path.join(checkpointfolder, "params.dat")):
            system('cp {0} {1}'.format(path.join(checkpointfolder, "params.dat"),
                                       path.join(checkpointfolder, "params_old.dat")))
        f = open(path.join(checkpointfolder, "params.dat"), 'wb')
        pickle.dump(NS_parameters,  f)

    MPI.barrier(MPI.comm_world)
    for ui in q_:
        h5file = path.join(checkpointfolder, ui + '.h5')
        oldfile = path.join(checkpointfolder, ui + '_old.h5')
        # For safety reasons...
        if path.exists(h5file):
            if MPI.rank(MPI.comm_world) == 0:
                system('cp {0} {1}'.format(h5file, oldfile))
        MPI.barrier(MPI.comm_world)
        ###
        newfile = HDF5File(MPI.comm_world, h5file, 'w')
        newfile.flush()
        newfile.write(q_[ui].vector(), '/current')
        if ui in u_components:
            newfile.write(q_1[ui].vector(), '/previous')
        if path.exists(oldfile):
            if MPI.rank(MPI.comm_world) == 0:
                system('rm {0}'.format(oldfile))
        MPI.barrier(MPI.comm_world)
    if MPI.rank(MPI.comm_world) == 0 and path.exists(path.join(checkpointfolder, "params_old.dat")):
        system('rm {0}'.format(path.join(checkpointfolder, "params_old.dat")))


def check_if_kill(folder):
    """Check if user has put a file named killoasis in folder."""
    found = 0
    if 'killoasis' in listdir(folder):
        found = 1
    collective = MPI.sum(MPI.comm_world, found)
    if collective > 0:
        if MPI.rank(MPI.comm_world) == 0:
            remove(path.join(folder, 'killoasis'))
            info_red('killoasis Found! Stopping simulations cleanly...')
        return True
    else:
        return False


def check_if_pause(folder):
    """Check if user has put a file named pauseoasis in folder."""
    found = 0
    if 'pauseoasis' in listdir(folder):
        found = 1
    collective = MPI.sum(MPI.comm_world, found)
    if collective > 0:
        if MPI.rank(MPI.comm_world) == 0:
            info_red('pauseoasis Found! Simulations paused. Remove ' + path.join(folder, 'pauseoasis') + ' to resume simulations...')
        return True
    else:
        return False


def check_if_reset_statistics(folder):
    """Check if user has put a file named resetoasis in folder."""
    found = 0
    if 'resetoasis' in listdir(folder):
        found = 1
    collective = MPI.sum(MPI.comm_world, found)
    if collective > 0:
        if MPI.rank(MPI.comm_world) == 0:
            remove(path.join(folder, 'resetoasis'))
            info_red('resetoasis Found!')
        return True
    else:
        return False


def init_from_restart(restart_folder, sys_comp, uc_comp, u_components,
                      q_, q_1, q_2, tstep, **NS_namespace):
    """Initialize solution from checkpoint files """
    if restart_folder:
        if MPI.rank(MPI.comm_world) == 0:
            info_red('Restarting from checkpoint at time step {}'.format(tstep))

        for ui in sys_comp:
            filename = path.join(restart_folder, ui + '.h5')
            hdf5_file = HDF5File(MPI.comm_world, filename, "r")
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


def merge_visualization_files(newfolder, **namesapce):
    timefolder = path.join(newfolder, 'Timeseries')
    # Gather files
    xdmf_files = list(glob.glob(path.join(timefolder, "*.xdmf")))
    xdmf_velocity = [f for f in xdmf_files if "u_from_tstep" in f.__str__()]
    xdmf_pressure = [f for f in xdmf_files if "p_from_tstep" in f.__str__()]

    # Merge files
    for files in [xdmf_velocity, xdmf_pressure]:
        if len(files) > 1:
            merge_xml_files(files)


def merge_xml_files(files):
    # Get first timestep and trees
    first_timesteps = []
    trees = []
    for f in files:
        trees.append(ET.parse(f))
        root = trees[-1].getroot()
        first_timesteps.append(float(root[0][0][0][2].attrib["Value"]))

    # Index valued sort (bypass numpy dependency)
    first_timestep_sorted = sorted(first_timesteps)
    indexes = [first_timesteps.index(i) for i in first_timestep_sorted]

    # Get last timestep of first tree
    base_tree = trees[indexes[0]]
    last_node = base_tree.getroot()[0][0][-1]
    ind = 1 if len(last_node.getchildren()) == 3 else 2
    last_timestep = float(last_node[ind].attrib["Value"])

    # Append
    for index in indexes[1:]:
        tree = trees[index]
        for node in tree.getroot()[0][0].getchildren():
            ind = 1 if len(node.getchildren()) == 3 else 2
            if last_timestep < float(node[ind].attrib["Value"]):
                base_tree.getroot()[0][0].append(node)
                last_timestep = float(node[ind].attrib["Value"])

    # Seperate xdmf files
    new_file = [f for f in files if "_0" in f]
    old_files = [f for f in files if "_" in f and f not in new_file]

    # Write new xdmf file
    base_tree.write(new_file[0], xml_declaration=True)

    # Delete xdmf file
    [remove(f) for f in old_files]
