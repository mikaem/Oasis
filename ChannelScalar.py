__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from Channel import *

# restart_folder in Channel.py must be set to None
assert(restart_folder==None) 

#restart_folder = 'channel_results/data/dt=5.0000e-02/4/Checkpoint'
#restart_folder = 'channel_results/data/dt=5.0000e-02/10/timestep=60'
restart_folder = None

### If restarting previous solution then read in parameters ########
if not restart_folder is None:
    restart_folder = path.join(getcwd(), restart_folder)
    f = open(path.join(restart_folder, 'params.dat'), 'r')
    NS_parameters.update(cPickle.load(f))
    NS_parameters['restart_folder'] = restart_folder
    NS_parameters['T'] = 2.0
    globals().update(NS_parameters)
#################### restart #######################################

if restart_folder is None:
    # Override some problem specific parameters and put the variables in DC_dict
    T = 1.
    dt = 0.05
    folder = "channelscalar_results"
    newfolder = create_initial_folders(folder, dt)
    NS_parameters.update(dict(
        update_statistics = 10,
        check_save_h5 = 10,
        nu = 2.e-5,
        Re_tau = 395.,
        T = T,
        dt = dt,
        folder = folder,
        newfolder = newfolder,
        use_krylov_solvers = True,
        use_lumping_of_mass_matrix = False
      )
    )
    NS_parameters.update(dict(statsfolder = path.join(newfolder, "Stats"),
                              h5folder = path.join(newfolder, "HDF5")))

    
# Specify boundary conditions
create_bcs0 = create_bcs
def create_bcs(V, q_, q_1, q_2, sys_comp, u_components, **NS_namespace):
    bcs = create_bcs0(V, q_, q_1, q_2, sys_comp, u_components, **NS_namespace)
    bcs['c'] = [DirichletBC(V, Constant(1), walls)]        
    return bcs
    
initialize0 = initialize
def initialize(V, Vv, q_, q_1, q_2, bcs, restart_folder, **NS_namespace):
    kw = initialize0(V, Vv, q_, q_1, q_2, bcs, restart_folder, **NS_namespace)
    if restart_folder is None:
        bcs['c'][0].apply(q_['c'].vector())
    return kw

temporal_hook0 = temporal_hook
def temporal_hook(q_, u_, V, Vv, tstep, uv, voluviz, stats, statsfolder, h5folder, 
                  update_statistics, check_save_h5, **NS_namespace):
    temporal_hook0(q_, u_, V, Vv, tstep, uv, voluviz, stats, statsfolder, h5folder, 
                   update_statistics, check_save_h5, **NS_namespace)
    if tstep % check_save_h5 == 0:
        voluviz(q_['c'])
        voluviz.toh5(0, tstep, filename=h5folder+"/snapshot_c_{}.h5".format(tstep))
        voluviz.probes.clear()

get_solvers_0 = get_solvers
def get_solvers(use_krylov_solvers, use_lumping_of_mass_matrix, 
                krylov_solvers, sys_comp, **NS_namespace):
    sols = get_solvers_0(use_krylov_solvers, use_lumping_of_mass_matrix, 
                krylov_solvers, sys_comp, **NS_namespace)
    if use_krylov_solvers:
        c_sol = KrylovSolver('bicgstab', 'jacobi')
        c_sol.parameters.update(krylov_solvers)
        c_sol.parameters['preconditioner']['reuse'] = False
        c_sol.t = 0
    else:
        c_sol = LUSolver()
        c_sol.t = 0
    sols.append(c_sol)
    return sols
