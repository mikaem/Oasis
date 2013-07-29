__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from DrivenCavity import *

# Override some problem specific parameters 
T = 2.5
dt = 0.01
folder = "drivencavityscalar_results"
newfolder = create_initial_folders(folder, dt)
NS_parameters.update(dict(
    nu = 0.01,
    T = T,
    dt = dt,
    plot_interval = 10,
    folder = folder,
    newfolder = newfolder,
    velocity_degree = 1,
    use_lumping_of_mass_matrix = True,
    use_krylov_solvers = True
  )
)
NS_parameters['krylov_solvers']['monitor_convergence'] = True

scalar_components = ['c', 'k']

pre_solve_hook_0 = pre_solve_hook
def pre_solve_hook(Vv, p_, c_, k_, **NS_namespace):    
    d = pre_solve_hook_0(Vv, p_, **NS_namespace)
    c_plotter = VTKPlotter(c_) 
    k_plotter = VTKPlotter(k_) 
    d.update(dict(c_plotter=c_plotter, k_plotter=k_plotter))
    return d

# Specify boundary conditions
create_bcs_0 = create_bcs
def create_bcs(V, sys_comp, **NS_namespace):
    bcs = create_bcs_0(V, sys_comp, **NS_namespace)
    bcs['c'] = [DirichletBC(V, 1., lid)]
    bcs['k'] = [DirichletBC(V, 2., lid)]
    return bcs

temporal_hook0 = temporal_hook    
def temporal_hook(tstep, u_, Vv, uv, p_, c_plotter, k_plotter, plot_interval, **NS_namespace):
    temporal_hook0(tstep, u_, Vv, uv, p_, plot_interval, **NS_namespace)
    if tstep % plot_interval == 0:
        c_plotter.plot()
        k_plotter.plot()

get_solvers_0 = get_solvers
def get_solvers(use_krylov_solvers, use_lumping_of_mass_matrix, 
                krylov_solvers, sys_comp, **NS_namespace):
    sols = get_solvers_0(use_krylov_solvers, use_lumping_of_mass_matrix, 
                krylov_solvers, sys_comp, **NS_namespace)
    if use_krylov_solvers:
        c_sol = KrylovSolver('bicgstab', 'ilu')
        c_sol.parameters.update(krylov_solvers)
        c_sol.parameters['preconditioner']['reuse'] = False
        c_sol.t = 0
    else:
        c_sol = LUSolver()
        c_sol.t = 0
    sols.append(c_sol)
    return sols
