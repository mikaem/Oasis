__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from Oasis import *
from fenicstools import StructuredGrid, ChannelGrid
from numpy import arctan, array, cos, pi
import random

# Create a mesh here
Lx = 2.*pi
Ly = 2.
Lz = pi
Nx = 20
Ny = 12
Nz = 12
NS_parameters.update(dict(Lx=Lx, Ly=Ly, Lz=Lz, Nx=Nx, Ny=Ny, Nz=Nz))

#restart_folder = 'channel_results/data/dt=5.0000e-02/4/Checkpoint'
#restart_folder = 'channel_results/data/dt=5.0000e-02/10/timestep=60'
restart_folder = None

### If restarting from previous solution then read in parameters ########
if not restart_folder is None:
    restart_folder = path.join(getcwd(), restart_folder)
    f = open(path.join(restart_folder, 'params.dat'), 'r')
    NS_parameters.update(cPickle.load(f))
    NS_parameters['restart_folder'] = restart_folder
    NS_parameters['T'] = 2.0
    globals().update(NS_parameters)
#################### restart #######################################

mesh = BoxMesh(0., -Ly/2., -Lz/2., Lx, Ly/2., Lz/2., Nx, Ny, Nz)
# Create stretched mesh in y-direction
x = mesh.coordinates() 
x[:, 1] = cos(pi*(x[:, 1]-1.) / 2.)  
del x

class PeriodicDomain(SubDomain):

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        #return bool((near(x[0], 0) or near(x[2], -Lz/2.)) and 
                #(not ((near(x[0], Lx) and near(x[2], -Lz/2.)) or 
                      #(near(x[0], 0) and near(x[2], Lz/2.)))) and on_boundary)
        return bool((near(x[0], 0) or near(x[2], -Lz/2.)) and 
                (not (near(x[0], Lx) or near(x[2], Lz/2.))) and on_boundary)
                      
    def map(self, x, y):
        if near(x[0], Lx) and near(x[2], Lz/2.):
            y[0] = x[0] - Lx
            y[1] = x[1] 
            y[2] = x[2] - Lz
        elif near(x[0], Lx):
            y[0] = x[0] - Lx
            y[1] = x[1]
            y[2] = x[2]
        else: # near(x[2], Lz/2.):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - Lz
            
constrained_domain = PeriodicDomain()

if restart_folder is None:
    # Override some problem specific parameters and put the variables in DC_dict
    T = 1.
    dt = 0.05
    nu = 2.e-5
    Re_tau = 395.
    folder = "channel_results"
    newfolder = create_initial_folders(folder, dt)
    NS_parameters.update(dict(
        update_statistics = 10,
        check_save_h5 = 10,
        nu = nu,
        Re_tau = Re_tau,
        T = T,
        dt = dt,
        folder = folder,
        newfolder = newfolder,
        use_krylov_solvers = True,
        use_lumping_of_mass_matrix = False
      )
    )

if NS_parameters['velocity_degree'] > 1:
    NS_parameters['use_lumping_of_mass_matrix'] = False

NS_parameters.update(dict(statsfolder = path.join(newfolder, "Stats"),
                          h5folder = path.join(newfolder, "HDF5")))

# Specify body force
utau = nu * Re_tau
def body_force(**NS_namespace):
    return Constant((utau**2, 0., 0.))

# Normalize pressure or not? 
#normalize = False

def pre_solve(Vv, V, p_, **NS_namespace):    
    """Called prior to time loop"""
    uv = Function(Vv) 
    velocity_plotter = VTKPlotter(uv)
    pressure_plotter = VTKPlotter(p_) 
    tol = 1e-8
    voluviz = StructuredGrid(V, [Nx, Ny, Nz], [tol, -Ly/2.+tol, -Lz/2.+tol], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [Lx-2*tol, Ly-2*tol, Lz-2*tol], statistics=False)
    stats = ChannelGrid(V, [Nx/5, Ny, Nz/5], [tol, -Ly/2.+tol, -Lz/2.+tol], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [Lx-2*tol, Ly-2*tol, Lz-2*tol], statistics=True)
    return dict(uv=uv, velocity_plotter=velocity_plotter, pressure_plotter=pressure_plotter,
                voluviz=voluviz, stats=stats)
    
def inlet(x, on_bnd):
    return on_bnd and near(x[0], 0)

def walls(x, on_bnd):
    return on_bnd and (near(x[1], -Ly/2.) or near(x[1], Ly/2.))

def create_bcs(V, q_, q_1, q_2, sys_comp, u_components, **NS_namespace):    
    bcs = dict((ui, []) for ui in sys_comp)    
    Inlet = AutoSubDomain(inlet)
    facets = FacetFunction('size_t', mesh)
    facets.set_all(0)
    Inlet.mark(facets, 1)    
    bc = [DirichletBC(V, Constant(0), walls)]
    bcs['u0'] = bc
    bcs['u1'] = bc
    bcs['u2'] = bc
    bcs['p'] = []    
    return bcs

class RandomStreamVector(Expression):
    def __init__(self):
        random.seed(2 + MPI.process_number())
    def eval(self, values, x):
        values[0] = 0.0005*random.random()
        values[1] = 0.0005*random.random()
        values[2] = 0.0005*random.random()
    def value_shape(self):
        return (3,)  

def initialize(V, Vv, q_, q_1, q_2, bcs, u_components, **NS_namespace):
    if restart_folder is None:
        psi = interpolate(RandomStreamVector(), Vv)
        u0 = project(curl(psi), Vv)
        u0x = project(u0[0], V)
        u1x = project(u0[1], V)
        u2x = project(u0[2], V)
        y = interpolate(Expression("x[1] > 0 ? 1-x[1] : 1+x[1]"), V)
        uu = project(1.25*(utau/0.41*ln(conditional(y<1e-12, 1.e-12, y)*utau/nu)+5.*utau), V)
        q_['u0'].vector()[:] = uu.vector()[:] 
        q_['u0'].vector().axpy(1.0, u0x.vector())
        q_['u1'].vector()[:] = u1x.vector()[:]
        q_['u2'].vector()[:] = u2x.vector()[:]
        q_1['u0'].vector()[:] = q_['u0'].vector()[:]
        q_2['u0'].vector()[:] = q_['u0'].vector()[:]
        q_1['u1'].vector()[:] = q_['u1'].vector()[:]
        q_2['u1'].vector()[:] = q_['u1'].vector()[:]
        q_1['u2'].vector()[:] = q_['u2'].vector()[:]
        q_2['u2'].vector()[:] = q_['u2'].vector()[:]
        for ui in u_components:
            [bc.apply(q_[ui].vector()) for bc in bcs[ui]]
            [bc.apply(q_1[ui].vector()) for bc in bcs[ui]]
            [bc.apply(q_2[ui].vector()) for bc in bcs[ui]]
    
def pre_velocity_tentative_solve(ui, use_krylov_solvers, u_sol, 
                                 **NS_namespace):
    if use_krylov_solvers:
        if ui == "u0":
            u_sol.parameters['preconditioner']['reuse'] = False
            u_sol.parameters['relative_tolerance'] = 1e-9
            u_sol.parameters['absolute_tolerance'] = 1e-9
        else:
            u_sol.parameters['preconditioner']['reuse'] = True
            u_sol.parameters['relative_tolerance'] = 1e-8
            u_sol.parameters['absolute_tolerance'] = 1e-8

def update_end_of_timestep(q_, u_, V, Vv, tstep, uv, voluviz, stats, 
                           statsfolder, h5folder, pressure_plotter, 
                           velocity_plotter, update_statistics, check_save_h5, **NS_namespace):
    if tstep % update_statistics == 0:
        stats(q_['u0'], q_['u1'], q_['u2'])
        
    if tstep % check_save_h5 == 0:
        stats.toh5(0, tstep, filename=statsfolder+"/dump_mean_{}.h5".format(tstep))
        voluviz(q_['u0'])
        voluviz.toh5(0, tstep, filename=h5folder+"/snapshot_u0_{}.h5".format(tstep))
        voluviz.probes.clear()
        voluviz(q_['u1'])
        voluviz.toh5(0, tstep, filename=h5folder+"/snapshot_u1_{}.h5".format(tstep))
        voluviz.probes.clear()
        voluviz(q_['u2'])
        voluviz.toh5(0, tstep, filename=h5folder+"/snapshot_u2_{}.h5".format(tstep))
        voluviz.probes.clear()
        enstrophy = project(0.5*dot(curl(u_), curl(u_)), V)
        voluviz(enstrophy)
        voluviz.toh5(0, tstep, filename=h5folder+"/snapshot_enstrophy_{}.h5".format(tstep))
        voluviz.probes.clear()
        enstrophy = project(0.5*QC(u_), V)
        voluviz(enstrophy)
        voluviz.toh5(0, tstep, filename=h5folder+"/snapshot_Q_{}.h5".format(tstep))
        voluviz.probes.clear()
        
        uv.assign(project(u_, Vv))
        pressure_plotter.plot()
        velocity_plotter.plot()
    