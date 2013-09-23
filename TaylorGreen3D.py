__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from Oasis import *

def mesh(Nx, Ny, Nz, **params):
    return BoxMesh(-pi, -pi, -pi, pi, pi, pi, Nx, Ny, Nz)

class PeriodicDomain(SubDomain):
    
    def inside(self, x, on_boundary):
        return bool((near(x[0], -pi) or near(x[1], -pi) or near(x[2], -pi)) and 
                (not (near(x[0], pi) or near(x[1], pi) or near(x[2], pi))) and on_boundary)

    def map(self, x, y):
        if near(x[0], pi) and near(x[1], pi) and near(x[2], pi):
            y[0] = x[0] - 2.0*pi
            y[1] = x[1] - 2.0*pi
            y[2] = x[2] - 2.0*pi
        elif near(x[0], pi) and near(x[1], pi):
            y[0] = x[0] - 2.0*pi
            y[1] = x[1] - 2.0*pi
            y[2] = x[2]
        elif near(x[1], pi) and near(x[2], pi):
            y[0] = x[0] 
            y[1] = x[1] - 2.0*pi
            y[2] = x[2] - 2.0*pi
        elif near(x[1], pi):
            y[0] = x[0] 
            y[1] = x[1] - 2.0*pi
            y[2] = x[2]
        elif near(x[0], pi) and near(x[2], pi):
            y[0] = x[0] - 2.0*pi
            y[1] = x[1] 
            y[2] = x[2] - 2.0*pi
        elif near(x[0], pi):
            y[0] = x[0] - 2.0*pi
            y[1] = x[1] 
            y[2] = x[2]            
        else: # near(x[2], pi):
            y[0] = x[0] 
            y[1] = x[1]
            y[2] = x[2] - 2.0*pi

constrained_domain = PeriodicDomain()

# Override some problem specific parameters
recursive_update(NS_parameters, dict(
    nu = 0.01,
    T = 0.2,
    dt = 0.01,
    Nx = 20,
    Ny = 20, 
    Nz = 20,
    folder = "taylorgreen3D_results",
    max_iter = 1,
    velocity_degree = 1,
    save_step = 10000,
    checkpoint = 10000, 
    plot_interval = 100000,
    use_krylov_solvers = True,
    use_lumping_of_mass_matrix = True,
    krylov_solvers = dict(monitor_convergence=True)
  )
)

initial_fields = dict(
        u0='sin(x[0])*cos(x[1])*cos(x[2])',
        u1='-cos(x[0])*sin(x[1])*cos(x[2])',
        u2='0',
        p='0')
    
def initialize(q_, q_1, q_2, VV, **NS_namespace):
    for ui in q_:
        vv = project(Expression((initial_fields[ui])), VV[ui])
        q_[ui].vector()[:] = vv.vector()[:]
        if not ui == 'p':
            q_1[ui].vector()[:] = q_[ui].vector()[:]
            q_2[ui].vector()[:] = q_[ui].vector()[:]

def temporal_hook(q_, tstep, plot_interval, **NS_namespace):
    if tstep % plot_interval == 0:
        plot(q_['p'], title='pressure')
        plot(q_['u0'], title='velocity-x')
        plot(q_['u1'], title='velocity-y')


