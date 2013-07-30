__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from Oasis import *

# Create a mesh here
mesh = BoxMesh(-pi, -pi, -pi, pi, pi, pi, 32, 32, 32)

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
NS_parameters.update(dict(
    nu = 0.005,
    T = 4.,
    dt = 0.01,
    folder = "taylorgreen3D_results",
    max_iter = 1,
    velocity_degree = 1,
    use_krylov_solvers = True,
    use_lumping_of_mass_matrix = True,
  )
)

initial_fields = dict(
        u0='sin(x[0])*cos(x[1])*cos(x[2])',
        u1='-cos(x[0])*sin(x[1])*cos(x[2])',
        u2='0',
        p='0')
    
def initialize(q_, q_1, q_2, VV, sys_comp, **NS_namespace):
    for ui in sys_comp:
        vv = project(Expression((initial_fields[ui])), VV[ui])
        q_[ui].vector()[:] = vv.vector()[:]
        if not ui == 'p':
            q_1[ui].vector()[:] = q_[ui].vector()[:]
            q_2[ui].vector()[:] = q_[ui].vector()[:]

def temporal_hook(q_, p_, tstep, **NS_namespace):
    if tstep % 10 == 0:
        plot(p_)
        plot(q_['u0'])
        plot(q_['u1'])
