__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from Oasis import *
from numpy import pi, arctan, array

# Create a mesh here
L = 10.
H = 1.
mesh = RectangleMesh(0., -H, L, H, 40, 40)
# Squeeze towards walls
x = mesh.coordinates()        
x[:, 1] = arctan(1.*pi*(x[:, 1]))/arctan(1.*pi) 
del x

class PeriodicDomain(SubDomain):

    def inside(self, x, on_boundary):
        return bool(near(x[0], 0) and on_boundary)
                      
    def map(self, x, y):
        y[0] = x[0] - L
        y[1] = x[1] 
            
constrained_domain = PeriodicDomain()

# Override some problem specific parameters
T = 10
dt = 0.01
nu = 0.01
Re = 1./nu
folder = "laminarchannel_results"
newfolder = create_initial_folders(folder, dt)
NS_parameters.update(dict(
    nu = nu,
    T = T,
    dt = dt,
    Re = Re,
    folder = folder,
    max_iter = 1,
    newfolder = newfolder,
    velocity_degree = 1,
    use_lumping_of_mass_matrix = True,
    use_krylov_solvers = True
  )
)

def walls(x, on_boundary):
    return (on_boundary and (near(x[1], -H) or near(x[1], H)))
    
def create_bcs(V, sys_comp, **NS_namespace):
    bcs = dict((ui, []) for ui in sys_comp)    
    bc0  = DirichletBC(V, 0., walls)
    bcs['u0'] = [bc0]
    bcs['u1'] = [bc0]
    return bcs

def body_force(Re, **NS_namespace):
    return Constant((2./Re, 0.))

def reference(Re, t, num_terms=100):
    u = 1.0
    c = 1.0
    for n in range(1, 2*num_terms, 2):
        a = 32. / (pi**3*n**3)
        b = (0.25/Re)*pi**2*n**2
        c = -c
        u += a*exp(-b*t)*c
    return u

def temporal_hook(tstep, q_, t, Re, **NS_namespace):
    if tstep % 20 == 0:        
        plot(q_['u0'])
    u_exact = reference(Re, t)
    u_computed = q_['u0'](array([1.0, 0.]))
    print "Error = ", (u_exact-u_computed)/u_exact
