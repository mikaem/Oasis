__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from Oasis import *

# Create a mesh here
class Submesh(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 0.25 - DOLFIN_EPS and x[1] > 0.25 - DOLFIN_EPS

mesh_ = UnitSquareMesh(40, 40)
subm = Submesh()
mf1 = MeshFunction("size_t", mesh_, 2)
mf1.set_all(0)
subm.mark(mf1, 1)
mesh = SubMesh(mesh_, mf1, 0)
del mesh_, mf1, subm

# Override some problem specific parameters
T = 10
dt = 0.1
Re = 200.
nu = 1./Re
folder = "Lshape_results"
newfolder = create_initial_folders(folder, dt)
NS_parameters.update(dict(
    nu = nu,
    T = T,
    dt = dt,
    Re = Re,
    folder = folder,
    max_iter = 1,
    plot_interval = 1,
    newfolder = newfolder,
    velocity_degree = 2,
    use_lumping_of_mass_matrix = True,
    use_krylov_solvers = True
  )
)

def inlet(x, on_boundary):
    return near(x[1] - 1., 0.) and on_boundary

def outlet(x, on_boundary):
    return near(x[0] - 1., 0.) and on_boundary

def walls(x, on_boundary):
    return (near(x[0], 0.) or near(x[1], 0.) or 
            (x[0] > 0.25 - 5*DOLFIN_EPS  and 
             x[1] > 0.25 - 5*DOLFIN_EPS) and on_boundary)

p_in = Expression("sin(pi*t)", t=0.)
def create_bcs(V, Q, sys_comp, **NS_namespace):
    bcs = dict((ui, []) for ui in sys_comp)    
    bc0 = DirichletBC(V, 0., walls)
    pc0 = DirichletBC(Q, p_in, inlet)
    pc1 = DirichletBC(Q, 0.0, outlet)
    bcs['u0'] = [bc0]
    bcs['u1'] = [bc0]
    bcs['p'] = [pc0, pc1]
    return bcs

def pre_solve_hook(Vv, **NS_namespace):
    uv = Function(Vv)
    return dict(uv=uv)

def start_timestep_hook(t, **NS_namespace):
    p_in.t = t
    
def temporal_hook(tstep, q_, u_, uv, Vv, plot_interval, **NS_namespace):
    if tstep % plot_interval == 0:
        plot(q_['p'])
        uv.assign(project(u_, Vv))
        plot(uv)
