__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from problems import *
from numpy import cos, pi

# Create a mesh
def mesh(Nx, Ny, **params):
    m = UnitSquareMesh(Nx, Ny)
    x = m.coordinates()
    x[:] = (x - 0.5) * 2
    x[:] = 0.5*(cos(pi*(x-1.) / 2.) + 1.)
    return m

# Override some problem specific parameters
NS_parameters.update(
  nu = 0.001,
  T  = 1.0,
  dt = 0.001,
  Nx = 50,
  Ny = 50,
  plot_interval = 20,
  print_intermediate_info = 100,
  use_krylov_solvers = True)
            
# Specify boundary conditions
noslip = "std::abs(x[0]*x[1]*(1-x[0]))<1e-8"
top    = "std::abs(x[1]-1) < 1e-8"
def create_bcs(V, **NS_namespace):
    bc0  = DirichletBC(V, 0, noslip)
    bc00 = DirichletBC(V, 1, top)
    bc01 = DirichletBC(V, 0, top)
    return dict(u0 = [bc00, bc0],
                u1 = [bc01, bc0],
                p  = [])
                
def initialize(x_1, x_2, bcs, **NS_namespace):
    for ui in x_2:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
        [bc.apply(x_2[ui]) for bc in bcs[ui]]

def pre_solve_hook(mesh, velocity_degree, constrained_domain, **NS_namespace):
    Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree, constrained_domain=constrained_domain)
    return dict(Vv=Vv, uv=Function(Vv))

def temporal_hook(tstep, u_, Vv, uv, p_, plot_interval, **NS_namespace):
    if tstep % plot_interval == 0:
        uv.assign(project(u_, Vv))
        plot(uv, title='Velocity')
        plot(p_, title='Pressure')

def theend(u_, p_, uv, Vv, **NS_namespace):
    uv.assign(project(u_, Vv))
    plot(uv, title='Velocity')
    plot(p_, title='Pressure')

    try:
        from cbc.cfd.tools.Streamfunctions import StreamFunction
        psi = StreamFunction(u_, [], use_strong_bc=True)
        plot(psi, title='Streamfunction')
        interactive()
    except:
        pass
