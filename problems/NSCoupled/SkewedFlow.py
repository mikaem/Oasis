__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from problems import *
from numpy import cos, pi, cosh
from os import getcwd
import cPickle
from fenicstools import interpolate_nonmatching_mesh

# Create a mesh
h = 0.5
L = 1.
def mesh(N, **params):
    m = BoxMesh(0, 0, 0, L, 1, 1, N, N, N)
    return m

# Override some problem specific parameters
NS_parameters.update(
    nu = 0.1,
    omega = 1.0,
    N = 30,
    velocity_degree = 2,
    plot_interval = 10,
    max_iter = 100)
NS_parameters['krylov_solvers']['monitor_convergence'] = False

globals().update(NS_parameters)

# Specify boundary conditions
def inlet(x, on_bnd):
    return x[0] < 1e-8 and x[1] < h+1e-8 and x[2] > (1-h-1e-8)

def outlet(x, on_bnd):
    return x[0] > L-1e-8 and x[1] > 1-h-1e-8 and x[2] < h+1e-8

def walls(x, on_bnd):
    return (abs(x[1]*(1-x[1])*x[2]*(1-x[2])) < 1e-8) or (
                          (x[0] < 1e-8 and (x[1] > h-1e-8 or x[2] < (1-h+1e-8))) or
                          (x[0] > L-1e-8 and (x[1] < 1-h+1e-8 or x[2] > h-1e-8)))

def create_bcs(V, VQ, mesh, **NS_namespace):
    bmesh = BoundaryMesh(mesh, 'exterior')
    cc = CellFunction('size_t', bmesh, 0)
    ii = AutoSubDomain(inlet)
    ii.mark(cc, 1)
    smesh = SubMesh(bmesh, cc, 1)
    Vu = FunctionSpace(smesh, 'CG', 1)
    su = Function(Vu)
    us = TrialFunction(Vu)
    vs = TestFunction(Vu)
    solve(inner(grad(us), grad(vs))*dx == Constant(10.0)*vs*dx, su, 
          bcs=[DirichletBC(Vu, Constant(0), DomainBoundary())])
        
    class MyExp(Expression):
        def eval(self, values, x):
            try:
                values[0] = su(x)
                values[1] = 0
                values[2] = 0
            except:
                values[:] = 0
                
        def value_shape(self):
            return (3,)
        
    bc0 = DirichletBC(VQ.sub(0), (0,0,0), walls)
    bc1 = DirichletBC(VQ.sub(0), MyExp(), inlet)
    return dict(up = [bc0, bc1])
                
def initialize(**NS_namespace):
    pass

def pre_solve_hook(mesh, V, Q, **NS_namespace):
    n = FacetNormal(mesh)
    return dict(uv=Function(V), pp=Function(Q), n=n)

def temporal_hook(iter, up_, uv, pp, plot_interval, **NS_namespace):
    if iter % plot_interval == 0:
        assign(uv, up_.sub(0))
        assign(pp, up_.sub(1))
        plot(uv, title='Velocity')
        plot(pp, title='Pressure')

def theend_hook(up_, pp, uv, V, **NS_namespace):
    assign(uv, up_.sub(0))
    assign(pp, up_.sub(1))
    plot(uv, title='Velocity')
    plot(pp, title='Pressure')

def NS_hook(**NS_namespace):
    pass