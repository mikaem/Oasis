__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from Oasis import *
from numpy import array, sin

# Constants related to the geometry
xmin = 0.0
xmax = 0.4
ymin = 0.0
ymax = 0.41
xcenter = 0.2
ycenter = 0.2
radius = 0.05
Um = 1.5

circle = Circle(xcenter, ycenter, radius)
rect = Rectangle(xmin, ymin, xmax, ymax)
mesh = Mesh(rect-circle, 50)

recursive_update(NS_parameters,
   dict(nu = 0.01,
        T = 0.01,
        dt = 0.01,
        folder = "cylinder_results",
        plot_interval = 100,
        print_intermediate_info = 10,
        iters_first_timestep = 2,
        save_step = 1000,
        checkpoint = 1000,
        velocity_degree = 2,
        use_lumping_of_mass_matrix = True,
        use_krylov_solvers = True,
        krylov_solvers = dict(monitor_convergence=False))
)

# Inflow boundary
def InflowBoundary(x, on_boundary):
    return on_boundary and x[0] < xmin + DOLFIN_EPS

# No-slip boundary
def NoslipBoundary(x, on_boundary):
    return (on_boundary and (x[1] > ymax-DOLFIN_EPS or x[1] < DOLFIN_EPS 
                             or (DOLFIN_EPS < x[0] < xmax-DOLFIN_EPS and 
                                 DOLFIN_EPS < x[1] < ymax-DOLFIN_EPS)
                        ))

# Outflow boundary
def OutflowBoundary(x, on_boundary):
    return on_boundary and x[0] > xmax - DOLFIN_EPS

class Uin(Expression):
    def __init__(self, t):
        self.t = t
        
    def eval(self, values, x):
        values[0] = 4.*Um*x[1]*(ymax-x[1])*sin(pi*self.t/8.)/ymax**4
    
uin = Uin(0.0)
def create_bcs(V, Q, sys_comp, **NS_namespace):
    bcs = dict((ui, []) for ui in sys_comp)    
    bc0  = DirichletBC(V, 0., NoslipBoundary)    
    bc00 = DirichletBC(V, uin, InflowBoundary)
    bc01 = DirichletBC(V, 0., InflowBoundary)
    #bcs['p'] = [DirichletBC(Q, 0., OutflowBoundary)]
    bcs['p'] = []
    bcs['u0'] = [bc00, bc0]
    bcs['u1'] = [bc01, bc0]
    return bcs

def initialize(q_, **NS_namespace):
    q_['u0'].vector()[:] = 1e-12 # To help Krylov solver on first timestep

def start_timestep_hook(t, **NS_namespace):
    uin.t = t

mf = FacetFunction("size_t", mesh, 0)
AutoSubDomain(OutflowBoundary).mark(mf, 1)
def pre_solve_hook(Vv, p_, q, p, Ap, bcs, **NS_namespace):    
    #n = FacetNormal(mesh)    
    #Aps  = assemble(inner(q*n, grad(p))*ds(1), exterior_facet_domains=mf)
    #Ap.axpy(-1.0, Aps, True)
    #[bc.apply(Ap) for bc in bcs['p']]
    return dict(uv=Function(Vv))
    
def temporal_hook(tstep, u_, Vv, uv, p_, plot_interval, **NS_namespace):
    print "pressure drop = ", p_(xcenter+radius, ycenter) - p_(xcenter-radius, ycenter)
    if tstep % plot_interval == 0:
        uv.assign(project(u_, Vv))
        plot(uv, title='Velocity')
        plot(p_, title='Pressure')

def velocity_tentative_hook(v, p_, b, nu, q_, **NS_namespace):
    #n = FacetNormal(mesh)
    b["u0"].axpy(-1., assemble(inner(v, p_)*ds(1), exterior_facet_domains=mf))

def pressure_hook(q, p, p_, b, **NS_namespace):
    pass
    #n = FacetNormal(mesh)    
    #b['p'].axpy(-1.0, assemble(inner(q*n, grad(p_))*ds(1), exterior_facet_domains=mf))
    
    