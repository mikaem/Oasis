__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from Cylinder import *

recursive_update(NS_parameters,
   dict(nu = 0.01,
        T = 0.1,
        dt = 0.01,
        folder = "cylindervector_results",
        plot_interval = 100,
        print_intermediate_info = 10,
        iters_on_first_timestep = 2,
        save_step = 1000,
        checkpoint = 1000,
        velocity_degree = 2,
        max_iter = 1,
        use_lumping_of_mass_matrix = True,
        use_krylov_solvers = True,
        krylov_solvers = dict(monitor_convergence=False))
)

class Uin(Expression):
    def __init__(self, t):
        self.t = t
        
    def eval(self, values, x):
        values[0] = 4.*Um*x[1]*(ymax-x[1])*sin(pi*self.t/8.)/ymax**4
        values[1] = 0.
        
    def value_shape(self):
        return (2,)
    
uin = Uin(0.0)
def create_bcs(Vv, Q, sys_comp, **NS_namespace):
    bcs = dict((ui, []) for ui in sys_comp)    
    bc0  = DirichletBC(Vv, (0., 0.), NoslipBoundary)    
    bc1 = DirichletBC(Vv, uin, InflowBoundary)
    bcs['p'] = [DirichletBC(Q, 0., OutflowBoundary)]
    #bcs['p'] = []
    bcs['u'] = [bc0, bc1]
    return bcs

def initialize(q_, **NS_namespace):
    q_['u'].vector()[:] = 1e-12 # To help Krylov solver on first timestep

def start_timestep_hook(t, **NS_namespace):
    uin.t = t

mf = FacetFunction("size_t", mesh, 0)
AutoSubDomain(OutflowBoundary).mark(mf, 1)
def pre_solve_hook(Vv, nu, p_, q, v, u, p, Ap, bcs, K, **NS_namespace):    
    n = FacetNormal(mesh)    
    #Aps  = assemble(inner(q*n, grad(p))*ds(1), exterior_facet_domains=mf)
    #Aps  = assemble(inner(q*n, grad(p))*ds)
    #Ap.axpy(-1.0, Aps, True)
    #[bc.apply(Ap) for bc in bcs['p']]
    #K.axpy(-1., assemble(inner(v, grad(u).T*n)*ds(1), exterior_facet_domains=mf), True)

    return dict(uv=Function(Vv))
    
def temporal_hook(tstep, u_, Vv, uv, p_, plot_interval, **NS_namespace):
    print "pressure drop = ", p_(xcenter+radius, ycenter) - p_(xcenter-radius, ycenter)
    if tstep % plot_interval == 0:
        uv.assign(project(u_, Vv))
        plot(uv, title='Velocity')
        plot(p_, title='Pressure')

def velocity_tentative_hook(v, p_, b, nu, q_, **NS_namespace):
    pass
    #n = FacetNormal(mesh)
    #b["u"].axpy(-1., assemble(inner(v, p_)*ds(1), exterior_facet_domains=mf))

def pressure_hook(q, p, p_, b, **NS_namespace):
    pass
    #n = FacetNormal(mesh)    
    #b['p'].axpy(-1.0, assemble(inner(q*n, grad(p_))*ds(1), exterior_facet_domains=mf))
    #b['p'].axpy(-1.0, assemble(inner(q*n, grad(p_))*ds))
