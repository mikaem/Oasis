__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-03-21"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..NSfracStep import *
from os import getcwd
import cPickle

#mesh = Mesh("/home/mikael/MySoftware/Oasis/mesh/cyl_dense.xml")
mesh = Mesh("/home/mikael/MySoftware/Oasis/mesh/cyl_dense2.xml")

H = 0.41
L = 2.2
D = 0.1
center = 0.2
case = {
      1: {'Um': 0.3,
          'Re': 20.0},
      
      2: {'Um': 1.5,
          'Re': 100.0}
      }

# Specify boundary conditions
Inlet = AutoSubDomain(lambda x, on_bnd: on_bnd and x[0] < 1e-8)
Wall = AutoSubDomain(lambda x, on_bnd: on_bnd and near(x[1]*(H-x[1]), 0))
Cyl = AutoSubDomain(lambda x, on_bnd: on_bnd and x[0]>1e-6 and x[0]<1 and x[1] < 3*H/4 and x[1] > H/4)
Outlet = AutoSubDomain(lambda x, on_bnd: on_bnd and x[0] > L-1e-8)

#restart_folder = "results/data/8/Checkpoint"
restart_folder = None

if restart_folder:
    restart_folder = path.join(getcwd(), restart_folder)
    f = open(path.join(restart_folder, 'params.dat'), 'r')
    NS_parameters.update(cPickle.load(f))
    NS_parameters['restart_folder'] = restart_folder
    globals().update(NS_parameters)
    
else:

    # Override some problem specific parameters
    NS_parameters.update(
        T  = 5.0,
        dt = 0.05,
        checkpoint = 1000,
        save_step = 5000, 
        plot_interval = 10,
        velocity_degree = 2,
        print_intermediate_info = 100,
        velocity_update_type = 'gradient_matrix',
        use_krylov_solvers = True)

scalar_components = ["alfa"]
Schmidt["alfa"] = 0.1

def post_import_problem(NS_parameters, **NS_namespace):
    """ Choose case - check if case is defined through command line."""
    c = 1 # default is case 1
    if "case" in NS_parameters:        
        if NS_parameters["case"] in [1, 2]:
            c = NS_parameters["case"]
    Um = case[c]["Um"]
    Re = case[c]["Re"]
    Umean = 2./3.* Um
    nu = Umean*D/Re
    NS_parameters.update(nu=nu, Re=Re, Um=Um, Umean=Umean)
    return NS_parameters

def create_bcs(V, Q, Um, **NS_namespace):
    inlet = Expression("4.*{0}*x[1]*({1}-x[1])/pow({1}, 2)".format(Um, H))
    ux = Expression("0.00*x[1]")
    uy = Expression("-0.00*(x[0]-{})".format(center))
    bc00  = DirichletBC(V, inlet, Inlet)
    bc01  = DirichletBC(V, 0, Inlet)    
    bc10 = DirichletBC(V, ux, Cyl)
    bc11 = DirichletBC(V, uy, Cyl)
    bc2 = DirichletBC(V, 0, Wall)
    bcp = DirichletBC(Q, 0, Outlet)
    bca = DirichletBC(V, 1, Cyl)
    return dict(u0 = [bc00, bc10, bc2],
                u1 = [bc01, bc11, bc2],
                p  = [bcp],
                alfa = [bca])

def initialize(x_1, x_2, bcs, **NS_namespace):
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:    
        [bc.apply(x_2[ui]) for bc in bcs[ui]]

def pre_solve_hook(mesh, velocity_degree, constrained_domain, V, 
                   newfolder, tstepfiles, tstep, **NS_namespace):
    Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree, constrained_domain=constrained_domain)
    omega = Function(V, name='omega')
    # Store omega each save_step
    add_function_to_tstepfiles(omega, newfolder, tstepfiles, tstep)
    return dict(Vv=Vv, uv=Function(Vv), omega=omega)

def temporal_hook(q_, tstep, u_, Vv, V, uv, p_, plot_interval, omega, 
                  save_step, **NS_namespace):
    if tstep % plot_interval == 0:
        uv.assign(project(u_, Vv))
        plot(uv, title='Velocity')
        plot(p_, title='Pressure')
        plot(q_['alfa'], title='alfa')
        
    if tstep % save_step == 0:
        try:
            from fenicstools import StreamFunction
            omega.assign(StreamFunction(u_, []))
        except:
            omega.assign(project(curl(u_), V, 
                         bcs=[DirichletBC(V, 0, DomainBoundary())]))

def theend_hook(q_, u_, p_, uv, Vv, mesh, ds, V, **NS_namespace):
    uv.assign(project(u_, Vv))
    plot(uv, title='Velocity')
    plot(p_, title='Pressure')
    plot(q_['alfa'], title='alfa')
    R = VectorFunctionSpace(mesh, 'R', 0)
    c = TestFunction(R)
    tau = -p_*Identity(2)+nu*(grad(u_)+grad(u_).T)
    ff = FacetFunction("size_t", mesh, 0)
    Cyl.mark(ff, 1)
    n = FacetNormal(mesh)
    ds = ds[ff]
    forces = assemble(dot(dot(tau, n), c)*ds(1)).array()*2/Umean**2/D
    
    print "Cd = {}, CL = {}".format(*forces)

    from fenicstools import Probes
    from numpy import linspace, repeat, where, resize
    xx = linspace(0, L, 10000)
    x = resize(repeat(xx, 2), (10000, 2))
    x[:, 1] = 0.2
    probes = Probes(x.flatten(), V)
    probes(u_[0])
    nmax = where(probes.array() < 0)[0][-1]
    print "L = ", x[nmax, 0]-0.25
    print "dP = ", p_(Point(0.15, 0.2)) - p_(Point(0.25, 0.2))
