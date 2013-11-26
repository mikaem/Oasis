__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-09-19"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from problems import *
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from numpy import zeros, floor

# Override some problem specific parameters
recursive_update(NS_parameters, 
    dict(nu = 0.00345,
         T = 1002.,
         dt = 1.0,
         folder = "aneurysm_results",
         max_iter = 1,
         plot_interval = 10,
         save_step = 10,
         checkpoint = 10,
         use_krylov_solvers = True,
         velocity_update_type = "lumping",
         velocity_degree = 1,
         pressure_degree = 1,
         krylov_solvers = dict(monitor_convergence=False))
)

mesh = Mesh("/home/mikael/Fenics/master/dolfin/demo/undocumented/bcs/aneurysm.xml.gz")
n = FacetNormal(mesh)
facet_domains = MeshFunction("size_t", mesh, 2, mesh.domains())
ds = ds[facet_domains]

# 0 = Walls
# 1 = Inlet (fix velocity)
# 2, 3 = Outlets (fix pressure)

# These inlet velocities were found in an old file    
MCAtime = array([    0.,    27.,    42.,    58.,    69.,    88.,   110.,   130.,                                                                    
        136.,   168.,   201.,   254.,   274.,   290.,   312.,   325.,                                                                                      
        347.,   365.,   402.,   425.,   440.,   491.,   546.,   618.,                                                                                      
        703.,   758.,   828.,   897.,  1002.])
    
MCAval = array([ 390.        ,  398.76132931,  512.65861027,  642.32628399,                                                        
        710.66465257,  770.24169184,  779.00302115,  817.55287009,                                                                                          
        877.12990937,  941.96374622,  970.        ,  961.2386707 ,                                                                                          
        910.42296073,  870.12084592,  843.83685801,  794.7734139 ,                                                                                          
        694.89425982,  714.16918429,  682.62839879,  644.07854985,                                                                                          
        647.58308157,  589.75830816,  559.96978852,  516.16314199,                                                                                          
        486.37462236,  474.10876133,  456.58610272,  432.05438066,  390.]
        )*0.001*2./3.

inflow_t_spline = ius(MCAtime, MCAval)

inlet_velocity0 = Constant(0)
inlet_velocity1 = Constant(0)
inlet_velocity2 = Constant(0)
p_out2 = Constant(0)
p_out3 = Constant(0)
def create_bcs(q_, V, **NS_namespace):
    bc0 = DirichletBC(V, Constant(0), 0)
    bc10 = DirichletBC(V, inlet_velocity0, 1)
    bc11 = DirichletBC(V, inlet_velocity1, 1)
    bc12 = DirichletBC(V, inlet_velocity2, 1)
    bc2 = DirichletBC(V, p_out2, 2)
    bc3 = DirichletBC(V, p_out3, 3)
    return {'u0': [bc10, bc0], 'u1': [bc11, bc0], 'u2': [bc12, bc0],
             'p': [bc2, bc3]}

def pre_solve_hook(v, mesh, n, Vv, **NS_namespace):
    """Called prior to entering timeloop."""
    A1 = assemble(Constant(1.)*ds(1), mesh=mesh)
    normal = [assemble(-n[i]*ds(1), mesh=mesh) for i in range(3)]
    A2 = []
    A3 = []
    for i in range(3):
        A2.append(assemble(v*n[i]*ds(2)))
        A3.append(assemble(v*n[i]*ds(3)))
    uv = Function(Vv)
    return dict(A1=A1, A2=A2, A3=A3, normal=normal, uv=uv)

def start_timestep_hook(u_, t, A1, A2, A3, n, normal, tstep, plot_interval, **NS_namespace):
    """Called at start of new timestep."""
    tt = t - floor(t/1002.0)*1002.0
    u_mean = inflow_t_spline(tt)/A1   
    inlet_velocity0.assign(u_mean*normal[0])
    inlet_velocity1.assign(u_mean*normal[1])
    inlet_velocity2.assign(u_mean*normal[2])
        
    p2 = 0
    p3 = 0
    for i in range(3):
        p2 += A2[i].inner(u_[i].vector())
        p3 += A3[i].inner(u_[i].vector())
    p_out2.assign(p2)
    p_out3.assign(p3)
    
    if tstep % plot_interval == 0:
        info_green('UMEAN = {0:2.5f} at time {1:2.5f}'.format(u_mean, t))
        info_green('Pressure outlet 2 = {0:2.5f}'.format(p_out2(0)))
        info_green('Pressure outlet 3 = {0:2.5f}'.format(p_out3(0)))

def initialize(q_, q_1, q_2, VV, t, nu, dt, **NS_namespace):
    """Initialize solution."""
    for ui in q_:
        q_[ui].vector()[:] = 1e-12 # Just to help Krylov solver on first iteration
        
def temporal_hook(p_, u_, uv, Vv, plot_interval, tstep, **NS_namespace):
    """Function called at end of timestep."""
    if tstep % plot_interval == 0:
        uv.assign(project(u_, Vv))
        plot(uv, title='u')
        plot(p_, title='p')
