__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from problems import *
from numpy import cos, pi, cosh

# Create a mesh
def mesh(Nx, Ny, Nz, a, b, L, **params):
    m = BoxMesh(0, -a, -b, L, a, b, Nx, Ny, Nz)
    x = m.coordinates()
    x[:, 1] = x[:, 1]*(1-x[:, 0]/16.)
    x[:, 2] = x[:, 2]*(1-x[:, 0]/16.)
    return m

# Override some problem specific parameters
NS_parameters.update(
    nu = 0.01,
    T  = 1.0,
    dt = 0.01,
    Nx = 25,
    Ny = 25,
    Nz = 25,
    dpdx = -0.01,
    a = 2,
    b = 1,
    L = 4.,
    velocity_degree = 1,
    plot_interval = 1,
    print_intermediate_info = 10,
    use_krylov_solvers = True)
NS_parameters['krylov_solvers']['monitor_convergence'] = True

globals().update(NS_parameters)

# Much faster C++ version
ue_code = '''
class U : public Expression
{
  public:

    double a, b, mu, dpdx;

  void eval(Array<double>& values, const Array<double>& x) const
    {
      double u = 0.;
      double factor = 16.*a*a/mu/pow(DOLFIN_PI, 3)*(-dpdx);
      for (std::size_t i=1; i<600; i=i+2)
        u += pow(-1, (i-1)/2 % 2)*(1.-cosh(i*DOLFIN_PI*x[2]/2./a)/
           cosh(i*DOLFIN_PI*b/2./a))*cos(i*DOLFIN_PI*x[1]/2./a)/pow(i, 3);
      values[0] = u*factor;      
    }
};'''
u_c = Expression(ue_code)
u_c.a = float(a); u_c.b = float(b)
u_c.mu = float(nu); u_c.dpdx = float(dpdx)

def body_force(mesh, **NS_namespace):
    """Specify body force"""
    return Constant((0.01, 0, 0))

# Specify boundary conditions
walls = "on_boundary &! std::abs(x[0]*(4-x[0]))<1e-8"
inlet    = "x[0] < 1e-8 && on_boundary"
outlet   = "x[0] > 4.0-1e-8 && on_boundary"
def create_bcs(V, Q, **NS_namespace):
    bc0  = DirichletBC(V, 0, walls)
    bc1  = DirichletBC(V, u_c, inlet)
    bc2  = DirichletBC(V, 0, inlet)
    bcp1 = DirichletBC(Q, 0, outlet)
    return dict(u0 = [bc0],
                u1 = [bc0],
                u2 = [bc0],
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

def theend_hook(u_, p_, uv, Vv, **NS_namespace):
    uv.assign(project(u_, Vv))
    plot(uv, title='Velocity')
    plot(p_, title='Pressure')
