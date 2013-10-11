from Channel import *
from NSVector_default_hooks import *

def create_bcs(Vv, q_, q_1, q_2, sys_comp, u_components, **NS_namespace):
    bcs = dict((ui, []) for ui in sys_comp)    
    bcs['u'] = [DirichletBC(Vv, Constant((0, 0, 0)), walls)]
    return bcs

class RandomStreamVector(Expression):
    def __init__(self):
        random.seed(2 + MPI.process_number())
    def eval(self, values, x):
        values[0] = 0.0005*random.random()
        values[1] = 0.0005*random.random()
        values[2] = 0.0005*random.random()
    def value_shape(self):
        return (3,)  

class U0(Expression):
    def __init__(self, u):
        self.u = u
        
    def eval(self, values, x):
        values[0] = self.u(x)
        
    def value_shape(self):
        return (3,)

c_code = '''
class U0 : public Expression
{
  public:

    double utau, nu;

    U0() : Expression() {_value_shape.resize(1); 
                         _value_shape[0]=3;}

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
    {      
      double y = x[1] > 0 ? 1.-x[1] : 1.+x[1];   
      y = y < 1e-12 ? 1e-12 : y;
      values[0] = 1.01*(utau/0.41*std::log(y*utau/nu)+5.*utau);
      values[1] = 0;
      values[2] = 0;
    }
    
};'''

U0 = Expression(c_code)
        
def initialize(V, Vv, q_, q_1, q_2, bcs, restart_folder, mesh, **NS_namespace):
    if restart_folder is None:
        psi = interpolate(RandomStreamVector(), Vv)
        u0 = project(curl(psi), Vv)
        U0.utau, U0.nu = utau, nu
        u1 = interpolate(U0, Vv)
        q_['u'].vector()[:] = u1.vector()[:]
        q_['u'].vector().axpy(1.0, u0.vector())
        q_1['u'].vector()[:] = q_['u'].vector()[:]
        q_2['u'].vector()[:] = q_['u'].vector()[:]
    
def velocity_tentative_hook(**NS_namespace):
    pass

def pre_solve_hook(Vv, V, Nx, Ny, Nz, mesh, **NS_namespace):    
    """Called prior to time loop"""
    uv = Function(Vv) 
    tol = 1e-8
    voluviz = StructuredGrid(Vv, [Nx, Ny+1, Nz], [tol, -Ly/2., -Lz/2.+tol], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [Lx-Lx/Nx, Ly, Lz-Lz/Nz], statistics=False)
    stats = ChannelGrid(Vv, [Nx/5, Ny+1, Nz/5], [tol, -Ly/2., -Lz/2.+tol], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [Lx-Lx/Nx*5, Ly, Lz-Lz/Nz*5], statistics=True)
    
    Inlet = AutoSubDomain(inlet)
    facets = FacetFunction('size_t', mesh)
    facets.set_all(0)
    Inlet.mark(facets, 1)    
    normal = FacetNormal(mesh)

    return dict(uv=uv, voluviz=voluviz, stats=stats, facets=facets, normal=normal)

def temporal_hook(q_, u_, V, Vv, tstep, uv, voluviz, stats, update_statistics,
                  check_save_h5, newfolder, check_flux,
                  facets, normal, **NS_namespace):
    if tstep % update_statistics == 0:
        stats(u_)
        
    if tstep % check_save_h5 == 0:
        statsfolder = path.join(newfolder, "Stats")
        h5folder = path.join(newfolder, "Voluviz")
        stats.toh5(0, tstep, filename=statsfolder+"/dump_mean_{}.h5".format(tstep))
        voluviz(u_)
        voluviz.toh5(0, tstep, filename=h5folder+"/snapshot_u_{}.h5".format(tstep))
        voluviz.probes.clear()
        
    if tstep % check_flux == 0:
        u1 = assemble(dot(u_, normal)*ds(1), exterior_facet_domains=facets)
        normv = norm(u_.vector())
        if MPI.process_number() == 0:
            print "Flux = ", u1, " tstep = ", tstep, " norm = ", normv
