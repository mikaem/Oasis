from dolfin import Mesh, AutoSubDomain, near

#mesh = Mesh("/home/mikael/MySoftware/Oasis/mesh/cyl_dense.xml")
mesh = Mesh("/home/mikael/MySoftware/Oasis/mesh/cyl_dense2.xml")

H = 0.41
L = 2.2
D = 0.1
center = 0.2
cases = {
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

# Overload post_import_problem to choose between the two cases
def post_import_problem(NS_parameters, commandline_kwargs, **NS_namespace):
    """ Choose case - case could be defined through command line."""
    NS_parameters.update(commandline_kwargs)
    case = NS_parameters['case'] if 'case' in NS_parameters else 1
    Um = cases[case]["Um"]
    Re = cases[case]["Re"]
    Umean = 2./3.* Um
    nu = Umean*D/Re
    NS_parameters.update(nu=nu, Re=Re, Um=Um, Umean=Umean)

    return NS_parameters