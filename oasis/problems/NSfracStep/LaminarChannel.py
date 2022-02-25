from __future__ import print_function

__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"


from oasis.problems import (
    constrained_domain,
    scalar_components,
    Schmidt,
    Schmidt_T,
    body_force,
    initialize,
    scalar_hook,
    scalar_source,
    pre_solve_hook,
    theend_hook,
    get_problem_parameters,
    post_import_problem,
    create_bcs,
)
from oasis.problems.NSfracStep import (
    velocity_tentative_hook,
    pressure_hook,
    start_timestep_hook,
    temporal_hook,
    default_parameters,
)

# from oasis.problems.LaminarChannel import mesh
import dolfin as df
from numpy import pi, arctan, array, exp

# set_log_active(False)


def get_problem_parameters(**kwargs):
    nu = 0.01
    Re = 1.0 / nu
    L = 10.0
    NS_parameters = dict(
        scalar_components=scalar_components,
        Schmidt=Schmidt,
        Schmidt_T=Schmidt_T,
        nu=nu,
        L=L,
        H=1.0,
        T=10,
        dt=0.01,
        Re=Re,
        Nx=40,
        Ny=40,
        folder="laminarchannel_results",
        max_iter=1,
        velocity_degree=1,
        use_krylov_solvers=False,
    )
    # set default parameters
    for key, val in default_parameters.items():
        if key not in NS_parameters.keys():
            NS_parameters[key] = val

    NS_expressions = dict(constrained_domain=PeriodicDomain(L))
    return NS_parameters, NS_expressions


# Create a mesh here
def mesh(Nx, Ny, L, H, **params):
    m = df.RectangleMesh(df.Point(0.0, -H), df.Point(L, H), Nx, Ny)

    # Squeeze towards walls
    x = m.coordinates()
    x[:, 1] = arctan(1.0 * pi * (x[:, 1])) / arctan(1.0 * pi)
    return m


class PeriodicDomain(df.SubDomain):
    def __init__(self, L):
        self.L = L
        df.SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(df.near(x[0], 0) and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - self.L
        y[1] = x[1]


def create_bcs(V, H, sys_comp, **NS_namespace):
    def walls(x, on_boundary):
        return on_boundary and (df.near(x[1], -H) or df.near(x[1], H))

    bcs = dict((ui, []) for ui in sys_comp)
    bc0 = df.DirichletBC(V, 0.0, walls)
    bcs["u0"] = [bc0]
    bcs["u1"] = [bc0]
    return bcs


def body_force(Re, **NS_namespace):
    return df.Constant((2.0 / Re, 0.0))


def reference(Re, t, num_terms=100):
    u = 1.0
    c = 1.0
    for n in range(1, 2 * num_terms, 2):
        a = 32.0 / (pi ** 3 * n ** 3)
        b = (0.25 / Re) * pi ** 2 * n ** 2
        c = -c
        u += a * exp(-b * t) * c
    return u


def temporal_hook(tstep, q_, t, Re, L, **NS_namespace):
    if tstep % 20 == 0:
        df.plot(q_["u0"])
    try:
        # point is found on one processor, the others pass
        u_computed = q_["u0"](array([L, 0.0]))
        u_exact = reference(Re, t)
        print("Error = ", (u_exact - u_computed) / u_exact)
    except:
        pass
