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
from oasis.problems import create_bcs
from oasis.problems.NSCoupled import (
    NS_hook,
    start_iter_hook,
    end_iter_hook,
    default_parameters,
)
from oasis.problems.Nozzle2D import mesh, walls, inlet, outlet, centerline
import dolfin as df

from math import sqrt, pi
from fenicstools import StructuredGrid, StatisticsProbes
import sys
from numpy import array, linspace


def get_problem_parameters(**kwargs):
    re_high = False
    NS_parameters = dict(
        omega=0.4,
        nu=0.0035 / 1056.0,
        folder="nozzle_results",
        max_error=1e-13,
        max_iter=25,
        re_high=re_high,
        solver="cylindrical",
    )
    NS_parameters["scalar_components"] = scalar_components
    NS_parameters["Schmidt"] = Schmidt
    NS_parameters["Schmidt_T"] = Schmidt_T
    # set default parameters
    for key, val in default_parameters.items():
        if key not in NS_parameters.keys():
            NS_parameters[key] = val
    return NS_parameters


def create_bcs(VQ, mesh, sys_comp, re_high, **NS_namespce):
    # Q = 5.21E-6 if not re_high else 6.77E-5  # From FDA
    Q = 5.21e-6 if not re_high else 3e-5  # From FDA
    r_0 = 0.006
    # Analytical, could be more exact numerical, different r_0
    u_maks = Q / (4.0 * r_0 * r_0 * (1.0 - 2.0 / pi))
    # inn = Expression(("u_maks * cos(sqrt(pow(x[1],2))/r_0/2.*pi)", "0"), u_maks=u_maks, r_0=r_0)
    inn = df.Expression(
        ("u_maks * (1-x[1]*x[1]/r_0/r_0)", "0"), u_maks=u_maks, r_0=r_0, degree=2
    )

    bc0 = df.DirichletBC(VQ.sub(0), inn, inlet)
    bc1 = df.DirichletBC(VQ.sub(0), (0, 0), walls)
    bc2 = df.DirichletBC(VQ.sub(0).sub(1), 0, centerline)

    return dict(up=[bc0, bc1, bc2])


def pre_solve_hook(mesh, V, **NS_namespace):
    # Normals and facets to compute flux at inlet and outlet
    normal = df.FacetNormal(mesh)
    Inlet = df.AutoSubDomain(inlet)
    Outlet = df.AutoSubDomain(outlet)
    Walls = df.AutoSubDomain(walls)
    Centerline = df.AutoSubDomain(centerline)
    facets = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    Inlet.mark(facets, 1)
    Outlet.mark(facets, 2)
    Walls.mark(facets, 3)
    Centerline.mark(facets, 4)

    z_senterline = linspace(-0.18269, 0.320, 1000)
    x = array([[i, 0.0] for i in z_senterline])
    senterline = StatisticsProbes(x.flatten(), V)

    return dict(uv=df.Function(V), senterline=senterline, facets=facets, normal=normal)


def temporal_hook(**NS_namespace):
    pass
