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
import numpy as np

# from oasis.problems.TaylorGreen2D import mesh
import dolfin as df

try:
    from matplotlib import pyplot as plt
except:
    pass


def get_problem_parameters(**kwargs):
    NS_parameters = dict(
        scalar_components=scalar_components,
        Schmidt=Schmidt,
        Schmidt_T=Schmidt_T,
        nu=0.01,
        T=1.0,
        dt=0.001,
        Nx=20,
        Ny=20,
        folder="taylorgreen2D_results",
        plot_interval=1000,
        save_step=10000,
        checkpoint=10000,
        print_intermediate_info=1000,
        compute_error=1,
        use_krylov_solvers=True,
        velocity_degree=1,
        pressure_degree=1,
        krylov_report=False,
    )

    NS_parameters["krylov_solvers"] = {
        "monitor_convergence": False,
        "report": False,
        "relative_tolerance": 1e-12,
        "absolute_tolerance": 1e-12,
    }
    # set default parameters
    for key, val in default_parameters.items():
        if key not in NS_parameters.keys():
            NS_parameters[key] = val
    NS_expressions = dict(
        dict(
            constrained_domain=PeriodicDomain(),
            initial_fields=dict(
                u0="-sin(pi*x[1])*cos(pi*x[0])*exp(-2.*pi*pi*nu*t)",
                u1="sin(pi*x[0])*cos(pi*x[1])*exp(-2.*pi*pi*nu*t)",
                p="-(cos(2*pi*x[0])+cos(2*pi*x[1]))*exp(-4.*pi*pi*nu*t)/4.",
            ),
            dpdx=(
                "sin(2*pi*x[0])*2*pi*exp(-4.*pi*pi*nu*t)/4.",
                "sin(2*pi*x[1])*2*pi*exp(-4.*pi*pi*nu*t)/4.",
            ),
            total_error=np.zeros(3),
        )
    )
    return NS_parameters, NS_expressions


def mesh(Nx, Ny, **params):
    return df.RectangleMesh(df.Point(0, 0), df.Point(2, 2), Nx, Ny)


class PeriodicDomain(df.SubDomain):
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 2) and (2, 0)
        return bool(
            (df.near(x[0], 0) or df.near(x[1], 0))
            and (
                not (
                    (df.near(x[0], 0) and df.near(x[1], 2))
                    or (df.near(x[0], 2) and df.near(x[1], 0))
                )
            )
            and on_boundary
        )

    def map(self, x, y):
        if df.near(x[0], 2) and df.near(x[1], 2):
            y[0] = x[0] - 2.0
            y[1] = x[1] - 2.0
        elif df.near(x[0], 2):
            y[0] = x[0] - 2.0
            y[1] = x[1]
        else:
            y[0] = x[0]
            y[1] = x[1] - 2.0


def initialize(q_, q_1, q_2, VV, t, nu, dt, initial_fields, solver, **NS_namespace):
    """Initialize solution.

    Use t=dt/2 for pressure since pressure is computed in between timesteps.

    """
    for ui in q_:
        if "IPCS" in solver:
            # if "IPCS" in NS_parameters["solver"]:
            deltat = dt / 2.0 if ui is "p" else 0.0
        else:
            deltat = 0.0
        vv = df.interpolate(
            df.Expression(
                (initial_fields[ui]), element=VV[ui].ufl_element(), t=t + deltat, nu=nu
            ),
            VV[ui],
        )
        q_[ui].vector()[:] = vv.vector()[:]
        if not ui == "p":
            q_1[ui].vector()[:] = vv.vector()[:]
            deltat = -dt
            vv = df.interpolate(
                df.Expression(
                    (initial_fields[ui]),
                    element=VV[ui].ufl_element(),
                    t=t + deltat,
                    nu=nu,
                ),
                VV[ui],
            )
            q_2[ui].vector()[:] = vv.vector()[:]
    q_1["p"].vector()[:] = q_["p"].vector()[:]


def temporal_hook(
    q_,
    t,
    nu,
    VV,
    dt,
    plot_interval,
    initial_fields,
    tstep,
    sys_comp,
    compute_error,
    total_error,
    solver,
    **NS_namespace
):
    """Function called at end of timestep.

    Plot solution and compute error by comparing to analytical solution.
    Remember pressure is computed in between timesteps.

    """
    if tstep % plot_interval == 0:
        df.plot(q_["u0"], title="u")
        df.plot(q_["u1"], title="v")
        df.plot(q_["p"], title="p")
        try:
            plt.show()
        except:
            pass

    if tstep % compute_error == 0:
        err = {}
        for i, ui in enumerate(sys_comp):
            if "IPCS" in solver:
                # if "IPCS" in NS_parameters["solver"]:
                deltat_ = dt / 2.0 if ui is "p" else 0.0
            else:
                deltat_ = 0.0
            ue = df.Expression(
                (initial_fields[ui]), element=VV[ui].ufl_element(), t=t - deltat_, nu=nu
            )
            ue = df.interpolate(ue, VV[ui])
            uen = df.norm(ue.vector())
            ue.vector().axpy(-1, q_[ui].vector())
            error = df.norm(ue.vector()) / uen
            err[ui] = "{0:2.6e}".format(df.norm(ue.vector()) / uen)
            total_error[i] += error * dt
        if df.MPI.rank(df.MPI.comm_world) == 0:
            print("Error is ", err, " at time = ", t)


def theend_hook(
    mesh,
    q_,
    t,
    dt,
    nu,
    VV,
    sys_comp,
    total_error,
    initial_fields,
    solver,
    **NS_namespace
):
    final_error = np.zeros(len(sys_comp))
    for i, ui in enumerate(sys_comp):
        if "IPCS" in solver:
            # if "IPCS" in NS_parameters["solver"]:
            deltat = dt / 2.0 if ui is "p" else 0.0
        else:
            deltat = 0.0
        ue = df.Expression(
            (initial_fields[ui]), element=VV[ui].ufl_element(), t=t - deltat, nu=nu
        )
        ue = df.interpolate(ue, VV[ui])
        final_error[i] = df.errornorm(q_[ui], ue)

    hmin = mesh.hmin()
    if df.MPI.rank(df.MPI.comm_world) == 0:
        print("hmin = {}".format(hmin))
    s0 = "Total Error:"
    s1 = "Final Error:"
    for i, ui in enumerate(sys_comp):
        s0 += " {0:}={1:2.6e}".format(ui, total_error[i])
        s1 += " {0:}={1:2.6e}".format(ui, final_error[i])

    if df.MPI.rank(df.MPI.comm_world) == 0:
        print(s0)
        print(s1)
