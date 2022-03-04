__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-09"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

# from oasis.problems import *
from oasis.problems import (
    subprocess,
    getpid,
    path,
    defaultdict,
    array,
    maximum,
    zeros,
    getMemoryUsage,
    # NS_parameters,
    # NS_expressions,
    constrained_domain,
    scalar_components,
    Schmidt,
    Schmidt_T,
    Scalar,
    RED,
    BLUE,
    GREEN,
    info_blue,
    info_green,
    info_red,
    OasisTimer,
    OasisMemoryUsage,
    initial_memory_use,
    oasis_memory,
    strain,
    omega,
    Omega,
    Strain,
    QC,
    recursive_update,
    OasisXDMFFile,
    add_function_to_tstepfiles,
    body_force,
    initialize,
    # create_bcs,
    scalar_hook,
    scalar_source,
    pre_solve_hook,
    theend_hook,
    get_problem_parameters,
    post_import_problem,
    Domain,
)
import dolfin as df
from ufl import Coefficient

# Default parameters NSfracStep solver
NS_expressions = {}
default_parameters = dict(
    nu=0.01,  # Kinematic viscosity
    folder="results",  # Relative folder for storing results
    velocity_degree=2,  # default velocity degree
    pressure_degree=1,  # default pressure degree
    # Physical constants and solver parameters
    t=0.0,  # Time
    tstep=0,  # Timestep
    T=1.0,  # End time
    dt=0.01,  # Time interval on each timestep
    # Some discretization options
    # Use Adams Bashforth projection as first estimate for pressure on new timestep
    AB_projection_pressure=False,
    solver="IPCS_ABCN",  # "IPCS_ABCN", "IPCS_ABE", "IPCS", "Chorin", "BDFPC", "BDFPC_Fast"
    # Parameters used to tweek solver
    max_iter=1,  # Number of inner pressure velocity iterations on timestep
    max_error=1e-6,  # Tolerance for inner iterations (pressure velocity iterations)
    iters_on_first_timestep=2,  # Number of iterations on first timestep
    use_krylov_solvers=True,  # Otherwise use LU-solver
    print_intermediate_info=10,
    print_velocity_pressure_convergence=False,
    # Parameters used to tweek output
    plot_interval=10,
    checkpoint=10,  # Overwrite solution in Checkpoint folder each checkpoint
    save_step=10,  # Store solution each save_step
    restart_folder=None,  # If restarting solution, set the folder holding the solution to start from here
    output_timeseries_as_vector=True,  # Store velocity as vector in Timeseries
    # Stop simulations cleanly after the given number of seconds
    killtime=None,
    # Choose LES model and set default parameters
    # NoModel, Smagorinsky, Wale, DynamicLagrangian, ScaleDepDynamicLagrangian
    les_model="NoModel",
    # LES model parameters
    Smagorinsky=dict(Cs=0.1677),  # Standard Cs, same as OpenFOAM
    Wale=dict(Cw=0.325),
    DynamicSmagorinsky=dict(
        Cs_comp_step=1
    ),  # Time step interval for Cs to be recomputed
    KineticEnergySGS=dict(Ck=0.08, Ce=1.05),
    # Choose Non-Newtonian model and set default parameters
    # NoModel, ModifiedCross
    nn_model="NoModel",
    # Non-Newtonian model parameters
    ModifiedCross=dict(
        lam=3.736,  # s
        m_param=2.406,  # for Non-Newtonian model
        a_param=0.34,  # for Non-Newtonian model
        mu_inf=0.00372,  # Pa-s for non-Newtonian model
        mu_o=0.09,  # Pa-s for non-Newtonian model
        rho=1085,  # kg/m^3
    ),
    # Parameter set when enabling test mode
    testing=False,
    # Solver parameters that will be transferred to dolfins parameters['krylov_solver']
    krylov_solvers=dict(
        monitor_convergence=False,
        report=False,
        error_on_nonconvergence=False,
        nonzero_initial_guess=True,
        maximum_iterations=200,
        relative_tolerance=1e-8,
        absolute_tolerance=1e-8,
    ),
    # Velocity update
    velocity_update_solver=dict(
        method="default",  # "lumping", "gradient_matrix"
        solver_type="cg",
        preconditioner_type="jacobi",
        low_memory_version=False,
    ),
    velocity_krylov_solver=dict(solver_type="bicgstab", preconditioner_type="jacobi"),
    pressure_krylov_solver=dict(solver_type="gmres", preconditioner_type="hypre_amg"),
    scalar_krylov_solver=dict(solver_type="bicgstab", preconditioner_type="jacobi"),
    nut_krylov_solver=dict(
        method="WeightedAverage",  # Or 'default'
        solver_type="cg",
        preconditioner_type="jacobi",
    ),
    nu_nn_krylov_solver=dict(
        method="WeightedAverage",  # Or 'default'
        solver_type="cg",
        preconditioner_type="jacobi",
    ),
)


def velocity_tentative_hook(**NS_namespace):
    """Called just prior to solving for tentative velocity."""
    pass


def pressure_hook(**NS_namespace):
    """Called prior to pressure solve."""
    pass


def start_timestep_hook(**NS_namespace):
    """Called at start of new timestep"""
    pass


def temporal_hook(**NS_namespace):
    """Called at end of a timestep."""
    pass


class FracDomain(Domain):
    def __init__(self):
        # self.nu = 0.01  # Kinematic viscosity
        self.folder = "results"  # Relative folder for storing results
        self.velocity_degree = 2  # default velocity degree
        self.pressure_degree = 1  # default pressure degree

        # Physical constants and solver parameters
        # self.t = 0.0  # Time
        # self.tstep = 0  # Timestep
        self.T = 1.0  # End time
        self.dt = 0.01  # Time interval on each timestep

        # Some discretization options
        # Use Adams Bashforth projection as first estimate for pressure on new timestep
        self.AB_projection_pressure = False
        # "IPCS_ABCN", "IPCS_ABE", "IPCS", "Chorin", "BDFPC", "BDFPC_Fast"
        self.solver = "IPCS_ABCN"

        # Parameters used to tweek solver
        # Number of inner pressure velocity iterations on timestep
        self.max_iter = 1
        # Tolerance for inner iterations pressure velocity iterations
        self.max_error = 1e-6
        # Number of iterations on first timestep
        self.iters_on_first_timestep = 2
        self.use_krylov_solvers = True  # Otherwise use LU-solver
        self.print_intermediate_info = 10
        self.print_velocity_pressure_convergence = False

        # Parameters used to tweek output
        self.plot_interval = 10
        # Overwrite solution in Checkpoint folder each checkpoint
        self.checkpoint = 10
        self.save_step = 10  # Store solution each save_step
        # If restarting solution, set the folder holding the solution to start from here
        self.restart_folder = None
        # Store velocity as vector in Timeseries
        self.output_timeseries_as_vector = True
        # Stop simulations cleanly after the given number of seconds
        self.killtime = None

        # Choose LES model and set default parameters
        # NoModel, Smagorinsky, Wale, DynamicLagrangian, ScaleDepDynamicLagrangian
        self.les_model = "NoModel"
        # LES model parameters
        self.Smagorinsky = {"Cs": 0.1677}  # Standard Cs, same as OpenFOAM
        self.Wale = {"Cw": 0.325}
        # Time step interval for Cs to be recomputed
        self.DynamicSmagorinsky = {"Cs_comp_step": 1}
        self.KineticEnergySGS = {"Ck": 0.08, "Ce": 1.05}

        # Choose Non-Newtonian model and set default parameters
        # NoModel, ModifiedCross
        self.nn_model = "NoModel"
        # Non-Newtonian model parameters
        self.ModifiedCross = {
            "lam": 3.736,  # s
            "m_param": 2.406,  # for Non-Newtonian model
            "a_param": 0.34,  # for Non-Newtonian model
            "mu_inf": 0.00372,  # Pa-s for non-Newtonian model
            "mu_o": 0.09,  # Pa-s for non-Newtonian model
            "rho": 1085,  # kg/m^3
        }
        # Parameter set when enabling test mode
        self.testing = False
        # Solver parameters that will be transferred to dolfins parameters['krylov_solver']
        self.krylov_solvers = {
            "monitor_convergence": False,
            "report": False,
            "error_on_nonconvergence": False,
            "nonzero_initial_guess": True,
            "maximum_iterations": 200,
            "relative_tolerance": 1e-8,
            "absolute_tolerance": 1e-8,
        }
        # Velocity update
        self.velocity_update_solver = {
            "method": "default",  # "lumping", "gradient_matrix"
            "solver_type": "cg",
            "preconditioner_type": "jacobi",
            "low_memory_version": False,
        }
        self.velocity_krylov_solver = {
            "solver_type": "bicgstab",
            "preconditioner_type": "jacobi",
        }
        self.pressure_krylov_solver = {
            "solver_type": "gmres",
            "preconditioner_type": "hypre_amg",
        }
        self.scalar_krylov_solver = {
            "solver_type": "bicgstab",
            "preconditioner_type": "jacobi",
        }
        self.nut_krylov_solver = {
            "method": "WeightedAverage",  # Or 'default'
            "solver_type": "cg",
            "preconditioner_type": "jacobi",
        }
        self.nu_nn_krylov_solver = {
            "method": "WeightedAverage",  # Or 'default'
            "solver_type": "cg",
            "preconditioner_type": "jacobi",
        }

        self.constrained_domain = None
        return

    def initialize_problem_components(self):
        # Create lists of components solved for
        # self.scalar_components = scalar_components
        if self.mesh.geometry().dim() == 1:
            self.u_components = ["u0"]
        elif self.mesh.geometry().dim() == 2:
            self.u_components = ["u0", "u1"]
        elif self.mesh.geometry().dim() == 3:
            self.u_components = ["u0", "u1", "u2"]
        self.sys_comp = self.u_components + ["p"] + self.scalar_components
        self.uc_comp = self.u_components + self.scalar_components
        # sys_comp = ['u0', 'u1', 'p', 'alfa']
        # u_components = ['u0', 'u1']
        # uc_comp = ['u0', 'u1', 'alfa']
        return

    def dolfin_variable_declaration(self):
        cd = self.constrained_domain
        mesh = self.mesh
        sys_comp = self.sys_comp
        deg_v, deg_p = self.velocity_degree, self.pressure_degree
        V = Q = df.FunctionSpace(mesh, "CG", deg_v, constrained_domain=cd)
        if deg_v != deg_p:
            Q = df.FunctionSpace(mesh, "CG", deg_p, constrained_domain=cd)
        self.V, self.Q = V, Q
        self.u, self.v = u, v = df.TrialFunction(V), df.TestFunction(V)
        self.p, self.q = p, q = df.TrialFunction(Q), df.TestFunction(Q)

        # Use dictionary to hold all FunctionSpaces
        VV = dict((ui, V) for ui in self.uc_comp)
        VV["p"] = Q

        # removed unused name argument and reassigning q_...[...].vector() to x_...
        # Create dictionaries for the solutions at three timesteps
        self.q_ = dict((ui, df.Function(VV[ui])) for ui in sys_comp)
        self.q_1 = dict((ui, df.Function(VV[ui])) for ui in sys_comp)
        self.q_2 = dict((ui, df.Function(V)) for ui in self.u_components)
        # Create vectors of the segregated velocity components
        self.u_ = df.as_vector([self.q_[ui] for ui in self.u_components])
        self.u_1 = df.as_vector([self.q_1[ui] for ui in self.u_components])
        self.u_2 = df.as_vector([self.q_2[ui] for ui in self.u_components])
        # Adams Bashforth projection of velocity at t - dt/2
        self.U_AB = 1.5 * self.u_1 - 0.5 * self.u_2
        # Create vectors to hold rhs of equations
        self.b = dict((ui, df.Vector(self.q_[ui].vector())) for ui in sys_comp)
        self.b_tmp = dict((ui, df.Vector(self.q_[ui].vector())) for ui in sys_comp)
        self.dp_ = df.Function(Q)  # pressure correction

        # TODO: remove u_, u_1, u_w -> redundand!
        # x_, x_1, x_2 removed, they are in q_[...].vector(), q_1[...].vector(), q_2[...].vector()
        # alpha_, alpha_1 removed, they are in q_ and q_1
        # p_, p_1 removed, they are in q_, q_1

        # Get constant body forces
        self.f = f = self.body_force()
        assert isinstance(f, Coefficient)
        self.b0 = b0 = {}
        for i, ui in enumerate(self.u_components):
            b0[ui] = df.assemble(v * f[i] * df.dx)

        # Get scalar sources
        self.fs = fs = self.scalar_source()
        for ci in scalar_components:
            assert isinstance(fs[ci], Coefficient)
            b0[ci] = df.assemble(v * fs[ci] * df.dx)

    def apply_bcs(self):
        # used to be initialize(x_1, x_2, bcs, **NS_namespace)
        for ui in self.sys_comp:
            [bc.apply(self.q_1[ui].vector()) for bc in self.bcs[ui]]
        for ui in self.u_components:
            [bc.apply(self.q_2[ui].vector()) for bc in self.bcs[ui]]
        return

    def advance(self):
        # Update to a new timestep
        # replaced axpy with assign
        for ui in self.u_components:
            self.q_2[ui].assign(self.q_1[ui])
            self.q_1[ui].assign(self.q_[ui])
            # self.q_2[ui].vector().zero()
            # self.q_2[ui].vector().axpy(1.0, self.q_1[ui].vector())
            # self.q_1[ui].vector().zero()
            # self.q_1[ui].vector().axpy(1.0, self.q_[ui].vector())

        for ci in self.scalar_components:
            self.q_1[ci].assign(self.q_[ci])
            # self.q_1[ci].vector().zero()
            # self.q_1[ci].vector().axpy(1.0, self.q_[ci].vector())
        return

    def velocity_tentative_hook(self, **kvargs):
        """Called just prior to solving for tentative velocity."""
        pass

    def pressure_hook(self, **kvargs):
        """Called prior to pressure solve."""
        pass

    def start_timestep_hook(self, **kvargs):
        """Called at start of new timestep"""
        pass

    def temporal_hook(self, **kvargs):
        """Called at end of a timestep."""
        pass

    def print_velocity_pressure_info(self, num_iter, inner_iter, udiff):
        if num_iter > 1 and self.print_velocity_pressure_convergence:
            if inner_iter == 1:
                info_blue("  Inner iterations velocity pressure:")
                info_blue("                 error u  error p")
            info_blue(
                "    Iter = {0:4d}, {1:2.2e} {2:2.2e}".format(
                    inner_iter, udiff[0], df.norm(self.dp_.vector())
                )
            )
