__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

# from dolfin import *
import dolfin as df
import subprocess
from os import getpid, path
from collections import defaultdict
from numpy import array, maximum, zeros


# UnitSquareMesh(20, 20) # Just due to MPI bug on Scinet

# try:
# from fenicstools import getMemoryUsage

# except:


def getMemoryUsage(rss=True):
    mypid = str(getpid())
    rss = "rss" if rss else "vsz"
    process = subprocess.Popen(["ps", "-o", rss, mypid], stdout=subprocess.PIPE)
    out, _ = process.communicate()
    mymemory = out.split()[1]
    return eval(mymemory) / 1024


df.parameters["linear_algebra_backend"] = "PETSc"
df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"
# df.parameters["form_compiler"]["quadrature_degree"] = 4
# df.parameters["form_compiler"]["cache_dir"] = "instant"
df.parameters["form_compiler"]["cpp_optimize_flags"] = "-O3"
# df.parameters["mesh_partitioner"] = "ParMETIS"
# df.parameters["form_compiler"].add("no_ferari", True)
# df.set_log_active(False)

# Default parameters for all solvers should not be defined here, but rather in NSfrac.py or NScoupled.__init__.py with the rest
# NS_parameters = dict(
#     nu=0.01,  # Kinematic viscosity
#     folder="results",  # Relative folder for storing results
#     velocity_degree=2,  # default velocity degree
#     pressure_degree=1,  # default pressure degree
# )

# NS_expressions = {}

constrained_domain = None

# To solve for scalars provide a list like ['scalar1', 'scalar2']
scalar_components = []

# With diffusivities given as a Schmidt number defined by:
#   Schmidt = nu / D (= momentum diffusivity / mass diffusivity)
Schmidt = defaultdict(lambda: 1.0)  # Schmidt["any_key"] returns 1.0
Schmidt_T = defaultdict(lambda: 0.7)  # Turbulent Schmidt number (LES)

Scalar = defaultdict(lambda: dict(Schmidt=1.0, family="CG", degree=1))

# The following helper functions are available in dolfin
# They are redefined here for printing only on process 0.
RED = "\033[1;37;31m%s\033[0m"
BLUE = "\033[1;37;34m%s\033[0m"
GREEN = "\033[1;37;32m%s\033[0m"


def info_blue(s, check=True):
    if df.MPI.rank(df.MPI.comm_world) == 0 and check:
        print(BLUE % s)


def info_green(s, check=True):
    if df.MPI.rank(df.MPI.comm_world) == 0 and check:
        print(GREEN % s)


def info_red(s, check=True):
    if df.MPI.rank(df.MPI.comm_world) == 0 and check:
        print(RED % s)


class OasisTimer(df.Timer):
    def __init__(self, task, verbose=False):
        df.Timer.__init__(self, task)
        info_blue(task, verbose)


class OasisMemoryUsage:
    def __init__(self, s):
        self.memory = 0
        self.memory_vm = 0
        self(s)

    def __call__(self, s, verbose=False):
        self.prev = self.memory
        self.prev_vm = self.memory_vm
        self.memory = df.MPI.sum(df.MPI.comm_world, getMemoryUsage())
        self.memory_vm = df.MPI.sum(df.MPI.comm_world, getMemoryUsage(False))
        if df.MPI.rank(df.MPI.comm_world) == 0 and verbose:
            info_blue(
                "{0:26s}  {1:10d} MB {2:10d} MB {3:10d} MB {4:10d} MB".format(
                    s,
                    int(self.memory - self.prev),
                    int(self.memory),
                    int(self.memory_vm - self.prev_vm),
                    int(self.memory_vm),
                )
            )


# Print memory use up til now
initial_memory_use = getMemoryUsage()
oasis_memory = OasisMemoryUsage("Start")


# Convenience functions
def strain(u):
    return 0.5 * (df.grad(u) + df.grad(u).T)


def omega(u):
    return 0.5 * (df.grad(u) - df.grad(u).T)


def Omega(u):
    return df.inner(omega(u), omega(u))


def Strain(u):
    return df.inner(strain(u), strain(u))


def QC(u):
    return Omega(u) - Strain(u)


# dont use this
def recursive_update(dst, src):
    """Update dict dst with items from src deeply ("deep update")."""
    for key, val in src.items():
        if key in dst and isinstance(val, dict) and isinstance(dst[key], dict):
            dst[key] = recursive_update(dst[key], val)
        else:
            dst[key] = val
    return dst


class OasisXDMFFile(df.XDMFFile, object):
    def __init__(self, comm, filename):
        df.XDMFFile.__init__(self, comm, filename)


def add_function_to_tstepfiles(function, newfolder, tstepfiles, tstep):
    name = function.name()
    tstepfolder = path.join(newfolder, "Timeseries")
    tstepfiles[name] = OasisXDMFFile(
        df.MPI.comm_world,
        path.join(tstepfolder, "{}_from_tstep_{}.xdmf".format(name, tstep)),
    )
    tstepfiles[name].function = function
    tstepfiles[name].parameters["rewrite_function_mesh"] = False


def body_force(mesh, **NS_namespace):
    """Specify body force"""
    return df.Constant((0,) * mesh.geometry().dim())


def initialize(**NS_namespace):
    """Initialize solution."""
    pass


def create_bcs(sys_comp, **NS_namespace):
    """Return dictionary of Dirichlet boundary conditions."""
    return dict((ui, []) for ui in sys_comp)


def scalar_hook(**NS_namespace):
    """Called prior to scalar solve."""
    pass


def scalar_source(scalar_components, **NS_namespace):
    """Return a dictionary of scalar sources."""
    return dict((ci, df.Constant(0)) for ci in scalar_components)


def pre_solve_hook(**NS_namespace):
    """Called just prior to entering time-loop. Must return a dictionary."""
    return {}


def theend_hook(**NS_namespace):
    """Called at the very end."""
    pass


def get_problem_parameters(**NS_namespace):
    """Updates problem specific parameters, and handles restart"""
    pass


def post_import_problem(NS_parameters, NS_expressions, mesh, commandline_kwargs):
    """Called after importing from problem."""

    # # set default parameters
    # for key, val in NS_parameters.items():
    #     if key not in NS_parameters.keys():
    #         NS_parameters[key] = val

    # Update NS_namespace with all parameters modified through command line
    for key, val in commandline_kwargs.items():
        if key not in NS_parameters.keys():
            raise KeyError("unknown key", key)
        if isinstance(val, dict):
            NS_parameters[key].update(val)
        else:
            NS_parameters[key] = val

    NS_parameters.update(NS_expressions)

    # If the mesh is a callable function, then create the mesh here.
    if callable(mesh):
        mesh = mesh(**NS_parameters)
    assert isinstance(mesh, df.Mesh)

    # split :
    # this is the dictionary that will be saved, it should not be changed
    problem_parameters = {}  # problem specific parameters
    NS_namespace = {}  # objects, functions, ... that are needed to solve the problem
    NS_namespace["mesh"] = mesh

    for key, val in NS_parameters.items():
        if type(val) in [str, bool, type(None), float, int]:
            problem_parameters[key] = val
        elif type(val) == dict:
            k0 = list(val.keys())[0]
            if type(val[k0]) in [str, bool, type(None), float, int]:
                problem_parameters[key] = val
            else:
                print(key, val)
                NS_namespace[key] = val
        elif type(val) == list:
            if len(val) == 0:
                problem_parameters[key] = val
            elif (len(val) == 0) | (
                type(val[0]) in [str, bool, type(None), float, int]
            ):
                problem_parameters[key] = val
            else:
                print(key, val)
                NS_namespace[key] = val
        else:
            print(key, val)
            NS_namespace[key] = val
    return NS_namespace, problem_parameters


class Domain:
    def __init__(self):
        #
        return

    def get_problem_parameters(self):
        raise NotImplementedError()

    def scalar_source(self):
        self.scalar_components
        return dict((ci, df.Constant(0)) for ci in scalar_components)

    def create_bcs(self):
        sys_comp = self.sys_comp
        return dict((ui, []) for ui in sys_comp)

    def initialize(self):
        raise NotImplementedError()

    def body_force(self):
        """Specify body force"""
        mesh = self.mesh
        return df.Constant((0,) * mesh.geometry().dim())

    def pre_solve_hook(self):
        raise NotImplementedError()

    def scalar_hook(self):
        raise NotImplementedError()

    def theend_hook(self):
        raise NotImplementedError()

    def recommend_dt(self):
        Cmax = 0.05
        dt = Cmax * self.mesh.hmin() / self.Umean
        print("recommended dt =", dt)
        return dt

    def set_parameters_from_commandline(self, commandline_kwargs):
        # Update NS_namespace with all parameters modified through command line
        for key, val in commandline_kwargs.items():
            setattr(self, key, commandline_kwargs[key])
            if key not in self.__dict__.keys():
                raise KeyError("unknown key", key)
            elif isinstance(val, dict):
                setattr(self, key, commandline_kwargs[key])
            else:
                setattr(self, key, commandline_kwargs[key])
        return

    def show_info(self, t, tstep, toc):

        info_green(
            "Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}".format(
                t, tstep, self.T
            )
        )
        info_red(
            "Total computing time on previous {0:d} timesteps = {1:f}".format(
                self.print_intermediate_info, toc
            )
        )
