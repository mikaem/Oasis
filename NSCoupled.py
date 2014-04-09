__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-04"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from common import *

################### Problem dependent parameters ####################
###  Should import a mesh and a dictionary called NS_parameters   ###
###  See problems/NSCoupled/__init__.py for possible parameters   ###
#####################################################################

commandline_kwargs = parse_command_line()

default_problem = 'DrivenCavity'
exec("from problems.NSCoupled.{} import *".format(commandline_kwargs.get('problem', default_problem)))

# Update NS_parameters with parameters modified through the command line 
NS_parameters.update(commandline_kwargs)
vars().update(NS_parameters)  

vars().update(post_import_problem(**vars()))

# If the mesh is a callable function, then create the mesh here.
if callable(mesh):
    mesh = mesh(**NS_parameters)

assert(isinstance(mesh, Mesh)) 

# Import chosen functionality from solvers
default_solver = 'default'
exec("from solvers.NSCoupled.{} import *".format(commandline_kwargs.get('solver', default_solver)))

# Create lists of components solved for
u_components = ['u']
sys_comp =  ['up'] + scalar_components

# Set up initial folders for storing results
#newfolder, tstepfiles = create_initial_folders(**vars())

# Get the chosen mixed elment
element = commandline_kwargs.get("element", "CR")
vars().update(elements[element])

# TaylorHood may overload degree of elements
if element == "TaylorHood":
    degree["u"] = commandline_kwargs.get('velocity_degree', degree["u"])
    degree["p"] = commandline_kwargs.get('pressure_degree', degree["p"])
    # Should assert that degree["p"] = degree["u"]-1 ??
    
# Declare FunctionSpaces and arguments
V = VectorFunctionSpace(mesh, family["u"], degree["u"], constrained_domain=constrained_domain)
Q = FunctionSpace(mesh, family["p"], degree["p"], constrained_domain=constrained_domain)

# MINI element has bubble, add to V
if bubble:
    V = V + VectorFunctionSpace(mesh, "Bubble", 3)

# Create Mixed space
VQ = V * Q

# Create trial and test functions
up = TrialFunction(VQ)
u, p = split(up)
v, q = TestFunctions(VQ)

# For scalars use Q space
c  = TrialFunction(Q)
ct = TestFunction(Q)

VV = dict(up=VQ)
VV.update(dict((ui, Q) for ui in scalar_components))

# Create dictionaries for the solutions at two timesteps
q_  = dict((ui, Function(VV[ui], name=ui)) for ui in sys_comp)
q_1 = dict((ui, Function(VV[ui], name=ui+"_1")) for ui in sys_comp)

# Short forms
up_  = q_ ["up"] # Solution at next iteration
up_1 = q_1["up"] # Solution at previous iteration 
u_, p_ = split(up_)
u_1, p_1 = split(up_1)

# Create short forms for accessing the solution vectors
x_  = dict((ui, q_ [ui].vector()) for ui in sys_comp)     # Solution vectors 
x_1 = dict((ui, q_1[ui].vector()) for ui in sys_comp)     # Solution vectors previous iteration

# Create vectors to hold rhs of equations
b = dict((ui, Vector(x_[ui])) for ui in sys_comp)

# Short form scalars
for ci in scalar_components:
    exec("{}_   = q_ ['{}']".format(ci, ci))
    exec("{}_1  = q_1['{}']".format(ci, ci))

# Boundary conditions
bcs = create_bcs(**vars())

# Initialize solution
initialize(**vars())

#  Fetch linear algebra solvers 
up_sol, c_sol = get_solvers(**vars())

# Get constant body forces
f = body_force(**vars())
assert(isinstance(f, Coefficient))
b0 = dict(up=assemble(dot(v, f)*dx))

# Get scalar sources
fs = scalar_source(**vars())
for ci in scalar_components:
    assert(isinstance(fs[ci], Coefficient))
    b0[ci] = assemble(ct*fs[ci]*dx)

# Preassemble and allocate
vars().update(setup(**vars()))

# Assemble rhs once, before entering loop
b["up"] = assemble(F, tensor=b["up"])
for bc in bcs["up"]:
    bc.apply(b["up"], up_.vector())

# Anything problem specific
vars().update(pre_solve_hook(**vars()))

tic()
total_timer = OasisTimer("Start iterations", True)
iter = 0
while iter < max_iter and error > max_error:
    start_new_iter_hook(**vars())

    NS_assemble(**vars())
    NS_hook(**vars())
    NS_solve(**vars())
        
    # Solve for scalars
    if len(scalar_components) > 0:
        scalar_assemble(**vars())
        for ci in scalar_components:    
            t1 = OasisTimer('Solving scalar {}'.format(ci), print_solve_info)
            scalar_hook (**vars())
            scalar_solve(**vars())
            t1.stop()
            
    error = b["up"].norm("l2")
    
    print_velocity_pressure_info(**vars())
        
    end_iter_hook(**vars())

    # Update to next iteration
    for ui in sys_comp:
        x_1[ui].zero(); x_1[ui].axpy(1.0, x_ [ui])

    iter += 1

# FIXME not implemented yet for coupled
#save_solution(**vars()) 
                                              
total_timer.stop()
list_timings()
info_red('Total computing time = {0:f}'.format(total_timer.value()))
oasis_memory('Final memory use ')
total_initial_dolfin_memory = MPI.sum(mpi_comm_world(), initial_memory_use)
info_red('Memory use for importing dolfin = {} MB (RSS)'.format(total_initial_dolfin_memory))
info_red('Total memory use of solver = ' + str(oasis_memory.memory - total_initial_dolfin_memory) + " MB (RSS)")

# Final hook
theend_hook(**vars())
