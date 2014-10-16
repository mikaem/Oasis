__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-06"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

"""
This is a highly tuned and stripped down Navier-Stokes solver optimized
for both speed and memory. The algorithm used is a second order in time 
fractional step method (incremental pressure correction). The fractional 
step method can be used both non-iteratively or with iterations over the 
pressure velocity system.

Crank-Nicolson discretization is used in time for the Laplacian. There
are two options for convection - implicit or explicit. The implicit
version uses Crank-Nicolson for the convected velocity and an
Adams-Bashforth projection for the convecting velocity. The explicit
version is pure Adams-Bashforth.

The differences between the versions of the solver are only visible in
functions imported from the solverhooks folder:
  solvers/NSfracStep/IPCS_ABCN.py    # Implicit
  solvers/NSfracStep/IPCS_ABE.py     # Explicit
  solvers/NSfracStep/IPCS.py         # Naive

The third naive solver is very simple and not optimized. It is intended 
for validation of the other versions. A solver is chosen through command-
line keyword convection="ABCN", "ABE" or "naive".

The velocity vector is segregated, and we use three (in 3D) scalar 
velocity components

V = FunctionSpace(mesh, 'CG', 1)
u_components = ['u0', 'u1', 'u2'] in 3D, ['u0', 'u1'] in 2D
q_[ui] = Function(V) for ui = u_components
u_ = as_vector(q_['u0'], q_['u1'], q_['u2'])

A single coefficient matrix is assembled and used by all velocity 
componenets. It is built by preassembling as much as possible. 

The system of momentum equations solved are, for the implicit version, 
basically:

u = TrialFunction(V)
v = TestFunction(V)
U = 0.5*(u+q_1['u0'])     # Scalar
U1 = 1.5*u_1 - 0.5*u_2    # Vector
F = (1/dt)*inner(u - u_1, v)*dx + inner(grad(U)*U1, v)*dx + inner(p_.dx(0), v)*dx \
     nu*inner(grad(U), grad(v))*dx + inner(f[0], v)*dx
                
where (q_['u0'], p_.dx(0), f[0]) is replaced by (q_1['u1'], p_.dx(1), f[1]) and 
(q_1['u2'], p_.dx(2), f[2]) for the two other velocity components.
We solve an equation corresponding to lhs(F) == rhs(F) for all ui.
     
The variables u_1 and u_2 are velocity vectors at time steps k-1 and k-2. We 
are solving for u, which is the velocity at time step k. p_ is the latest 
approximation for the pressure.

The matrix corresponding to assemble(lhs(F)) is the same for all velocity
components and it is computed as:

    A  = 1/dt*M + 0.5*Ac + 0.5*nu*K
    
where

    M  = assemble(inner(u, v)*dx)
    Ac = assemble(inner(grad(u)*U1, v)*dx)
    K  = assemble(inner(grad(u), grad(v))*dx)

However, we start by assembling a coefficient matrix (A_rhs) that is used 
to compute parts of the rhs vector corresponding to mass, convection
and diffusion:

    A_rhs = 1/dt*M - 0.5*Ac - 0.5*nu*K
    b[ui]  = A_rhs*q_1[ui].vector()

The pressure gradient and body force need to be added to b as well. Three
matrices are preassembled for the computation of the pressure gradient:

  P = dict((ui, assemble(v*p.dx(i)*dx)) for i, ui in enumerate(u_components))

and the pressure gradient for each component of the momentum equation is 
then computed as

  assemble(p_.dx(i)*v*dx) = P[ui] * p_.vector()

If memory is an limiting factor, this term may be computed directly through
only the lhs each timestep. Memory is then saved since P is not preassembled.
Set parameter low_memory_version = True for lhs version.

Ac needs to be reassembled each new timestep. Ac is assembled into A to 
save memory. A and A_rhs are recreated each new timestep by assembling Ac, 
setting up A_rhs and then using the following to create A:

   A = -A_rhs + 2/dt*M

We then solve the linear system A * u = b[ui] for all q_[ui].vector()

Pressure is solved through

  inner(grad(p), grad(q))*dx == inner(grad(p_), grad(q))*dx - 1/dt*inner(div(u_), q)*dx

Here we assemble the rhs by:

  Ap = assemble(inner(grad(p), grad(q))*dx)
  bp = Ap * p_.vector()
  for ui in u_components:
    bp.axpy(-1./dt, Rx[ui]*x_[ui])
  where the preassemble Rx is:
    Rx = dict((ui, assemble(q*u.dx(i)*dx)) for i, ui in  enumerate(u_components))
  
  Alternatively, if low_memory_version is set to True, then Rx is not preassembled
  and assemble(q*u.dx(i)*dx is called each timestep.

We then solve Ap * p = bp for p_.vector().
  
Velocity update is computed through:

  inner(u, v)*dx == inner(q_[ui], v)*dx - dt*inner(dp_.dx(i), v)*dx

where each component on the rhs of the equation is computed effectively as
  assemble(inner(q_[ui], v)*dx) = M * q_[ui].vector()
  assemble(dt*inner(dp_.dx(i), v)*dx) = dt * P[ui] * dp_.vector()

where dp_ is the pressure correction, i.e., th newly computed pressure 
at the new timestep minus the pressure at previous timestep.

The lhs mass matrix is either the regular M, or the lumped and diagonal
mass matrix ML computed as
  ones = Function(V)
  ones.vector()[:] = 1.
  ML = M * ones.vector()

A scalar equation is solved through
  C = 0.5*(u + q_1['c'])
  F = (1/dt)*inner(u - q_1['c'], v)*dx + inner(grad(C)*U1, v)*dx \
     nu/Sc*inner(grad(C), grad(v))*dx + inner(fs['c'], v)*dx
     
where Sc is the constant Schmidt number and fs['c'] is a source to the
scalar equation. The scalar is using the same FunctionSpace as the 
velocity components and it is computed by reusing much of the velocity 
matrices

    A_rhs = 1/dt*M - 0.5*Ac - 0.5*nu/Sc*K
    b['c']  = A_rhs*q_1['c'].vector()
    Ta = -A_rhs + 2/dt*M

and we solve:    

    Ta * c = b['c'] for q_['c'].vector()
  
Ta may differ from A due to Dirichlet boundary conditions.
    
"""
from common import *

################### Problem dependent parameters ####################
###  Should import a mesh and a dictionary called NS_parameters   ###
###  See problems/NSfracStep/__init__.py for possible parameters  ###
#####################################################################

commandline_kwargs = parse_command_line()

default_problem = 'DrivenCavity'
exec("from problems.NSfracStep.{} import *".format(commandline_kwargs.get('problem', default_problem)))

# Update current namespace with NS_parameters and commandline_kwargs ++
vars().update(post_import_problem(**vars()))

# Import chosen functionality from solvers
exec("from solvers.NSfracStep.{} import *".format(solver))

# Create lists of components solved for
dim = mesh.geometry().dim()
u_components = map(lambda x: 'u'+str(x), range(dim))
sys_comp =  u_components + ['p'] + scalar_components
uc_comp  =  u_components + scalar_components

# Set up initial folders for storing results
newfolder, tstepfiles = create_initial_folders(**vars())

# Declare FunctionSpaces and arguments
V = Q = FunctionSpace(mesh, 'CG', velocity_degree, constrained_domain=constrained_domain)
if velocity_degree != pressure_degree:
    Q = FunctionSpace(mesh, 'CG', pressure_degree, constrained_domain=constrained_domain)

u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)
    
# Use dictionary to hold all FunctionSpaces
VV = dict((ui, V) for ui in uc_comp); VV['p'] = Q

# Create dictionaries for the solutions at three timesteps
q_  = dict((ui, Function(VV[ui], name=ui)) for ui in sys_comp)
q_1 = dict((ui, Function(VV[ui], name=ui+"_1")) for ui in sys_comp)
q_2 = dict((ui, Function(V, name=ui+"_2")) for ui in u_components)

# Read in previous solution if restarting
init_from_restart(**vars())

# Create vectors of the segregated velocity components    
u_  = as_vector([q_ [ui] for ui in u_components]) # Velocity vector at t
u_1 = as_vector([q_1[ui] for ui in u_components]) # Velocity vector at t - dt
u_2 = as_vector([q_2[ui] for ui in u_components]) # Velocity vector at t - 2*dt

# Adams Bashforth projection of velocity at t - dt/2
U_AB = 1.5*u_1 - 0.5*u_2

# Create short forms for accessing the solution vectors
x_  = dict((ui, q_ [ui].vector()) for ui in sys_comp)     # Solution vectors t
x_1 = dict((ui, q_1[ui].vector()) for ui in sys_comp)     # Solution vectors t - dt
x_2 = dict((ui, q_2[ui].vector()) for ui in u_components) # Solution vectors t - 2*dt

# Create vectors to hold rhs of equations
b     = dict((ui, Vector(x_[ui])) for ui in sys_comp)     # rhs vectors (final)
b_tmp = dict((ui, Vector(x_[ui])) for ui in sys_comp)     # rhs temp storage vectors

# Short forms pressure and scalars
p_  = q_ ['p']              # pressure at t
p_1 = q_1['p']              # pressure at t - dt
dp_ = Function(Q)           # pressure correction
for ci in scalar_components:
    exec("{}_   = q_ ['{}']".format(ci, ci))
    exec("{}_1  = q_1['{}']".format(ci, ci))

print_solve_info = use_krylov_solvers and krylov_solvers['monitor_convergence']

# Boundary conditions
bcs = create_bcs(**vars())

# Initialize solution
initialize(**vars())

#  Fetch linear algebra solvers 
u_sol, p_sol, c_sol = get_solvers(**vars())

# Get constant body forces
f = body_force(**vars())
assert(isinstance(f, Coefficient))
b0 = dict((ui, assemble(v*f[i]*dx)) for i, ui in enumerate(u_components))

# Get scalar sources
fs = scalar_source(**vars())
for ci in scalar_components:
    assert(isinstance(fs[ci], Coefficient))
    b0[ci] = assemble(v*fs[ci]*dx)

# Preassemble and allocate
vars().update(setup(**vars()))

# Anything problem specific
vars().update(pre_solve_hook(**vars()))

# At this point only convection is left to be assembled. Enable ferari
if parameters["form_compiler"].has_key("no_ferari") and not solver in ("IPCS", "Chorin"):
    parameters["form_compiler"].remove("no_ferari")

tic()
stop = False
total_timer = OasisTimer("Start simulations", True)
while t < (T - tstep*DOLFIN_EPS) and not stop:
    t += dt
    tstep += 1
    inner_iter = 0
    udiff = array([1e8]) # Norm of velocity change over last inner iter
    num_iter = max(iters_on_first_timestep, max_iter) if tstep == 1 else max_iter
    
    start_timestep_hook(**vars())
    
    while udiff[0] > max_error and inner_iter < num_iter:
        inner_iter += 1
        
        t0 = OasisTimer("Tentative velocity")
        if inner_iter == 1:
            assemble_first_inner_iter(**vars())
        udiff[0] = 0.0
        for i, ui in enumerate(u_components):
            t1 = OasisTimer('Solving tentative velocity '+ui, print_solve_info)
            velocity_tentative_assemble(**vars())
            velocity_tentative_hook    (**vars())
            velocity_tentative_solve   (**vars())
            t1.stop()
            
        t0 = OasisTimer("Pressure solve", print_solve_info)
        pressure_assemble(**vars())
        pressure_hook    (**vars())
        pressure_solve   (**vars())
        t0.stop()
                             
        print_velocity_pressure_info(**vars())

    # Update velocity 
    t0 = OasisTimer("Velocity update")
    velocity_update(**vars())
    t0.stop()
    
    # Solve for scalars
    if len(scalar_components) > 0:
        scalar_assemble(**vars())
        for ci in scalar_components:    
            t1 = OasisTimer('Solving scalar {}'.format(ci), print_solve_info)
            scalar_hook (**vars())
            scalar_solve(**vars())
            t1.stop()
        
    temporal_hook(**vars())
    
    # Save solution if required and check for killoasis file
    stop = save_solution(**vars())

    # Update to a new timestep
    for ui in u_components:
        x_2[ui].zero(); x_2[ui].axpy(1.0, x_1[ui])
        x_1[ui].zero(); x_1[ui].axpy(1.0, x_ [ui])
                
    for ci in scalar_components:
        x_1[ci].zero(); x_1[ci].axpy(1., x_[ci])

    # Print some information
    if tstep % print_intermediate_info == 0:
        info_green('Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}'.format(t, tstep, T)) 
        info_red('Total computing time on previous {0:d} timesteps = {1:f}'.format(print_intermediate_info, toc()))
        list_timings(True)
        tic()
          
    # AB projection for pressure on next timestep
    if AB_projection_pressure and t < (T - tstep*DOLFIN_EPS) and not stop:
        x_['p'].axpy(0.5, dp_.vector())
                                    
total_timer.stop()
list_timings()
info_red('Total computing time = {0:f}'.format(total_timer.value()))
oasis_memory('Final memory use ')
total_initial_dolfin_memory = MPI.sum(mpi_comm_world(), initial_memory_use)
info_red('Memory use for importing dolfin = {} MB (RSS)'.format(total_initial_dolfin_memory))
info_red('Total memory use of solver = ' + str(oasis_memory.memory - total_initial_dolfin_memory) + " MB (RSS)")

# Final hook
theend_hook(**vars())
