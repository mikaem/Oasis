__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-12-19"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

"""
This is a highly tuned and stripped down Navier-Stokes solver optimized
for both speed and memory. The algorithm used is a second order in time 
fractional step method (incremental pressure correction).

Crank-Nicolson discretization is used in time for the Laplacian and 
the convected velocity. The convecting velocity is computed with an 
Adams-Bashforth projection. The fractional step method can be used
both non-iteratively or with iterations over the pressure velocity 
system.

The velocity vector is segregated, and we use three scalar velocity 
components

V = FunctionSpace(mesh, 'CG', 1)
u_components = ['u0', 'u1', 'u2'] in 3D, ['u0', 'u1'] in 2D
q_[ui] = Function(V) for ui = u_components
u_ = as_vector(q_['u0'], q_['u1'], q_['u2'])

A single coefficient matrix is assembled and used by all velocity 
componenets. It is built by preassembling as much as possible. 

The system of momentum equations solved are
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

The pressure gradient and body force needs to be added to b as well. Three
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
     nu/Sc*inner(grad(C), grad(v))*dx
     
where Sc is the constant Schmidt number. The scalar is using the same
FunctionSpace as the velocity components and it is computed by reusing 
much of the velocity matrices

    A_rhs = 1/dt*M - 0.5*Ac - 0.5*nu/Sc*K
    b['c']  = A_rhs*q_1['c'].vector()
    Ta = -A_rhs + 2/dt*M

and we solve:    

    Ta * c = b['c'] for q_['c'].vector()
  
Ta might differ from A due to Dirichlet boundary conditions.
    
"""
from common import *
from petsc4py import PETSc

################### Problem dependent parameters ####################
### Should import a mesh and a dictionary called NS_parameters    ###
### See common/default_hooks.py for possible parameters                   ###

default_problem = 'DrivenCavity'
commandline_kwargs = parse_command_line()
exec("from problems.{} import *".format(commandline_kwargs.get('problem', default_problem)))

assert(isinstance(NS_parameters, dict))
NS_parameters.update(commandline_kwargs)

# If the mesh is a callable function, then create the mesh here.
if callable(mesh):
    mesh = mesh(**NS_parameters)

assert(isinstance(mesh, Mesh))    
#####################################################################

if NS_parameters['velocity_degree'] > 1:
    NS_parameters['use_lumping_of_mass_matrix'] = False

# Put NS_parameters in global namespace
vars().update(NS_parameters)  
print_solve_info = use_krylov_solvers and krylov_solvers['monitor_convergence']

# Update dolfin parameters
parameters['krylov_solver'].update(krylov_solvers)

# Create lists of components solved for
dim = mesh.geometry().dim()
u_components = map(lambda x: 'u'+str(x), range(dim))
sys_comp =  u_components + ['p'] + scalar_components
uc_comp  =  u_components + scalar_components

# Set up initial folders for storing results
newfolder, tstepfiles = create_initial_folders(**vars())

# Declare FunctionSpaces and arguments
V = FunctionSpace(mesh, 'CG', velocity_degree, constrained_domain=constrained_domain)
Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree, constrained_domain=constrained_domain)
if velocity_degree == pressure_degree:
    Q = V
else:
    Q = FunctionSpace(mesh, 'CG', pressure_degree, constrained_domain=constrained_domain)
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Use dictionaries to hold all Functions and FunctionSpaces
VV = dict((ui, V) for ui in uc_comp); VV['p'] = Q

# Create dictionaries for the solutions at three timesteps
q_  = dict((ui, Function(VV[ui], name=ui)) for ui in sys_comp)
q_1 = dict((ui, Function(V, name=ui+"_1")) for ui in uc_comp)
q_2 = dict((ui, Function(V, name=ui+"_2")) for ui in u_components)

# Read in previous solution if restarting
init_from_restart(**vars())

# Create vectors of the segregated velocity components    
u_  = as_vector([q_[ui]  for ui in u_components]) # Velocity vector at t
u_1 = as_vector([q_1[ui] for ui in u_components]) # Velocity vector at t - dt
u_2 = as_vector([q_2[ui] for ui in u_components]) # Velocity vector at t - 2*dt

# Adams Bashforth projection of velocity at t - dt/2
U_AB = 1.5*u_1 - 0.5*u_2

# Create short forms for accessing the solution vectors
x_  = dict((ui, q_ [ui].vector()) for ui in sys_comp)     # Solution vectors t
x_1 = dict((ui, q_1[ui].vector()) for ui in uc_comp)      # Solution vectors t - dt
x_2 = dict((ui, q_2[ui].vector()) for ui in u_components) # Solution vectors t - 2*dt

# Create vectors to hold rhs of equations
b     = dict((ui, Vector(x_[ui])) for ui in sys_comp)       # rhs vectors (final)
b_tmp = dict((ui, Vector(x_[ui])) for ui in sys_comp)       # rhs temp storage vectors

# Short forms pressure and scalars
p_  = q_['p']               # pressure at t - dt/2
dp_ = Function(Q)           # pressure correction
for ci in scalar_components:
    exec("{}_   = q_ ['{}']".format(ci, ci))
    exec("{}_1  = q_1['{}']".format(ci, ci))

###################   Boundary conditions   #########################

bcs = create_bcs(**vars())

###################   Initialize solution   #########################

initialize(**vars())

###################  Fetch linear solvers   #########################

u_sol, p_sol, du_sol, c_sol = get_solvers(**vars())

################### Get constant body forces ########################

f = body_force(**vars())
assert(isinstance(f, Constant))

################### Preassemble and allocate ########################

if low_memory_version:
    P = None
    Rx = None
    
else:
    # Constant pressure gradient matrix
    P = dict((ui, assemble(v*p.dx(i)*dx)) for i, ui in enumerate(u_components))

    # Constant velocity divergence matrix
    if velocity_degree == pressure_degree:
        Rx = P
    else:
        Rx = dict((ui, assemble(q*u.dx(i)*dx)) for i, ui in  enumerate(u_components))

# Constant body force
b0 = dict((ui, assemble(v*f[i]*dx)) for i, ui in enumerate(u_components))

# Mass matrix
M = assemble(inner(u, v)*dx)

# Stiffness matrix (without viscosity coefficient)
K = assemble(inner(grad(u), grad(v))*dx)

# Pressure Laplacian
if velocity_degree == pressure_degree and bcs['p'] == []:
    Ap = K
    Mp = M
    
else:
    Ap = assemble(inner(grad(q), grad(p))*dx)
    [bc.apply(Ap) for bc in bcs['p']]
    Mp = assemble(inner(p, q)*dx)
    
# Allocate coefficient matrix (needs reassembling)
A = Matrix(M)

# Allocate coefficient matrix and work vectors for scalars. Matrix differs from velocity in boundary conditions only
if len(scalar_components) > 0:
    Ta = Matrix(M)                  
    Tb, bb, bx = None, None, None            
    if len(scalar_components) > 1:
        # For more than one scalar we use the same linear algebra solver for all.
        # For this to work we need some additional tensors. The extra matrix
        # is required since different scalars may have different boundary conditions
        Tb = Matrix(M)
        bb = Vector(x_[scalar_components[0]])
        bx = Vector(x_[scalar_components[0]])

#####################################################################

if parameters["form_compiler"].has_key("no_ferari"):
    parameters["form_compiler"].remove("no_ferari")

# Set convection form 
u_ab = as_vector([Function(V) for i in range(len(u_components))])
a_conv = 0.5*inner(v, dot(u_ab, nabla_grad(u)))*dx  # Faster version
#a_conv = 0.5*inner(v, dot(U_AB, nabla_grad(u)))*dx

for i, ui in enumerate(u_components):
    u_ab[i].vector().axpy(1.5, x_1[ui])
    u_ab[i].vector().axpy(-0.5, x_2[ui])

# A scalar always uses the Standard convection form
a_scalar = a_conv

lp = LocalAverageOperator(V)
dp = Function(V) 
#if velocity_degree == pressure_degree:
    #lpq = lp
#else:
    #lpq = LocalAverageOperator(Q)
lpq = LocalAverageOperator(Q)
#lpq = LocalAverageOperator(Q, Form(q*dx))
lpq.bcs = [homogenize(bc) for bc in bcs["u0"]]

#### Do something problem specific ####
vars().update(pre_solve_hook(**vars()))
#######################################
tic()
stop = False
total_timer = Timer("Full time integration")
info_blue("Start simulation")
while t < (T - tstep*DOLFIN_EPS) and not stop:
    complete_timestep = Timer("Complete timestep")
    t += dt
    tstep += 1
    inner_iter = 0
    err = 1e8
    num_iter = max(iters_on_first_timestep, max_iter) if tstep == 1 else max_iter
    #############################
    start_timestep_hook(**vars())
    #############################
    while err > max_error and inner_iter < num_iter:
        inner_iter += 1
        ### Assemble matrices and compute rhs vector for tentative velocity ###
        t0 = Timer("Tentative velocity")
        if inner_iter == 1:
            t1 = Timer("Assemble first inner iter")
            # Only on the first iteration because nothing here is changing in time
            # Set up coefficient matrix for computing the rhs:
            A = assemble(a_conv, tensor=A, reset_sparsity=False)
            A._scale(-1.)            # Negative convection on the rhs 
            A.axpy(1./dt, M, True)   # Add mass
            
            #Set up scalar matrix for rhs using the same convection as velocity
            if len(scalar_components) > 0:                
               if a_scalar is a_conv:
                   Ta.zero()
                   Ta.axpy(1., A, True)
                   
            # Add diffusion and compute rhs for all velocity components 
            A.axpy(-0.5*nu, K, True) 
            for ui in u_components:
                b[ui].zero()              # start with body force
                b[ui].axpy(1., b0[ui])
                b[ui].axpy(1., A*x_1[ui]) # Add transient, convection and diffusion
                
            # Reset matrix for lhs
            A._scale(-1.)
            A.axpy(2./dt, M, True)
            [bc.apply(A) for bc in bcs['u0']]
            t1.stop()
                   
        err = 0
        for i, ui in enumerate(u_components):
            info_blue('Solving tentative velocity '+ui, inner_iter == 1 and print_solve_info)
            b_tmp[ui].zero()
            b_tmp[ui].axpy(1., b[ui])
            add_pressure_gradient_rhs(**vars())
            #################################
            velocity_tentative_hook(**vars())
            #################################
            [bc.apply(b[ui]) for bc in bcs[ui]]
            x_2[ui].zero()                 # x_2 only used on inner_iter 1, so use here as work vector
            x_2[ui].axpy(1., x_[ui])
            t1 = Timer("Tentative Linear Algebra Solve")
            u_sol.solve(A, x_[ui], b[ui])
            t1.stop()
            b[ui].zero()
            b[ui].axpy(1., b_tmp[ui])
            err += norm(x_2[ui] - x_[ui])
        t0.stop()
        
        ### Solve pressure ###
        t0 = Timer("Pressure solve")
        info_blue('Solving pressure', inner_iter == 1 and print_solve_info)
        assemble_pressure_rhs(**vars())
        #######################
        pressure_hook(**vars())
        #######################
        [bc.apply(b['p']) for bc in bcs['p']]        
        solve_pressure(**vars())
                
        if num_iter > 1 and print_velocity_pressure_convergence:
            if inner_iter == 1: 
                info_blue('  Inner iterations velocity pressure:')
                info_blue('                 error u  error p')
            info_blue('    Iter = {0:4d}, {1:2.2e} {2:2.2e}'.format(inner_iter, err, norm(dp_.vector())))
        t0.stop()

    ### Update velocity if noniterative scheme is used ###
    if inner_iter == 1:
        t0 = Timer("Velocity update")
        update_velocity_lao(**vars())                
        t0.stop()
        
    # Solve for scalars
    if len(scalar_components) > 0:
        t0 = Timer("Scalar solve")
        if not a_scalar is a_conv:
            Ta = assemble(a_scalar, tensor=Ta, reset_sparsity=False)
            Ta._scale(-1.)            # Negative convection on the rhs 
            Ta.axpy(1./dt, M, True)   # Add mass
            
        # Compute rhs for all scalars
        for ci in scalar_components:
            Ta.axpy(-0.5*nu/Schmidt[ci], K, True) # Add diffusion
            b[ci].zero()                          # Compute rhs
            b[ci].axpy(1., Ta*x_1[ci])
            Ta.axpy(0.5*nu/Schmidt[ci], K, True)  # Subtract diffusion
        # Reset matrix for lhs - Note scalar matrix does not contain diffusion
        Ta._scale(-1.)
        Ta.axpy(2./dt, M, True)

        for ci in scalar_components:    
            info_blue('Solving scalar {}'.format(ci), print_solve_info)
            Ta.axpy(0.5*nu/Schmidt[ci], K, True) # Add diffusion
            #####################
            scalar_hook(**vars())
            #####################
            solve_scalar(**vars())
            Ta.axpy(-0.5*nu/Schmidt[ci], K, True) # Subtract diffusion
        t0.stop()
        
    ##############################
    temporal_hook(**vars())
    ##############################
    
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
        complete_timestep.stop()
        info_green('Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}'.format(t, tstep, T)) 
        info_red('Total computing time on previous {0:d} timesteps = {1:f}'.format(print_intermediate_info, toc()))
        list_timings(True)
        tic()
          
    if t < (T - tstep*DOLFIN_EPS) and not stop:
        # AB projection for pressure on next timestep
        if AB_projection_pressure:
            x_['p'].axpy(0.5, dp_.vector())

        # Update u_ used as convecting AB-velocity in convection assembly
        for i, ui in enumerate(u_components):
            #x_[ui]._scale(1.5)
            #x_[ui].axpy(-0.5, x_1[ui])
            u_ab[i].vector().zero()
            u_ab[i].vector().axpy(1.5, x_1[ui])
            u_ab[i].vector().axpy(-0.5, x_2[ui])
                            
total_timer.stop()
list_timings()
info_red('Total computing time = {0:f}'.format(total_timer.value()))
final_memory_use = dolfin_memory_usage('at end')
mymem = eval(final_memory_use)-eval(initial_memory_use)
print 'Additional memory use of processor = {0}'.format(mymem)
info_red('Total memory use of solver = ' + str(MPI.sum(mymem)))

###### Final hook ######        
theend(**vars())
########################
