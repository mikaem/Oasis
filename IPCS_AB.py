__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-10-05"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

"""
This is a highly tuned and stripped down Navier-Stokes solver optimized
for both speed and memory. The algorithm used is a second order in time 
fractional step method (incremental pressure correction).

Crank-Nicolson discretization is used in time of the Laplacian and 
convection is computed with an Adams-Bashforth projection. The fractional 
step method can be used both non-iteratively or with iterations over the 
pressure velocity system.

The velocity vector uses a VectorFunctionSpace

V = VectorFunctionSpace(mesh, 'CG', velocity_degree)
u_components = ['u'] in 2D and 3D
q_['u'] = Function(V)
u_ = q_['u']

A single coefficient matrix is built by preassembling. 

The system of momentum equations solved are
u = TrialFunction(V)
v = TestFunction(V)
U = 0.5*(u+u_)            
F = (1/dt)*inner(u - u_1, v)*dx + inner(1.5*grad(u_1)*u_1-0.5*grad(u_2)*u_2, v)*dx 
    - inner(grad(p_), v)*dx + nu*inner(grad(U), grad(v))*dx - inner(f, v)*dx

We solve an equation corresponding to lhs(F) == rhs(F).
     
The variables u_1 and u_2 are velocity vectors at time steps k-1 and k-2. We 
are solving for u, which is the velocity at time step k. p_ is the latest 
approximation for the pressure.

Two matrices are preassembled. The matrix corresponding to assemble(lhs(F)) is A_lhs 
and is the same for all velocity components. A_rhs is used to compute the 
corresponding terms for the previous timestep, that en up on the rhs of
the equation system

    A_lhs  = 1/dt*M + 0.5*nu*K
    A_rhs  = 1/dt*M - 0.5*nu*K
    b      = A_rhs*u_.vector()
    
where

    M  = assemble(inner(u, v)*dx)
    K  = assemble(inner(grad(u), grad(v))*dx)

The pressure gradient and body force needs to be added to b as well. One
matrix is preassembled for the computation of the pressure gradient:

  P = assemble(dot(v, grad(p))*dx)

and the pressure gradient for each component of the momentum equation is 
then computed as

   b += P * p_.vector()  (corresponding to += assemble(dot(grad(p_), v)*dx))

We then solve the linear system A_lhs * u = b for u_.vector()

Pressure is solved through

  inner(grad(p), grad(q))*dx == inner(grad(p_), grad(q))*dx - 1/dt*inner(div(u_), q)*dx

Here we assemble the rhs by:

  Ap = assemble(inner(grad(p), grad(q))*dx)
  bp = Ap * p_.vector()
  bp.axpy(-1./dt, Rx*u_.vector())
  where the preassembled Rx is:
    Rx = assemble(dot(div(u), q)*dx)
  
We then solve Ap * p = bp for p_.vector().
  
Velocity update is computed through:

  inner(u, v)*dx == inner(u_, v)*dx - dt*inner(grad(dp_), v)*dx

where the rhs of the equation is computed effectively as
  assemble(inner(u_, v)*dx) = M * u_.vector()
  assemble(dt*inner(grad(dp_), v)*dx) = dt * P * dp_.vector()

where dp_ is the pressure correction, i.e., th newly computed pressure 
at the new timestep minus the pressure used in computing the tentative velocity.

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
FunctionSpace as the velocity components (V.sub(0)).

"""
from common import *

################### Problem dependent parameters ####################
### Should import a mesh and a dictionary called NS_parameters    ###
### See common/default_hooks.py for possible parameters                   ###

default_problem = 'LshapeVector'
commandline_kwargs = parse_command_line()
exec("from problems.{}Vector import *".format(commandline_kwargs.get('problem', default_problem)))

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
u_components = ['u']
sys_comp =  u_components + ['p'] + scalar_components
uc_comp  =  u_components + scalar_components

# Update dolfin parameters
parameters['krylov_solver'].update(krylov_solvers)

# Set up initial folders for storing results
newfolder, tstepfiles = create_initial_folders(**vars())

# Declare solution Functions and FunctionSpaces
V = FunctionSpace(mesh, 'CG', velocity_degree, constrained_domain=constrained_domain)
Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree, constrained_domain=constrained_domain)
if velocity_degree == pressure_degree:
    Q = V
else:
    Q = FunctionSpace(mesh, 'CG', pressure_degree, constrained_domain=constrained_domain)
u = TrialFunction(Vv)
v = TestFunction(Vv)
c = TrialFunction(V)
d = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Use dictionaries to hold all Functions and FunctionSpaces
VV = dict((ci, V) for ci in scalar_components)
VV.update({'u': Vv, 'p': Q})

# Create dictionaries for the solutions at three timesteps
q_  = dict((ui, Function(VV[ui], name=ui)) for ui in sys_comp)
q_1 = dict((ui, Function(VV[ui], name=ui+"_1")) for ui in uc_comp)
q_2 = dict((ui, Function(VV[ui], name=ui+"_2")) for ui in u_components)

# Read in previous solution if restarting
init_from_restart(**vars())

# Velocity short forms    
u_  = q_ ['u'] # Velocity vector at t
u_1 = q_1['u'] # Velocity vector at t - dt
u_2 = q_2['u'] # Velocity vector at t - 2*dt

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

# Constant body force
b0 = assemble(inner(v, f)*dx)

# Constant pressure gradient matrix
if low_memory_version:
    P = None
    Rx = None
    
else:
    P = assemble(dot(v, grad(p))*dx)
    Rx = assemble(dot(div(u), q)*dx)

# Mass matrix
M = assemble(inner(u, v)*dx)                    

# Stiffness matrix (without viscosity coefficient)
K = assemble(inner(grad(u), grad(v))*dx)

# Pressure Laplacian
Ap = assemble(inner(grad(q), grad(p))*dx)
[bc.apply(Ap) for bc in bcs['p']]

# Allocate coefficient matrix
A_rhs = Matrix(M)
A_rhs._scale(1./dt)
A_rhs.axpy(-0.5*nu, K, True) # Add diffusion 
A_lhs = Matrix(M)
A_lhs._scale(1./dt)
A_lhs.axpy(0.5*nu, K, True) # Add diffusion 
[bc.apply(A_lhs) for bc in bcs['u']]
A_rhs.compress()
A_lhs.compress()
del K

# Allocate coefficient matrix and work vectors for scalars. Matrix differs from velocity in boundary conditions only
if len(scalar_components) > 0:
    Mc = assemble(inner(c, d)*dx) 
    Kc = assemble(inner(grad(c), grad(d))*dx)        
    Ta = Matrix(Mc)
    if len(scalar_components) > 1:
        # For more than one scalar we use the same linear algebra solver for all.
        # For this to work we need some additional tensors
        Tb = Matrix(Mc)
        bb = Vector(x_[scalar_components[0]])
        bx = Vector(x_[scalar_components[0]])

# Velocity update may use lumping of the mass matrix for P1-elements
# Compute inverse of the lumped diagonal mass matrix 
if use_lumping_of_mass_matrix:
    ML = assemble_lumped_P1_diagonal(**vars())    
    del M
    
else:  # Use regular mass matrix for velocity update
    Mu = M 
    [bc.apply(Mu) for bc in bcs['u']]

#####################################################################

# Set explicit convection form. Adams-Bashforth convection reads 
# 1.5*dot(grad(u_1), u_1)-0.5*dot(grad(u_2), u_2), but only dot(grad(u_1), u_1)
# requires reassembling each timestep
a_conv = inner(dot(grad(u_1), u_1), v)*dx
#a_conv = inner(1.5*dot(grad(u_1), u_1)-0.5*dot(grad(u_2), u_2), v)*dx # Old
b_conv = assemble(inner(dot(grad(u_2), u_2), v)*dx)

if parameters["form_compiler"].has_key("no_ferari"):
    parameters["form_compiler"].remove("no_ferari")

# A scalar always uses the Standard convection form
a_scalar = 0.5*inner(dot(grad(c), U_AB), d)*dx

#### Do something problem specific ####
vars().update(pre_solve_hook(**vars()))
#######################################
tic()
stop = False
total_timer = Timer("Full time integration")
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
        ### Assemble rhs vector for tentative velocity ###
        t0 = Timer("Tentative velocity")
        if inner_iter == 1:
            t1 = Timer("Assemble first inner iter")
                            
            b_tmp['u'].zero()
            b_tmp['u'].axpy(1., b_conv) # remember now inner(v, grad(u_2)*u_2)
            b_conv = assemble(a_conv, tensor=b_conv) 
            b['u'].zero()
            b['u'].axpy(-1.5, b_conv)
            b['u'].axpy(0.5, b_tmp['u'])
            #b['u'] = assemble(-a_conv, tensor=b['u'])  # old and slow
            
            b['u'].axpy(1.0, b0)                # start with body force
            b['u'].axpy(1., A_rhs*x_1['u'])     # Add transient and diffusion
                
            t1.stop()
                   
        info_blue('Solving tentative velocity ', inner_iter == 1 and print_solve_info)
        b_tmp['u'].zero()
        b_tmp['u'].axpy(1.0, b['u'])        
        add_pressure_gradient_rhs(**vars())
        #################################
        velocity_tentative_hook(**vars())
        #################################
        [bc.apply(b['u']) for bc in bcs['u']]
        x_2['u'].zero()
        x_2['u'].axpy(1., x_['u'])              # x_2 only used on inner_iter 1, so use here as work vector
        u_sol.solve(A_lhs, x_['u'], b['u'])
        b['u'].zero()
        b['u'].axpy(1., b_tmp['u'])
        err = norm(x_2['u'] - x_['u'])
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
        t0.stop()
                
        if num_iter > 1 and print_velocity_pressure_convergence:
            if inner_iter == 1: 
                info_blue('  Inner iterations velocity pressure:')
                info_blue('                 error u  error p')
            info_blue('    Iter = {0:4d}, {1:2.2e} {2:2.2e}'.format(inner_iter, err, norm(dp_.vector())))

    ### Update velocity if noniterative scheme is used ###
    if inner_iter == 1:
        t0 = Timer("Velocity update")
        if use_lumping_of_mass_matrix:
            update_velocity_lumping(**vars())
            
        else: # Use regular mass matrix
            b['u'].zero()
            b['u'].axpy(1., Mu*x_['u'])      
            add_pressure_gradient_rhs_update(**vars())
            
            ##############################
            velocity_update_hook(**vars())
            ##############################
            [bc.apply(b['u']) for bc in bcs['u']]
            info_blue('Updating velocity ', print_solve_info)
            du_sol.solve(Mu, x_['u'], b['u'])
        t0.stop()
        
    # Solve for scalars
    if len(scalar_components) > 0:
        t0 = Timer("Scalar solve")
        
        # Compute rhs for all scalars
        Ta.zero()
        Ta.axpy(1./dt, Mc, True)
        for ci in scalar_components:
            Ta.axpy(-0.5*nu/Schmidt[ci], Kc, True) # Add diffusion
            b[ci] = assemble(-a_conv[ci], tensor=b[ci])
            b[ci].axpy(1., Ta*x_1[ci])
            Ta.axpy(0.5*nu/Schmidt[ci], Kc, True)  # Subtract diffusion

        for ci in scalar_components:    
            info_blue('Solving scalar {}'.format(ci), print_solve_info)
            Ta.axpy(0.5*nu/Schmidt[ci], Kc, True) # Add diffusion
            #####################
            scalar_hook(**vars())
            #####################
            solve_scalar(**vars())
            Ta.axpy(-0.5*nu/Schmidt[ci], Kc, True) # Subtract diffusion
        t0.stop()        
        
    ##############################
    temporal_hook(**vars())
    ##############################
    
    # Save solution if required and check for killoasis file
    stop = save_solution(**vars())
    
    # Update to a new timestep
    x_2['u'].zero()
    x_2['u'].axpy(1., x_1['u'])
    x_1['u'].zero()
    x_1['u'].axpy(1., x_['u'])
        
    for ci in scalar_components:
        x_1[ci].zero()
        x_1[ci].axpy(1., x_[ci])
        
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
