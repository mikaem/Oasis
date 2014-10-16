__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-06"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
from IPCS_ABCN import *
from IPCS_ABCN import __all__, attach_pressure_nullspace

docstrings = {func: eval(func+".__doc__") for func in __all__}

def setup(u_components, u, v, p, q,
          bcs, scalar_components, V, Q, x_, U_AB, A_cache,
          velocity_update_solver, u_, u_1, u_2, p_, assemble_matrix,
          GradFunction, DivFunction, **NS_namespace):    
    """Preassemble mass and diffusion matrices. 
    
    Set up and prepare all equations to be solved. Called once, before 
    going into time loop.
    
    """
    # Mass matrix
    M = assemble_matrix(inner(u, v)*dx)                    

    # Stiffness matrix (without viscosity coefficient)
    K = assemble_matrix(inner(grad(u), grad(v))*dx)        
    
    # Pressure Laplacian. Either reuse K or assemble new
    Ap = assemble_matrix(inner(grad(q), grad(p))*dx, bcs['p'])

    if not Ap.id() == K.id():
        # Compress matrix (creates new matrix)
        Bp = Matrix()
        Ap.compressed(Bp)
        Ap = Bp
        # Replace cached matrix with compressed version
        key = (inner(grad(q), grad(p))*dx, tuple(bcs['p']))
        A_cache[key] = (Ap, A_cache[key][1])

    # Allocate coefficient matrix (needs reassembling)
    A = Matrix(M)

    # Allocate Function for holding and computing the velocity divergence on Q
    divu = DivFunction(u_, Q, name='divu', 
                       method=velocity_update_solver)

    # Allocate a dictionary of Functions for holding and computing pressure gradients
    gradp = {ui: GradFunction(p_, V, i=i, name='dpd'+('x','y','z')[i],
                              method=velocity_update_solver) 
                              for i, ui in enumerate(u_components)}

    # Create dictionary to be returned into global NS namespace
    d = dict(A=A, M=M, K=K, Ap=Ap, divu=divu, gradp=gradp)

    # Allocate coefficient matrix and work vectors for scalars. Matrix differs from velocity in boundary conditions only
    if len(scalar_components) > 0:
        d.update(Ta=Matrix(M))
        if len(scalar_components) > 1:
            # For more than one scalar we use the same linear algebra solver for all.
            # For this to work we need some additional tensors. The extra matrix
            # is required since different scalars may have different boundary conditions
            Tb = Matrix(M)
            bb = Vector(x_[scalar_components[0]])
            bx = Vector(x_[scalar_components[0]])
            d.update(Tb=Tb, bb=bb, bx=bx)    
        
    # Setup for solving convection
    a_conv = inner(v, dot(u_1, nabla_grad(u)))*dx
    A_conv = assemble(inner(v, dot(u_2, nabla_grad(u)))*dx)

    # A scalar always uses the Standard convection form
    a_scalar = None
    if len(scalar_components) > 0:
        a_scalar = 0.5*inner(v, dot(grad(u), U_AB))*dx

    d.update(a_conv=a_conv, A_conv=A_conv, a_scalar=a_scalar)
    
    return d

def get_solvers(use_krylov_solvers, krylov_solvers, sys_comp, bcs, x_, 
                Q, scalar_components, velocity_update_type, **NS_namespace):
    """Return linear solvers. 
    
    We are solving for
       - tentative velocity
       - pressure correction
       
       and possibly:       
       - scalars
            
    """
    if use_krylov_solvers:
        ## tentative velocity solver ##
        u_sol = KrylovSolver('bicgstab', 'additive_schwarz')
        u_sol.parameters['preconditioner']['structure'] = 'same'
        u_sol.parameters.update(krylov_solvers)
            
        ## pressure solver ##
        if bcs['p'] == []:
            p_sol = KrylovSolver('cg', 'hypre_amg')
        else:
            p_sol = KrylovSolver('gmres', 'hypre_amg')
            
        p_sol.parameters['preconditioner']['structure'] = 'same'
        p_sol.parameters.update(krylov_solvers)
        if bcs['p'] == []:
            attach_pressure_nullspace(p_sol, x_, Q)
        sols = [u_sol, p_sol]
        ## scalar solver ##
        if len(scalar_components) > 0:
            c_sol = KrylovSolver('bicgstab', 'additive_schwarz')
            c_sol.parameters['preconditioner']['structure'] = 'same_nonzero_pattern'
            c_sol.parameters.update(krylov_solvers)
            sols.append(c_sol)
        else:
            sols.append(None)
    else:
        ## tentative velocity solver ##
        u_sol = LUSolver('mumps')
        u_sol.parameters['reuse_factorization'] = True
        ## pressure solver ##
        p_sol = LUSolver('mumps')
        p_sol.parameters['reuse_factorization'] = True
        if bcs['p'] == []:
            p_sol.normalize = True
        sols = [u_sol, p_sol]
        ## scalar solver ##
        if len(scalar_components) > 0:
            c_sol = LUSolver('mumps')
            sols.append(c_sol)
        else:
            sols.append(None)
        
    return sols

def assemble_first_inner_iter(A, dt, M, nu, K, b0, b_tmp, A_conv, x_2, x_1,
                              a_conv, u_components, bcs, **NS_namespace):
    t0 = Timer("Assemble first inner iter")
    A.zero()
    A.axpy(1./dt, M, True)
    A.axpy(-0.5*nu, K, True) # Add diffusion 

    for ui in u_components:
        b_tmp[ui].zero()
        b_tmp[ui].axpy(1.0, b0[ui])                 # body force
        b_tmp[ui].axpy(0.5, A_conv*x_2[ui])
        
    A_conv = assemble(a_conv, tensor=A_conv)
    A.axpy(-1.5, A_conv, True)
    for ui in u_components:            
        b_tmp[ui].axpy(1.0, A*x_1[ui])              # Add transient and diffusion
                    
    A.axpy(nu, K, True)       # Reset for lhs
    A.axpy(1.5, A_conv, True) # Remove convection    
    [bc.apply(A) for bc in bcs['u0']]

def velocity_tentative_solve(ui, A, bcs, x_, x_2, u_sol, b, udiff, 
                             **NS_namespace):
    """Linear algebra solve of tentative velocity component."""    
    [bc.apply(b[ui]) for bc in bcs[ui]]
    x_2[ui].zero()                 # x_2 only used on inner_iter 1, so use here as work vector
    x_2[ui].axpy(1., x_[ui])
    t1 = Timer("Tentative Linear Algebra Solve")
    u_sol.solve(A, x_[ui], b[ui])
    t1.stop()
    udiff[0] += norm(x_2[ui] - x_[ui])
            
def scalar_assemble(Ta, a_scalar, dt, M, scalar_components, 
                    b, nu, Schmidt, K, x_1, b0, **NS_namespace):
    Ta = assemble(a_scalar, tensor=Ta)
    Ta._scale(-1.)            # Negative convection on the rhs 
    Ta.axpy(1./dt, M, True)   # Add mass
    
    # Compute rhs for all scalars
    for ci in scalar_components:
        Ta.axpy(-0.5*nu/Schmidt[ci], K, True) # Add diffusion
        b[ci].zero()                          # Compute rhs
        b[ci].axpy(1., Ta*x_1[ci])
        b[ci].axpy(1., b0[ci])
        Ta.axpy(0.5*nu/Schmidt[ci], K, True)  # Subtract diffusion
    # Reset matrix for lhs - Note scalar matrix does not contain diffusion
    Ta._scale(-1.)
    Ta.axpy(2./dt, M, True)
    
# Reuse docstrings from IPCS_ABCN if not defined here    
for func in __all__:
    doc = eval("{}.__doc__".format(func))
    if doc is None:
        exec("""{}.__doc__ = docstrings["{}"]""".format(func, func))
        
    
