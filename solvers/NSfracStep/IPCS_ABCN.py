__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-06"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
from ..NSfracStep import *
from ..NSfracStep import __all__

def setup(low_memory_version, u_components, u, v, p, q, velocity_degree,
          bcs, scalar_components, V, Q, x_, dim, mesh,
          constrained_domain, velocity_update_type, **NS_namespace):
    """Preassemble mass and diffusion matrices. 
    
    Set up and prepare all equations to be solved. Called once, before 
    going into time loop.
    
    """    
    P = None
    Rx = None        
    if not low_memory_version:
        # Constant pressure gradient matrix
        P = dict((ui, assemble(v*p.dx(i)*dx)) for i, ui in enumerate(u_components))

        # Constant velocity divergence matrix
        if V == Q:
            Rx = P
        else:
            Rx = dict((ui, assemble(q*u.dx(i)*dx)) for i, ui in  enumerate(u_components))

    # Mass matrix
    M = assemble(inner(u, v)*dx)                    

    # Stiffness matrix (without viscosity coefficient)
    K = assemble(inner(grad(u), grad(v))*dx)        
    
    # Pressure Laplacian. Either reuse K or assemble new
    if V == Q and bcs['p'] == []:
        Ap = K
        
    else:
        Bp = assemble(inner(grad(q), grad(p))*dx) 
        [bc.apply(Bp) for bc in bcs['p']]
        Ap = Matrix()
        Bp.compressed(Ap)

    # Allocate coefficient matrix (needs reassembling)
    A = Matrix(M)

    # Create dictionary to be returned into global NS namespace
    d = dict(P=P, Rx=Rx, A=A, M=M, K=K, Ap=Ap)

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
    
    # Allocate for velocity update 
    if velocity_update_type.upper() == "GRADIENT_MATRIX":
        from fenicstools.WeightedGradient import weighted_gradient_matrix
        dP = weighted_gradient_matrix(mesh, range(dim), degree=velocity_degree, 
                                      constrained_domain=constrained_domain)
        d.update(dP=dP)
        
    elif velocity_update_type.upper() == "LUMPING":
        ones = Function(V)
        ones.vector()[:] = 1.
        ML = M * ones.vector()
        ML.set_local(1. / ML.array())
        d.update(ML=ML)
        
    else:
        Mu = Matrix(M) if len(scalar_components) > 0 else M # Copy if used by scalars
        [bc.apply(Mu) for bc in bcs['u0']]
        d.update(Mu=Mu)
    
    # Setup for solving convection
    u_ab = as_vector([Function(V) for i in range(len(u_components))])
    a_conv = 0.5*inner(v, dot(u_ab, nabla_grad(u)))*dx  # Faster version
    #a_conv = 0.5*inner(v, dot(U_AB, nabla_grad(u)))*dx
    a_scalar = a_conv    
    d.update(u_ab=u_ab, a_conv=a_conv, a_scalar=a_scalar)
    return d

def get_solvers(use_krylov_solvers, krylov_solvers, bcs, 
                x_, Q, scalar_components, velocity_update_type, 
                **NS_namespace):
    """Return linear solvers. 
    
    We are solving for
       - tentative velocity
       - pressure correction
       - velocity update (unless lumping is switched on)
       
       and possibly:       
       - scalars
            
    """
    if use_krylov_solvers:
        ## tentative velocity solver ##
        #u_sol = KrylovSolver('bicgstab', 'jacobi')
        u_sol = KrylovSolver('bicgstab', 'additive_schwarz')
        if "structure" in u_sol.parameters['preconditioner']:
            u_sol.parameters['preconditioner']['structure'] = "same_nonzero_pattern"
        else:
            u_sol.parameters['preconditioner']['reuse'] = False
            u_sol.parameters['preconditioner']['same_nonzero_pattern'] = True
        u_sol.parameters.update(krylov_solvers)
            
        ## velocity correction solver
        if velocity_update_type != "default":
            du_sol = None
        else:
            du_sol = KrylovSolver('bicgstab', 'additive_schwarz')
            if "structure" in du_sol.parameters['preconditioner']:
                du_sol.parameters['preconditioner']['structure'] = "same"
            else:
                du_sol.parameters['preconditioner']['reuse'] = True
            du_sol.parameters.update(krylov_solvers)
                
            du_sol.parameters['preconditioner']['ilu']['fill_level'] = 2
            #PETScOptions.set("pc_hypre_euclid_bj", True)
            #PETScOptions.set("pc_hypre_euclid_print_statistics", True)

        ## pressure solver ##
        #p_prec = PETScPreconditioner('petsc_amg')
        #p_prec.parameters['report'] = True
        #p_prec.parameters['same_nonzero_pattern'] = True
        #p_prec.parameters['gamg']['verbose'] = 20
        #p_prec.parameters['gamg']['num_aggregation_smooths'] = 2
        #p_sol = PETScKrylovSolver('gmres', p_prec)
        #p_sol.p_prec = p_prec
        if bcs['p'] == []:
            p_sol = KrylovSolver('cg', 'hypre_amg')
        else:
            p_sol = KrylovSolver('gmres', 'hypre_amg')
        
        if "structure" in p_sol.parameters['preconditioner']:
            p_sol.parameters['preconditioner']['structure'] = "same"
        else:
            p_sol.parameters['preconditioner']['reuse'] = True    
            
        p_sol.parameters.update(krylov_solvers)
        if bcs['p'] == []:
            attach_pressure_nullspace(p_sol, x_, Q)
        sols = [u_sol, p_sol, du_sol]
        ## scalar solver ##
        if len(scalar_components) > 0:
            #c_sol = KrylovSolver('bicgstab', 'hypre_euclid')
            c_sol = KrylovSolver('bicgstab', 'additive_schwarz')
            c_sol.parameters.update(krylov_solvers)
            
            if "structure" in c_sol.parameters['preconditioner']:
                c_sol.parameters['preconditioner']['structure'] = "same_nonzero_pattern"
            else:
                c_sol.parameters['preconditioner']['reuse'] = False
                c_sol.parameters['preconditioner']['same_nonzero_pattern'] = True    
            sols.append(c_sol)
        else:
            sols.append(None)
    else:
        ## tentative velocity solver ##
        u_sol = LUSolver("mumps")
        u_sol.parameters["same_nonzero_pattern"] = True
        ## velocity correction ##
        if velocity_update_type != "default":
            du_sol = None
        else:
            du_sol = LUSolver("mumps")
            du_sol.parameters['reuse_factorization'] = True
        ## pressure solver ##
        p_sol = LUSolver("mumps")
        p_sol.parameters['reuse_factorization'] = True
        if bcs['p'] == []:
            p_sol.normalize = True
        sols = [u_sol, p_sol, du_sol]
        ## scalar solver ##
        if len(scalar_components) > 0:
            c_sol = LUSolver("mumps")
            sols.append(c_sol)
        else:
            sols.append(None)
        
    return sols

def assemble_first_inner_iter(A, a_conv, dt, M, scalar_components,
                              a_scalar, K, nu, u_components,
                              b_tmp, b0, x_1, x_2, u_ab, bcs, **NS_namespace):
    """Called on first inner iteration of velocity/pressure system.
    
    Assemble convection matrix, compute rhs of tentative velocity and 
    reset coefficient matrix for solve.
    
    """
    t0 = Timer("Assemble first inner iter")
    # Update u_ab used as convecting velocity 
    for i, ui in enumerate(u_components):
        u_ab[i].vector().zero()
        u_ab[i].vector().axpy(1.5, x_1[ui])
        u_ab[i].vector().axpy(-0.5, x_2[ui])

    A = assemble(a_conv, tensor=A) 
    A._scale(-1.)            # Negative convection on the rhs 
    A.axpy(1./dt, M, True)   # Add mass
    
    #Set up scalar matrix for rhs using the same convection as velocity
    if len(scalar_components) > 0:      
        Ta = NS_namespace["Ta"]
        if a_scalar is a_conv:
            Ta.zero()
            Ta.axpy(1., A, True)
            
    # Add diffusion and compute rhs for all velocity components 
    A.axpy(-0.5*nu, K, True) 
    for ui in u_components:
        b_tmp[ui].zero()              # start with body force
        b_tmp[ui].axpy(1., b0[ui])
        b_tmp[ui].axpy(1., A*x_1[ui]) # Add transient, convection and diffusion
        
    # Reset matrix for lhs
    A._scale(-1.)
    A.axpy(2./dt, M, True)
    [bc.apply(A) for bc in bcs['u0']]
            
def attach_pressure_nullspace(p_sol, x_, Q):
    """Create null space basis object and attach to Krylov solver."""
    null_vec = Vector(x_['p'])
    Q.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0/null_vec.norm("l2")
    null_space = VectorSpaceBasis([null_vec])
    p_sol.set_nullspace(null_space)
    p_sol.null_space = null_space

def velocity_tentative_assemble(ui, i, b, b_tmp, P, x_, v, p_, u_sol, **NS_namespace):
    """Add pressure gradient to rhs of tentative velocity system."""
    b[ui].zero()
    b[ui].axpy(1., b_tmp[ui])
    if P:
        b[ui].axpy(-1., P[ui]*x_['p'])
    else:
        b[ui].axpy(-1., assemble(v*p_.dx(i)*dx))
        
def velocity_tentative_solve(ui, A, bcs, x_, x_2, u_sol, b, udiff, 
                             use_krylov_solvers, **NS_namespace):    
    """Linear algebra solve of tentative velocity component."""    
    if use_krylov_solvers:
        if ui == "u0":
            if "structure" in u_sol.parameters['preconditioner']:
                u_sol.parameters['preconditioner']['structure'] = "same_nonzero_pattern"
            else:
                u_sol.parameters['preconditioner']['reuse'] = False
                u_sol.parameters['preconditioner']['same_nonzero_pattern'] = True    
        else:
            if "structure" in u_sol.parameters['preconditioner']:
                u_sol.parameters['preconditioner']['structure'] = "same"
            else:
                u_sol.parameters['preconditioner']['reuse'] = True

    [bc.apply(b[ui]) for bc in bcs[ui]]
    x_2[ui].zero()                 # x_2 only used on inner_iter 1, so use here as work vector
    x_2[ui].axpy(1., x_[ui])
    t1 = Timer("Tentative Linear Algebra Solve")
    u_sol.solve(A, x_[ui], b[ui])
    t1.stop()
    udiff[0] += norm(x_2[ui] - x_[ui])

def pressure_assemble(b, Rx, x_, dt, q, u_, Ap, b_tmp, **NS_namespace):
    """Assemble rhs of pressure equation."""
    b['p'].zero()
    if Rx:
        for ui in Rx:
            b['p'].axpy(-1./dt, Rx[ui]*x_[ui])
            
    else:
        b['p'].axpy(-1./dt, assemble(div(u_)*q*dx))
    b['p'].axpy(1., Ap*x_['p'])

def pressure_solve(dp_, x_, Ap, b, p_sol, bcs, **NS_namespace):
    """Solve pressure equation."""
    [bc.apply(b['p']) for bc in bcs['p']]
    dp_.vector().zero()
    dp_.vector().axpy(1., x_['p'])
    # KrylovSolvers use nullspace for normalization of pressure
    if hasattr(p_sol, 'null_space'):
        p_sol.null_space.orthogonalize(b['p'])

    t1 = Timer("Pressure Linear Algebra Solve")
    p_sol.solve(Ap, x_['p'], b['p'])
    t1.stop()
    # LUSolver use normalize directly for normalization of pressure
    if hasattr(p_sol, 'normalize'):
        normalize(x_['p'])

    dp_.vector().axpy(-1., x_['p'])
    dp_.vector()._scale(-1.)

def velocity_update(u_components, b, bcs, print_solve_info, du_sol, P, 
                    dp_, dt, v, x_, info_blue, velocity_update_type, **NS_namespace):
    """Update the velocity after regular pressure velocity iterations."""
    if velocity_update_type.upper() == 'GRADIENT_MATRIX':
        dP = NS_namespace["dP"]
        for i, ui in enumerate(u_components):
            x_[ui].axpy(-dt, dP[i] * dp_.vector())
            [bc.apply(x_[ui]) for bc in bcs[ui]]
        
    elif velocity_update_type.upper() == "LUMPING":
        ML = NS_namespace["ML"]
        for i, ui in enumerate(u_components):
            if P:
                x_[ui].axpy(-dt, (P[ui] * dp_.vector()) * ML)
            else:
                x_[ui].axpy(-dt, (assemble(v*dp_.dx(i)*dx)) * ML)
            [bc.apply(x_[ui]) for bc in bcs[ui]]
            
    else:
        Mu = NS_namespace["Mu"]
        for i, ui in enumerate(u_components):
            b[ui].zero()
            b[ui].axpy(1.0, Mu*x_[ui])
            if P:
                b[ui].axpy(-dt, P[ui]*dp_.vector())
            else:
                b[ui].axpy(-dt, assemble(v*dp_.dx(i)*dx))

            [bc.apply(b[ui]) for bc in bcs[ui]]
            info_blue('Updating velocity '+ui, print_solve_info)
            t1 = Timer("Update Linear Algebra Solve")
            du_sol.solve(Mu, x_[ui], b[ui])
            t1.stop()
            
def scalar_assemble(a_scalar, a_conv, Ta , dt, M, scalar_components, 
                    nu, Schmidt, b, K, x_1, b0, **NS_namespace):
    """Assemble scalar equation."""    
    # Just in case you want to use a different scalar convection
    if not a_scalar is a_conv:
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

def scalar_solve(ci, scalar_components, Ta, b, x_, bcs, c_sol, 
                 nu, Schmidt, K, **NS_namespace):
    """Solve scalar equation."""
    
    Ta.axpy(0.5*nu/Schmidt[ci], K, True) # Add diffusion
    if len(scalar_components) > 1: 
        # Reuse solver for all scalars. This requires the same matrix and vectors to be used by c_sol.
        Tb, bb, bx = NS_namespace["Tb"], NS_namespace["bb"], NS_namespace["bx"]
        Tb.zero()
        Tb.axpy(1., Ta, True)
        bb.zero(); bb.axpy(1., b[ci])
        bx.zero(); bx.axpy(1., x_[ci])
        [bc.apply(Tb, bb) for bc in bcs[ci]]
        c_sol.solve(Tb, bx, bb)
        x_[ci].zero(); x_[ci].axpy(1., bx)
        
    else:
        [bc.apply(Ta, b[ci]) for bc in bcs[ci]]
        c_sol.solve(Ta, x_[ci], b[ci])    
    Ta.axpy(-0.5*nu/Schmidt[ci], K, True) # Subtract diffusion
    #x_[ci][x_[ci] < 0] = 0.               # Bounded solution
    #x_[ci].set_local(maximum(0., x_[ci].array()))
    #x_[ci].apply("insert")
