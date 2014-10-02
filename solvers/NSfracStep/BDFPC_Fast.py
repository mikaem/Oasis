__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-07"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""This is a simplest possible naive implementation of a backwards
differencing solver with pressure correction in rotational form.

The idea is that this solver can be quickly modified and tested for 
alternative implementations. In the end it can be used to validate
the implementations of the more complex optimized solvers.

"""
from dolfin import *
from IPCS_ABCN import * # reuse code from IPCS_ABCN
from IPCS_ABCN import __all__

def setup(low_memory_version, u_components, u, v, p, q, velocity_degree,
          bcs, scalar_components, V, Q, x_, dim, mesh, u_, q_,
          constrained_domain, velocity_update_type, 
          OasisFunction, **NS_namespace):
    """Set up all equations to be solved."""
    P = None
    Rx = None
    Hx = None
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
    
    divu = OasisFunction(div(u_), Q, matvec=(Rx, q_), name="divu")

    # Create dictionary to be returned into global NS namespace
    d = dict(P=P, Rx=Rx, A=A, M=M, K=K, Ap=Ap, divu=divu)

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
    u_convecting = as_vector([Function(V) for i in range(len(u_components))])
    a_conv = inner(v, dot(u_convecting, nabla_grad(u)))*dx  # Faster version
    a_scalar = inner(v, dot(u_, nabla_grad(u)))*dx
    d.update(u_convecting=u_convecting, a_conv=a_conv, a_scalar=a_scalar)
    return d

def assemble_first_inner_iter(A, a_conv, dt, M, scalar_components,
                              a_scalar, K, nu, u_components,
                              b_tmp, b0, x_1, x_2, u_convecting, 
                              bcs, **NS_namespace):
    """Called on first inner iteration of velocity/pressure system.
    
    Assemble convection matrix, compute rhs of tentative velocity and 
    reset coefficient matrix for solve.
    
    """
    t0 = Timer("Assemble first inner iter")
    # Update u_convecting used as convecting velocity 
    for i, ui in enumerate(u_components):
        u_convecting[i].vector().zero()
        u_convecting[i].vector().axpy( 2.0, x_1[ui])
        u_convecting[i].vector().axpy(-1.0, x_2[ui])

    assemble(a_conv, tensor=A) 
    
    #Set up scalar matrix for rhs using the same convection as velocity
    if len(scalar_components) > 0:      
        Ta = NS_namespace["Ta"]
        if a_scalar is a_conv:
            Ta.zero()
            Ta.axpy(1., A, True)
        else:
            assemble(a_scalar, tensor=Ta)
            
    # Compute rhs for all velocity components 
    for ui in u_components:
        b_tmp[ui].zero()              # start with body force
        b_tmp[ui].axpy(1., b0[ui])
        b_tmp[ui].axpy(2.0/dt, M*x_1[ui]) 
        b_tmp[ui].axpy(-0.5/dt, M*x_2[ui]) 
        
    A.axpy(nu, K, True)
    A.axpy(1.5/dt, M, True)    
    [bc.apply(A) for bc in bcs['u0']]
         
def pressure_assemble(b, Rx, x_, dt, q, Ap, b_tmp, divu, **NS_namespace):
    """Assemble rhs of pressure equation."""
    b["p"].zero()
    divu()
    b["p"][:] = divu.rhs
    b["p"]._scale(-1.5/dt)
         
def pressure_solve(dp_, x_, Ap, b, p_sol, bcs, nu, divu, Q, **NS_namespace):
    """Solve pressure equation."""    
    [bc.apply(b['p']) for bc in bcs['p']]
    dp_.vector().zero()
    # KrylovSolvers use nullspace for normalization of pressure
    if hasattr(p_sol, 'null_space'):
        p_sol.null_space.orthogonalize(b['p'])

    t1 = Timer("Pressure Linear Algebra Solve")
    p_sol.solve(Ap, dp_.vector(), b['p'])
    t1.stop()
    # LUSolver use normalize directly for normalization of pressure
    if hasattr(p_sol, 'normalize'):
        normalize(dp_.vector())

    x_["p"].axpy(1, dp_.vector())
    x_["p"].axpy(-nu, divu.vector())
    dp_.vector()._scale(2./3.) # To reuse code from IPCS_ABCN

def scalar_assemble(a_scalar, a_conv, Ta , dt, M, scalar_components, 
                    nu, Schmidt, b, K, x_1, b0, **NS_namespace):
    """Assemble scalar equation."""    
    # Just in case you want to use a different scalar convection
    #if not a_scalar is a_conv:
        #Ta = assemble(a_scalar, tensor=Ta)
        #Ta._scale(-1.)            # Negative convection on the rhs 
        #Ta.axpy(1./dt, M, True)   # Add mass
        
    ## Compute rhs for all scalars
    #for ci in scalar_components:
        #Ta.axpy(-0.5*nu/Schmidt[ci], K, True) # Add diffusion
        #b[ci].zero()                          # Compute rhs
        #b[ci].axpy(1., Ta*x_1[ci])
        #b[ci].axpy(1., b0[ci])
        #Ta.axpy(0.5*nu/Schmidt[ci], K, True)  # Subtract diffusion
    ## Reset matrix for lhs - Note scalar matrix does not contain diffusion
    #Ta._scale(-1.)
    #Ta.axpy(2./dt, M, True)

def scalar_solve(ci, scalar_components, Ta, b, x_, bcs, c_sol, 
                 nu, Schmidt, K, **NS_namespace):
    """Solve scalar equation."""
    
    #Ta.axpy(0.5*nu/Schmidt[ci], K, True) # Add diffusion
    #if len(scalar_components) > 1: 
        ## Reuse solver for all scalars. This requires the same matrix and vectors to be used by c_sol.
        #Tb, bb, bx = NS_namespace["Tb"], NS_namespace["bb"], NS_namespace["bx"]
        #Tb.zero()
        #Tb.axpy(1., Ta, True)
        #bb.zero(); bb.axpy(1., b[ci])
        #bx.zero(); bx.axpy(1., x_[ci])
        #[bc.apply(Tb, bb) for bc in bcs[ci]]
        #c_sol.solve(Tb, bx, bb)
        #x_[ci].zero(); x_[ci].axpy(1., bx)
        
    #else:
        #[bc.apply(Ta, b[ci]) for bc in bcs[ci]]
        #c_sol.solve(Ta, x_[ci], b[ci])    
    #Ta.axpy(-0.5*nu/Schmidt[ci], K, True) # Subtract diffusion
    ##x_[ci][x_[ci] < 0] = 0.               # Bounded solution
    ##x_[ci].set_local(maximum(0., x_[ci].array()))
    ##x_[ci].apply("insert")
