__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-07"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""This is a simplest possible naive implementation of the IPCS solver.

The idea is that this solver can be quickly modified and tested for 
alternative implementations. In the end it can be used to validate
the implementations of the more complex optimized solvers.

"""
from dolfin import *

# declare all functions that must be imported by solver
__all__ = ["assemble_first_inner_iter", "tentative_velocity_assemble",
           "tentative_velocity_solve", "pressure_assemble", 
           "pressure_solve", "update_velocity", "scalar_assemble", 
           "scalar_solve", "get_solvers", "setup"]

def setup(ui, u, q_, q_1, uc_comp, u_components, dt, v, U_AB,
                nu, p_, dp_, mesh, f, fs, q, p, u_, Schmidt,
                scalar_components, **NS_namespace):
    """Set up all equations to be solved."""
    # Implicit Crank Nicholson velocity at t - dt/2
    U_CN = dict((ui, 0.5*(u+q_1[ui])) for ui in uc_comp)

    F = {}
    Fu = {}
    for i, ui in enumerate(u_components):
        # Tentative velocity step
        F[ui] = (1./dt)*inner(u - q_1[ui], v)*dx + inner(dot(U_AB, nabla_grad(U_CN[ui])), v)*dx + \
                nu*inner(grad(U_CN[ui]), grad(v))*dx + inner(p_.dx(i), v)*dx - inner(f[i], v)*dx
        
        # Velocity update
        Fu[ui] = inner(u, v)*dx - inner(q_[ui], v)*dx + dt*inner(dp_.dx(i), v)*dx

    # Pressure update
    Fp = inner(grad(q), grad(p))*dx - inner(grad(p_), grad(q))*dx + (1./dt)*div(u_)*q*dx 

    # Scalar with SUPG
    h = CellSize(mesh)
    vw = v + h*inner(grad(v), U_AB)
    n = FacetNormal(mesh)
    for ci in scalar_components:
        F[ci] = (1./dt)*inner(u - q_1[ci], vw)*dx + inner(dot(grad(U_CN[ci]), U_AB), vw)*dx \
                +nu/Schmidt[ci]*inner(grad(U_CN[ci]), grad(vw))*dx - inner(fs[ci], vw)*dx \
                #-nu/Schmidt[ci]*inner(dot(grad(U_CN[ci]), n), vw)*ds
    
    return dict(F=F, Fu=Fu, Fp=Fp)

def get_solvers(**NS_namespace):
    """Return 4 linear solvers. 
    
    We are solving for
       - tentative velocity
       - pressure correction
       - velocity update (unless lumping is switched on)
       
       and possibly:       
       - scalars
            
    """        
    return (None, )*4

def assemble_first_inner_iter(**NS_namespace):
    """Called first thing on a new velocity/pressure iteration."""
    pass

def tentative_velocity_assemble(**NS_namespace):
    """Assemble remaining system for tentative velocity component."""
    pass

def tentative_velocity_solve(ui, F, q_, bcs, x_, b_tmp, udiff, **NS_namespace):
    """Linear algebra solve of tentative velocity component."""
    b_tmp[ui][:] = x_[ui][:]
    solve(lhs(F[ui]) == rhs(F[ui]), q_[ui], bcs=bcs[ui])
    udiff[0] += norm(b_tmp[ui] - x_[ui])
    
def pressure_assemble(**NS_namespace):
    """Assemble rhs of pressure equation."""
    pass

def pressure_solve(Fp, p_, bcs, dp_, x_, **NS_namespace):
    """Solve pressure equation."""    
    dp_.vector()[:] = x_['p'][:]
    solve(lhs(Fp) == rhs(Fp), p_, bcs['p'])   
    if bcs['p'] == []:
        normalize(p_.vector())
    dp_.vector()[:] = x_['p'][:] - dp_.vector()[:]

def update_velocity(u_components, q_, bcs, Fu, **NS_namespace):
    """Update the velocity after finishing pressure velocity iterations."""
    for ui in u_components:
        solve(lhs(Fu[ui]) == rhs(Fu[ui]), q_[ui], bcs[ui])

def scalar_assemble(**NS_namespace):
    """Assemble scalar equation."""
    pass

def scalar_solve(ci, F, q_, bcs, **NS_namespace):
    """Solve scalar equation."""
    solve(lhs(F[ci]) == rhs(F[ci]), q_[ci], bcs[ci])
