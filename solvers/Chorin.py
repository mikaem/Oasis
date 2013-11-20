__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-20"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""This is a simplest possible naive implementation of the Chorin solver.

The idea is that this solver can be quickly modified and tested for 
alternative implementations. In the end it can be used to validate
the implementations of the more complex optimized solvers.

"""
from dolfin import *
from IPCS import *

# declare all functions that must be imported by solver
__all__ = ["assemble_first_inner_iter", "tentative_velocity_assemble",
           "tentative_velocity_solve", "pressure_assemble", 
           "pressure_solve", "update_velocity", "scalar_assemble", 
           "scalar_solve", "get_solvers", "setup", "max_iter",
           "iters_on_first_timestep"]

# Chorin is noniterative
max_iter = 1
iters_on_first_timestep = 1

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
                nu*inner(grad(U_CN[ui]), grad(v))*dx - inner(f[i], v)*dx
        
        # Velocity update
        Fu[ui] = inner(u, v)*dx - inner(q_[ui], v)*dx + dt*inner(p_.dx(i), v)*dx

    # Pressure solve
    Fp = inner(grad(q), grad(p))*dx + (1./dt)*div(u_)*q*dx 

    # Scalar with SUPG
    h = CellSize(mesh)
    vw = v + h*inner(grad(v), U_AB)
    n = FacetNormal(mesh)
    for ci in scalar_components:
        F[ci] = (1./dt)*inner(u - q_1[ci], vw)*dx + inner(dot(grad(U_CN[ci]), U_AB), vw)*dx \
                +nu/Schmidt[ci]*inner(grad(U_CN[ci]), grad(vw))*dx - inner(fs[ci], vw)*dx
    
    return dict(F=F, Fu=Fu, Fp=Fp)
