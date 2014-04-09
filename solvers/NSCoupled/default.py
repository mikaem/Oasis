__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-04"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
from ..NSCoupled import *
from ..NSCoupled import __all__

def setup(u_, p_, up_, up, u, p, v, q, nu, **NS_namespace):
    """Set up all equations to be solved."""
    F_nonlinear = inner(dot(grad(u_), u_), v)*dx
    F_linear    = nu*inner(grad(u_), grad(v))*dx - inner(p_, div(v))*dx \
                  - inner(q, div(u_))*dx

    F = F_linear + F_nonlinear
    J_linear    = derivative(F_linear, up_, up)
    J_nonlinear = derivative(F_nonlinear, up_, up)

    A_pre = assemble(J_linear)
    A = Matrix(A_pre)

    return dict(F_linear=F_linear, F_nonlinear=F_nonlinear, 
                J_linear=J_linear, J_nonlinear=J_nonlinear, 
                A_pre=A_pre, A=A, F=F)

def NS_assemble(A, J_nonlinear, A_pre, bcs, **NS_namespace):
    A = assemble(J_nonlinear, tensor=A)
    A.axpy(1.0, A_pre, True)
    for bc in bcs["up"]:
        bc.apply(A)
    
def NS_solve(A, up_1, b, omega, up_, F, bcs, up_sol, 
             **NS_namespace):
    up_1.vector().zero()
    up_sol.solve(A, up_1.vector(), b["up"])
    up_.vector().axpy(-omega, up_1.vector())

    b["up"] = assemble(F, tensor=b["up"])
    for bc in bcs["up"]:
        bc.apply(b["up"], up_.vector())

def scalar_assemble(**NS_namespace):
    """Assemble scalar equation."""
    pass

def scalar_solve(**NS_namespace):
    """Solve scalar equation."""
    pass
