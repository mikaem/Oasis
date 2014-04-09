__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-04"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
from ..NSCoupled import *
from ..NSCoupled import __all__

def setup(u_, p_, up_, up, u, p, v, q, nu, **NS_namespace):
    """Set up all equations to be solved."""
    F = inner(dot(grad(u_), u_), v)*dx + nu*inner(grad(u_), grad(v))*dx \
      - inner(p_, div(v))*dx - inner(q, div(u_))*dx

    J    = derivative(F, up_, up)
    A = Matrix()

    return dict(F=F, J=J, A=A)

def NS_assemble(A, J, bcs, **NS_namespace):
    A = assemble(J, tensor=A)
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
