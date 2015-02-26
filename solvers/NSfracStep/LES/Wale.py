__author__ = 'Mikael Mortensen <mikaem@math.uio.no>'
__date__ = '2015-01-22'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, FunctionSpace, assemble, TestFunction, sym, grad, tr, \
    Identity, dx, inner, Max, homogenize

__all__ = ['les_setup', 'les_update']

def les_setup(u_, mesh, Wale, assemble_matrix, bcs, OasisFunction, **NS_namespace):
    """Set up for solving Wale LES model
    """
    DG = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    delta = Function(DG)
    delta.vector().zero()
    delta.vector().axpy(1.0, assemble(TestFunction(DG)*dx))
    Gij = grad(u_)
    Sij = sym(Gij)
    Skk = tr(Sij)
    dim = mesh.geometry().dim()
    Sd = sym(Gij*Gij) - 1./3.*Identity(mesh.geometry().dim())*Skk*Skk 
    nut_form = Wale['Cw']**2 * pow(delta, 2./dim) * pow(inner(Sd, Sd), 1.5) / (Max(pow(inner(Sij, Sij), 2.5) + pow(inner(Sd, Sd), 1.25), 1e-6))
    nut_ = OasisFunction(nut_form, CG1)
    return dict(Sij=Sij, Sd=Sd, Skk=Skk, nut_=nut_, delta=delta)
    
def les_update(nut_, **NS_namespace):
    # Solve for nut_
    nut_()
    # Remove negative values
    nut_.vector().set_local(nut_.vector().array().clip(min=0))
    nut_.vector().apply("insert")

