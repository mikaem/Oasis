__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, FunctionSpace, assemble, TestFunction, sym, grad, dx, inner, sqrt

__all__ = ['les_setup', 'les_update']

def les_setup(u_, mesh, Smagorinsky, assemble_matrix, **NS_namespace):
    """
    Set up for solving Smagorinsky-Lilly LES model.
    """
    DG = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    dim = mesh.geometry().dim()
    delta = Function(DG)
    delta.vector().zero()
    delta.vector().axpy(1.0, assemble(TestFunction(DG)*dx))
    Sij = sym(grad(u_))
    magS = sqrt(2*inner(Sij,Sij))
    nut_form = Smagorinsky['Cs']**2 * delta**2 * magS
    nut_ = OasisFunction(nut_form, CG1)
    return dict(Sij=Sij, nut_=nut_, delta=delta)    
    
def les_update(nut_, **NS_namespace):
    # Solve for nut_
    nut_()
    # Remove negative values
    nut_.vector().set_local(nut_.vector().array().clip(min=0))
    nut_.vector().apply("insert")

