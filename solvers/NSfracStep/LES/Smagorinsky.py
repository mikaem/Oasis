__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, FunctionSpace, assemble, TestFunction, sym, grad, dx, inner, sqrt, \
    FacetFunction, DirichletBC, Constant

from common import derived_bcs

__all__ = ['les_setup', 'les_update']

def les_setup(U_AB, mesh, Smagorinsky, CG1Function, nut_krylov_solver, bcs, **NS_namespace):
    """
    Set up for solving traditional Smagorinsky-Lilly LES model.
    """
    
    dim = mesh.geometry().dim()
    delta = pow(CellVolume(mesh), 1./dim)

    Sij = sym(grad(U_AB))
    magS = sqrt(2*inner(Sij,Sij))
    nut_form = Smagorinsky['Cs']**2 * delta**2 * magS
    nut_ = CG1Function(nut_form, mesh, method=nut_krylov_solver, bcs=[],
            bounded=True, name="nut")

    return dict(Sij=Sij, nut_=nut_, delta=delta, bcs_nut=bcs_nut)   

def les_update(nut_, **NS_namespace):
    """Compute nut_"""
    nut_()

