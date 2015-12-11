__author__ = 'Mikael Mortensen <mikaem@math.uio.no>'
__date__ = '2015-01-22'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, FunctionSpace, assemble, TestFunction, sym, grad, tr, \
    Identity, dx, inner, Max, DirichletBC, Constant, dev

from common import derived_bcs

__all__ = ['les_setup', 'les_update']

def les_setup(U_AB, mesh, Wale, bcs, CG1Function, nut_krylov_solver, **NS_namespace):
    """
    Set up for solving the Wall-Adapting Local Eddy-Viscosity (WALE) LES model
    """
    
    dim = mesh.geometry().dim()
    delta = pow(CellVolume(mesh), 1./dim)
    
    gij = grad(U_AB)
    Sd = sym(gij*gij) - (1./dim)*Identity(dim)*tr(gij*gij)
    Sij = sym(grad(U_AB))
    
    nut_form = Wale['Cw']**2 * delta**2 * pow(inner(Sd, Sd), 1.5) / (pow(inner(Sij, Sij), 2.5) + pow(inner(Sd, Sd), 1.25))
    nut_ = CG1Function(nut_form, mesh, method=nut_krylov_solver, bcs=[], 
            name='nut', bounded=True)
    
    return dict(Sij=Sij, Sd=Sd, nut_=nut_, delta=delta, bcs_nut=bcs_nut)
    
def les_update(nut_, tstep, **NS_namespace):
    """Compute nut_"""
    if tstep > 1:
        nut_()
