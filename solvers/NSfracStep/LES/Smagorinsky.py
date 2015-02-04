__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, FunctionSpace, assemble, TestFunction, sym, grad,\
        dx, inner, as_backend_type, TrialFunction, project, CellVolume, sqrt
import numpy as np

__all__ = ['les_setup', 'les_update']

def les_setup(u_, mesh, Smagorinsky, **NS_namespace):
    """
    Set up for solving Smagorinsky-Lilly LES model.
    """
    DG = FunctionSpace(mesh, "DG", 0)
    dim = mesh.geometry().dim()
    
    delta = project(pow(CellVolume(mesh), 1./dim), DG)
    
    nut_ = Function(DG)
    Sij = sym(grad(u_))
    magS = sqrt(2*inner(Sij,Sij))
    nut_form = Smagorinsky['Cs']**2 * delta**2 * magS

    A_dg = as_backend_type(assemble(TrialFunction(DG)*TestFunction(DG)*dx))
    dg_diag = A_dg.mat().getDiagonal().array
    
    return dict(Sij=Sij, nut_form=nut_form, nut_=nut_, delta=delta,
                dg_diag=dg_diag, DG=DG, v_dg=TestFunction(DG))    
    
def les_update(nut_, nut_form, v_dg, dg_diag, **NS_namespace):
    nut_.vector().set_local(assemble(nut_form*v_dg*dx).array()/dg_diag)
    nut_.vector().apply("insert")
