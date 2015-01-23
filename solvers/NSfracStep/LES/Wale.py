__author__ = 'Mikael Mortensen <mikaem@math.uio.no>'
__date__ = '2015-01-22'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, FunctionSpace, assemble, TestFunction, sym, grad, tr, \
    elem_mult, Identity, dx, inner, as_backend_type, TrialFunction, Max

def les_setup(u_, mesh, Wale, **NS_namespace):
    """Set up for solving Wale LES model
    """
    DG = FunctionSpace(mesh, "DG", 0)
    delta = Function(DG)
    delta.vector().zero()
    delta.vector().axpy(1.0, assemble(TestFunction(DG)*dx))
    nut_ = Function(DG)
    Sij = sym(grad(u_))
    Skk= tr(Sij)
    Sd = 0.5*(elem_mult(Sij, Sij) + elem_mult(Sij.T, Sij.T)) - 1./3.*Identity(mesh.geometry().dim())*elem_mult(Skk, Skk) 
    nut_form = pow(Wale['Cw'], 2)*elem_mult(delta, delta)*pow(inner(Sd, Sd), 1.5) / (Max(pow(inner(Sij, Sij), 2.5) + pow(inner(Sd, Sd), 1.25), 1e-6))
    A_dg = as_backend_type(assemble(TrialFunction(DG)*TestFunction(DG)*dx))
    dg_diag = A_dg.mat().getDiagonal().array
    
    return dict(Sij=Sij, Sd=Sd, Skk=Skk, nut_form=nut_form, nut_=nut_, delta=delta,
                dg_diag=dg_diag, DG=DG, v_dg=TestFunction(DG))    
    
def les_nut(nut_, nut_form, v_dg, dg_diag, **NS_namespace):
    nut_.vector().set_local(assemble(nut_form*v_dg*dx).array()/dg_diag)
    
    
    