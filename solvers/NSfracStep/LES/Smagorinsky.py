__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, FunctionSpace, assemble, TestFunction, sym, grad, dx, inner, sqrt, \
    FacetFunction, DirichletBC, Constant

__all__ = ['les_setup', 'les_update']

def les_setup(u_, mesh, Smagorinsky, CG1Function, nut_krylov_solver, bcs, **NS_namespace):
    """
    Set up for solving Smagorinsky-Lilly LES model.
    """
    DG = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    dim = mesh.geometry().dim()
    delta = Function(DG)
    delta.vector().zero()
    delta.vector().axpy(1.0, assemble(TestFunction(DG)*dx))
    delta.vector().apply('insert')
    Sij = sym(grad(u_))
    magS = sqrt(2*inner(Sij,Sij))
    nut_form = Smagorinsky['Cs']**2 * delta**2 * magS
    nut_ = CG1Function(nut_form, mesh, method=nut_krylov_solver, bounded=True, name="nut")
    ff = FacetFunction("size_t", mesh, 0)
    bcs_nut = []
    for i, bc in enumerate(bcs['u0']):
        bc.apply(u_[0].vector()) # Need to initialize bc
        m = bc.markers() # Get facet indices of boundary
        ff.array()[m] = i+1
        bcs_nut.append(DirichletBC(CG1, Constant(0), ff, i+1))

    return dict(Sij=Sij, nut_=nut_, delta=delta, bcs_nut=bcs_nut)    
    
def les_update(nut_, **NS_namespace):
    """Compute nut_"""
    nut_()

