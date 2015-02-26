__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, FunctionSpace, assemble, TestFunction, sym, grad,\
        dx, inner, as_backend_type, TrialFunction, project, CellVolume, sqrt,\
        solve

__all__ = ['les_setup', 'les_update']

def les_setup(u_, mesh, Smagorinsky, assemble_matrix, **NS_namespace):
    """
    Set up for solving Smagorinsky-Lilly LES model.
    """
    DG = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    dim = mesh.geometry().dim()
    
    delta = project(pow(CellVolume(mesh), 1./dim), DG)
    
    nut_ = Function(CG1)
    Sij = sym(grad(u_))
    magS = sqrt(2*inner(Sij,Sij))
    nut_form = Smagorinsky['Cs']**2 * delta**2 * magS
    A_nut = assemble_matrix(TrialFunction(CG1)*TestFunction(CG1)*dx)

    return dict(Sij=Sij, nut_form=nut_form, nut_=nut_, delta=delta,
                nut_test=TestFunction(CG1), A_nut=A_nut)    
    
def les_update(nut_, nut_form, nut_test, A_nut, **NS_namespace):
    # Solve for nut_
    solve(A_nut, nut_.vector(), assemble(nut_form*nut_test*dx), "cg", "default")
    # Remove negative values
    nut_.vector().set_local(nut_.vector().array().clip(min=0))
    nut_.vector().apply("insert")

