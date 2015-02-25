__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-24'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, FunctionSpace, assemble, TestFunction, sym, grad,\
        dx, inner, as_backend_type, TrialFunction, project, CellVolume, sqrt,\
        solve, dot, lhs, rhs, interpolate, Constant, DirichletBC, plot,\
        interactive

__all__ = ['les_setup', 'les_update']

def les_setup(u_, mesh, KineticEnergySGS, assemble_matrix, **NS_namespace):
    """
    Set up for solving the Kinetic Energy SGS-model.
    """
    DG = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    dim = mesh.geometry().dim()
    
    delta = project(pow(CellVolume(mesh), 1./dim), DG)
    
    Ck = KineticEnergySGS["Ck"]
    Ce = KineticEnergySGS["Ce"]

    nut_ = Function(CG1)
    ksgs = interpolate(Constant(1E-7), CG1)
    nut_form = Ck * delta * sqrt(ksgs)
    A_nut = assemble_matrix(TrialFunction(CG1)*TestFunction(CG1)*dx)
    bc_ksgs = DirichletBC(CG1, 0, "on_boundary")

    return dict(nut_form=nut_form, nut_=nut_, delta=delta, ksgs=ksgs,
                CG1=CG1, A_nut=A_nut, bc_ksgs=bc_ksgs)    
    
def les_update(nut_, nut_form, A_nut, u_, dt, bc_ksgs,
        KineticEnergySGS, CG1, ksgs, delta, **NS_namespace):

    p, q = TrialFunction(CG1), TestFunction(CG1)

    Ck = KineticEnergySGS["Ck"]
    Ce = KineticEnergySGS["Ce"]
    
    Sij = sym(grad(u_))
    A = assemble(dt*inner(dot(u_,0.5*grad(p)), q)*dx \
            + inner((dt*Ce*sqrt(ksgs)/delta)*0.5*p,q)*dx \
            + inner(dt*Ck*sqrt(ksgs)*delta*grad(0.5*p),grad(q))*dx)
    b = A_nut*ksgs.vector() - A*ksgs.vector() + assemble(dt*2*Ck*delta*sqrt(ksgs)*inner(Sij,grad(u_))*q*dx)
    A.axpy(1.0, A_nut, True)
    
    # Solve for ksgs
    bc_ksgs.apply(A,b)
    solve(A, ksgs.vector(), b, "bicgstab", "additive_schwarz")
    ksgs.vector().set_local(ksgs.vector().array().clip(min=1e-7))
    ksgs.vector().apply("insert")
    
    # Solve for nut_
    solve(A_nut, nut_.vector(), assemble(nut_form*q*dx), "cg", "default")
    # Remove negative values
    nut_.vector().set_local(nut_.vector().array().clip(min=0))
    nut_.vector().apply("insert")

