__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-24'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, FunctionSpace, assemble, TestFunction, sym, grad,\
        dx, inner, as_backend_type, TrialFunction, project, CellVolume, sqrt,\
        solve, dot, lhs, rhs, interpolate, Constant, DirichletBC,FacetFunction,\
        KrylovSolver

__all__ = ['les_setup', 'les_update']

def les_setup(u_, mesh, KineticEnergySGS, assemble_matrix, CG1Function, nut_krylov_solver, bcs, **NS_namespace):
    """
    Set up for solving the Kinetic Energy SGS-model.
    """
    DG = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    dim = mesh.geometry().dim()
    
    delta = project(pow(CellVolume(mesh), 1./dim), DG)
    
    Ck = KineticEnergySGS["Ck"]
    Ce = KineticEnergySGS["Ce"]
    ksgs = interpolate(Constant(1E-7), CG1)
    bc_ksgs = DirichletBC(CG1, 0, "on_boundary")
    A_mass = assemble_matrix(TrialFunction(CG1)*TestFunction(CG1)*dx)
    ksgs_sol = KrylovSolver("bicgstab", "additive_schwarz")
    ksgs_sol.parameters["preconditioner"]["structure"] = "same_nonzero_pattern"
    ksgs_sol.parameters["error_on_nonconvergence"] = False
    ksgs_sol.parameters["monitor_convergence"] = False
    ksgs_sol.parameters["report"] = False
    
    nut_form = Ck * delta * sqrt(ksgs)
    # Create nut BCs
    ff = FacetFunction("size_t", mesh, 0)
    bcs_nut = []
    for i, bc in enumerate(bcs['u0']):
        bc.apply(u_[0].vector()) # Need to initialize bc
        m = bc.markers() # Get facet indices of boundary
        ff.array()[m] = i+1
        bcs_nut.append(DirichletBC(CG1, Constant(0), ff, i+1))
    nut_ = CG1Function(nut_form, mesh, method=nut_krylov_solver, bcs=bcs_nut, bounded=True, name="nut")

    return dict(nut_form=nut_form, nut_=nut_, delta=delta, ksgs=ksgs, ksgs_sol=ksgs_sol,
                CG1=CG1, A_mass=A_mass, bc_ksgs=bc_ksgs, bcs_nut=bcs_nut)    

def les_update(nut_, nut_form, A_mass, u_, dt, bc_ksgs, ksgs_sol,
        KineticEnergySGS, CG1, ksgs, delta, **NS_namespace):

    p, q = TrialFunction(CG1), TestFunction(CG1)

    Ck = KineticEnergySGS["Ck"]
    Ce = KineticEnergySGS["Ce"]

    Sij = sym(grad(u_))
    A = assemble(dt*inner(dot(u_,0.5*grad(p)), q)*dx \
            + inner((dt*Ce*sqrt(ksgs)/delta)*0.5*p,q)*dx \
            + inner(dt*Ck*sqrt(ksgs)*delta*grad(0.5*p),grad(q))*dx)
    b = A_mass*ksgs.vector() - A*ksgs.vector() + assemble(dt*2*Ck*delta*sqrt(ksgs)*inner(Sij,grad(u_))*q*dx)
    A.axpy(1.0, A_mass, True)

    # Solve for ksgs
    bc_ksgs.apply(A,b)
    ksgs_sol.solve(A, ksgs.vector(), b)
    ksgs.vector().set_local(ksgs.vector().array().clip(min=1e-7))
    ksgs.vector().apply("insert")

    # Update nut_
    nut_()
