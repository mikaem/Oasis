__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, FunctionSpace, assemble, TestFunction, sym, grad,\
        dx, inner, as_backend_type, TrialFunction, project, CellVolume, sqrt,\
        TensorFunctionSpace, assign, solve, lhs, rhs, LagrangeInterpolator,\
        dev, outer, as_vector, FunctionAssigner, KrylovSolver, plot,\
        interactive
import numpy as np

__all__ = ['les_setup', 'les_update']

def les_setup(u_, mesh, dt, krylov_solvers, **NS_namespace):
    """
    Set up for solving the Germano Dynamic LES model applying
    Lagrangian Averaging.
    """
    DG = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    TFS = TensorFunctionSpace(mesh, "CG", 1, symmetry=True)
    dim = mesh.geometry().dim()
    if dim == 2:
        tensdim = 3
    else:
        tensdim = 6

    delta = project(pow(CellVolume(mesh), 1./dim), DG)
    ll = LagrangeInterpolator()

    # Define nut
    nut_ = Function(DG)
    Sij = sym(grad(u_))
    magS = sqrt(2*inner(Sij,Sij))
    Cs = Function(CG1)
    nut_form = Cs**2 * delta**2 * magS
    A_dg = as_backend_type(assemble(TrialFunction(DG)*TestFunction(DG)*dx))
    dg_diag = A_dg.mat().getDiagonal().array

    # Define velocity helpers
    u_CG1 = as_vector([Function(CG1) for i in range(dim)])
    u_filtered = as_vector([Function(CG1) for i in range(dim)])
    dummy = Function(CG1)

    # Assemble required filter matrices and functions
    G_under = Function(CG1, assemble(TestFunction(CG1)*dx))
    G_under.vector().set_local(1./G_under.vector().array())
    G_under.vector().apply("insert")
    # G_matr is also the mass matrix to be used with Lagrangian avg.
    G_matr = assemble(TrialFunction(CG1)*TestFunction(CG1)*dx)

    # Assemble some required matrices for solving for rate of strain terms
    A_TFS = assemble(inner(TrialFunction(TFS), TestFunction(TFS))*dx)
    F_uiuj = Function(TFS)
    F_SSij = Function(TFS)

    # Set up function assigners
    # From TFS.sub(i) to CG1
    assigners = [FunctionAssigner(CG1, TFS.sub(i)) for i in range(tensdim)]
    # From CG1 to TFS.sub(i)
    assigners_rev = [FunctionAssigner(TFS.sub(i), CG1) for i in range(tensdim)]
    
    # Define Lagrangian solver
    lag_sol = KrylovSolver("bicgstab", "jacobi")
    lag_sol.parameters.update(krylov_solvers)
    lag_sol.parameters['preconditioner']['structure'] = 'same_nonzero_pattern'

    # Set up Lagrange Equations
    JLM = Function(CG1)
    JLM.vector()[:] += 1e-7
    JMM = Function(CG1)
    JMM.vector()[:] += 10.
    JQN = Function(CG1)
    JQN.vector()[:] += 1e-7
    JNN = Function(CG1)
    JNN.vector()[:] += 10.
    eps = Function(CG1)
    T_ = project(1.5*delta, CG1)
    T_.vector().set_local(dt/T_.vector().array())
    T_.vector().apply("insert")
    
    return dict(Sij=Sij, nut_form=nut_form, nut_=nut_, delta=delta,
                dg_diag=dg_diag, DG=DG, CG1=CG1, v_dg=TestFunction(DG),
                Cs=Cs, u_CG1=u_CG1, u_filtered=u_filtered, A_TFS=A_TFS, 
                TFS=TFS, F_uiuj=F_uiuj, F_SSij=F_SSij, JLM=JLM, JMM=JMM, 
                JQN=JQN, JNN=JNN, eps=eps, T_=T_, dim=dim, tensdim=tensdim,
                G_matr=G_matr, G_under=G_under, ll=ll, dummy=dummy, 
                assigners=assigners, assigners_rev=assigners_rev, 
                lag_sol=lag_sol)
    
def les_update(u_, nut_, nut_form, v_dg, dg_diag, dt, 
            CG1, delta, tstep, DynamicSmagorinsky, Cs, 
            u_CG1, u_filtered, A_TFS, TFS, F_uiuj, F_SSij, 
            JLM, JMM, JQN, JNN, eps, T_, dim, tensdim, G_matr, 
            G_under, ll, dummy, assigners, assigners_rev, lag_sol, 
            **NS_namespace):

    """
    For the dynamic model Cs needs to be recomputed for the wanted
    time intervals.
    """
    # Check if Cs is to be computed, if not update nut_ and break
    if tstep%DynamicSmagorinsky["Cs_comp_step"] != 0:
        ##################
        # Solve for nut_ #
        ##################
        nut_.vector().set_local(assemble(nut_form*v_dg*dx).array()/dg_diag)
        nut_.vector().apply("insert")
        # BREAK FUNCTION
        return

    # Ratio between filters, such that delta_tilde = 2delta,
    # where delta is the implicit mesh filter.
    alpha = 2

    #############################
    # Filter the velocity field #
    #############################
    # All velocity components must be interpolated to CG1
    # then filtered
    for i in xrange(dim):
        # Interpolate to CG1
        ll.interpolate(u_CG1[i], u_[i])
        # Filter
        tophatfilter(u_CG1[i], u_filtered[i], 1, G_matr, G_under, dummy)

    ##############
    # SET UP Lij #
    ##############
    # Compute outer product of uiuj
    compute_Fuiuj(u_CG1, F_uiuj, dim, dummy, G_matr, G_under, assigners_rev)
    # Define Lij = dev(F(uiuj)-F(ui)F(uj))
    Lij = dev(F_uiuj - outer(u_filtered, u_filtered))

    ##############
    # SET UP Mij #
    ##############
    # Compute |S|Sij
    compute_magSSij(as_vector(u_CG1), F_SSij, A_TFS, TFS)
    # Compute F(|S|Sij)
    tophatfilter(F_SSij, F_SSij, tensdim, G_matr, G_under, dummy,
            assigners=assigners, assigners_rev=assigners_rev)
    # Define F(Sij)
    Sijf = dev(sym(grad(u_filtered)))
    # Define F(|S|) = sqrt(2*Sijf:Sijf)
    magSf = sqrt(2*inner(Sijf,Sijf))
    # Define Mij = 2*delta**2(F(|S|Sij) - alpha**2F(|S|)F(Sij))
    Mij = 2*(delta**2)*(F_SSij - (alpha**2)*magSf*Sijf)

    ##################################################
    # Solve Lagrange Equations for LijMij and MijMij #
    ##################################################
    lagrange_average(eps, T_, JLM, JMM, Lij, Mij, u_, dt, G_matr, dummy, CG1,
            lag_sol)

    # Now u needs to be filtered once more
    for i in xrange(dim):
        # Filter
        tophatfilter(u_filtered[i], u_filtered[i], 1, G_matr, G_under, dummy)
    
    ###################################################
    # SET UP Qij = dev(F(F(uiuj)) - F(F(ui))F(F(uj))) #
    ##################################################
    # Filter F(uiuj) --> F(F(uiuj))
    tophatfilter(F_uiuj, F_uiuj, tensdim, G_matr, G_under, dummy,
            assigners=assigners, assigners_rev=assigners_rev)
    # Define Qij
    Qij = dev(F_uiuj - outer(u_filtered, u_filtered))
    
    ##############
    # SET UP Nij #
    ##############
    # F(|S|Sij) has all ready been computed, filter once more
    # Compute F(F(|S|Sij))
    tophatfilter(F_SSij, F_SSij, tensdim, G_matr, G_under, dummy, 
            assigners=assigners, assigners_rev=assigners_rev)
    # Define F(Sij)
    Sijf = dev(sym(grad(u_filtered)))
    # Define F(|S|) = sqrt(2*Sijf:Sijf)
    magSf = sqrt(2*inner(Sijf,Sijf))
    # Define Mij = 2*delta**2(F(|S|Sij) - alpha**2F(|S|)F(Sij))
    Nij = 2*(delta**2)*(F_SSij - (alpha**4)*magSf*Sijf)

    ##################################################
    # Solve Lagrange Equations for QijNij and NijNij #
    ##################################################
    lagrange_average(eps, T_, JQN, JNN, Qij, Nij, u_, dt, G_matr, dummy, CG1,
            lag_sol)

    #################################
    # UPDATE Cs**2 = (JLM*JMM)/beta #
    # beta = JQN/JNN                #
    #################################
    beta = JQN.vector().array()/JNN.vector().array()
    beta = beta.clip(min=0.125)
    Cs.vector().set_local(np.sqrt((JLM.vector().array()/JMM.vector().array())/\
            beta))
    Cs.vector().apply("insert")

    ##################
    # Solve for nut_ #
    ##################
    nut_.vector().set_local(assemble(nut_form*v_dg*dx).array()/dg_diag)
    nut_.vector().apply("insert")

def lagrange_average(eps, T_, J1, J2, Aij, Bij, u_, dt, A_,
        dummy, CG1, lag_sol):
    """
    Function for Lagrange Averaging two tensors
    AijBij and BijBij, PDE's are solved implicitly.

    d/dt(J1) + u*grad(J2) = 1/T(AijBij - J1)
    d/dt(J2) + u*grad(J1) = 1/T(BijBij - J2)

    Cs**2 = J1/J2

    - eps = dt/T and epsT are computed using fast array operations.
    - The convective term is assembled.
    - Lhs A is computed by axpying the mass term to the convective term.
    - Right hand sided are assembled.
    - Two equations are solved applying pre-defined krylov solvers.
    - J1 is clipped at 1E-32 (not zero, will lead to problems).
    - J2 is clipped at 10 (initial value).
    """

    # Update eps vector = dt/T
    eps.vector().set_local(
            ((J1.vector().array()*J2.vector().array())**0.125)\
            *T_.vector().array())
    epsT = dummy
    # Update epsT to dt/(1+dt/T)
    epsT.vector().set_local(dt/(1.+eps.vector().array()))
    epsT.vector().apply("insert")
    # Update eps to (dt/T)/(1+dt/T)
    eps.vector().set_local(eps.vector().array()/(1+eps.vector().array()))
    eps.vector().apply("insert")

    p, q = TrialFunction(CG1), TestFunction(CG1)
    # Assemble convective term
    A = assemble(-inner(epsT*u_*p, grad(q))*dx)
    # Axpy mass matrix
    A.axpy(1, A_, True)
    # Assemble right hand sides
    b1 = A_*J1.vector() + assemble(inner(eps*inner(Aij,Bij),q)*dx)
    b2 = A_*J2.vector() + assemble(inner(eps*inner(Bij,Bij),q)*dx)
    
    # Solve for J1 and J2, apply pre-defined krylov solver
    lag_sol.solve(A, J1.vector(), b1)
    lag_sol.solve(A, J2.vector(), b2)
    
    # Apply ramp function on J1 to remove negative values,
    # but not set to 0.
    J1.vector().set_local(J1.vector().array().clip(\
            min=1E-32))
    J1.vector().apply("insert")
    # Apply ramp function on J2 too; bound at initial value
    J2.vector().set_local(J2.vector().array().clip(\
            min=10))
    J2.vector().apply("insert")

def tophatfilter(unfiltered, filtered, N, G_matr, G_under, dummy,
        assigners=None, assigners_rev=None):
    """
    Filtering function for applying a generalized top hat filter.
    uf = int(G*u)/int(G).

    G = CG1-basis functions.
    u = unfiltered
    uf = filtered

    All functions must be in CG1!
    """

    uf = dummy
    
    if N > 1:
        code_1 = "assigners[i].assign(uf, filtered.sub(i))"
        code = "assigners_rev[i].assign(filtered.sub(i), uf)"
    else:
        code_1 = "uf.vector()[:] = unfiltered.vector()"
        code = "filtered.vector()[:] = uf.vector()"

    for i in xrange(N):
        exec(code_1)
        uf.vector()[:] = (G_matr*uf.vector())*G_under.vector()
        exec(code)

def compute_Fuiuj(u, F_uiuj, dim, dummy, G_matr, G_under, assigners):
    """
    Manually compute the term

    F(uiuj)

    and assign to tensor.

    The terms uiuj are computed, then the filter function
    is called for each term.
    """

    # Extract velocity components
    u0 = u[0]
    u1 = u[1]
    # Check if case is 2D or 3D
    if dim == 3:
        # 3D case
        # Extract z-velocity as well
        u2 = u[2]
        # u*u
        dummy.vector()[:] = u0.vector()*u0.vector()
        tophatfilter(dummy, dummy, 1, G_matr, G_under, dummy)
        assigners[0].assign(F_uiuj.sub(0), dummy)
        # u*v
        dummy.vector()[:] = u0.vector()*u1.vector()
        tophatfilter(dummy, dummy, 1, G_matr, G_under, dummy)
        assigners[1].assign(F_uiuj.sub(1), dummy)
        # u*w
        dummy.vector()[:] = u0.vector()*u2.vector()
        tophatfilter(dummy, dummy, 1, G_matr, G_under, dummy)
        assigners[2].assign(F_uiuj.sub(2), dummy)
        # v*v
        dummy.vector()[:] = u1.vector()*u1.vector()
        tophatfilter(dummy, dummy, 1, G_matr, G_under, dummy)
        assigners[3].assign(F_uiuj.sub(3), dummy)
        # v*w
        dummy.vector()[:] = u1.vector()*u2.vector()
        tophatfilter(dummy, dummy, 1, G_matr, G_under, dummy)
        assigners[4].assign(F_uiuj.sub(4), dummy)
        # w*w
        dummy.vector()[:] = u2.vector()*u2.vector()
        tophatfilter(dummy, dummy, 1, G_matr, G_under, dummy)
        assigners[5].assign(F_uiuj.sub(5), dummy)
    else:
        # 2D case
        # u*u
        dummy.vector()[:] = u0.vector()*u0.vector()
        tophatfilter(dummy, dummy, 1, G_matr, G_under, dummy)
        assigners[0].assign(F_uiuj.sub(0), dummy)
        # u*v
        dummy.vector()[:] = u0.vector()*u1.vector()
        tophatfilter(dummy, dummy, 1, G_matr, G_under, dummy)
        assigners[1].assign(F_uiuj.sub(1), dummy)
        # v*v
        dummy.vector()[:] = u1.vector()*u1.vector()
        tophatfilter(dummy, dummy, 1, G_matr, G_under, dummy)
        assigners[2].assign(F_uiuj.sub(2), dummy)

def compute_magSSij(u, F_SSij, A_TFS, TFS):
    """
    Solve for 
    
    sqrt(2*inner(Sij,Sij))*Sij
    
    applying a pre-assembled mass matrix for
    the TensorFunctionSpace.
    """

    # Define form for Sij
    Sij = dev(sym(grad(u)))
    # Assemble right hand side
    b = assemble(inner(sqrt(2*inner(Sij,Sij))*Sij,TestFunction(TFS))*dx)
    # Solve linear system for |S|Sij
    solve(A_TFS, F_SSij.vector(), b, "bicgstab", "additive_schwarz")
