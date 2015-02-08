__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import TrialFunction, TestFunction, assemble, inner, dx, grad,\
                    plot, interactive, solve
import numpy as np
import time

def lagrange_average(eps, T_, u_, dt, G_matr, dummy, CG1, lag_sol, 
        bcJ1, bcJ2, J1=None, J2=None, Aij=None, Bij=None, **NS_namespace):
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
    eps.vector().set_local(((J1.vector().array()*J2.vector().array())**0.125)*T_.vector().array())
    eps.vector().apply("insert")
    epsT = dummy
    # Update epsT to 1/(1+dt/T)
    epsT.vector().set_local(1./(1.+eps.vector().array()))
    epsT.vector().apply("insert")
    # Update eps to (dt/T)/(1+dt/T)
    eps.vector().set_local(eps.vector().array()/(1+eps.vector().array()))
    eps.vector().apply("insert")
    
    p, q = TrialFunction(CG1), TestFunction(CG1)
    # Assemble convective term
    A = assemble(-inner(dt*epsT*u_*p, grad(q))*dx)
    # Axpy mass matrix
    A.axpy(1, G_matr, True)
    # Assemble right hand sides
    b1 = assemble(inner(epsT*J1 + eps*inner(Aij,Bij),q)*dx)
    b2 = assemble(inner(epsT*J2 + eps*inner(Bij,Bij),q)*dx)

    # Solve for J1 and J2, apply pre-defined krylov solver
    bcJ1.apply(A, b1)
    lag_sol.solve(A, J1.vector(), b1)
    bcJ2.apply(A, b2)
    lag_sol.solve(A, J2.vector(), b2)

    # Apply ramp function on J1 to remove negative values,
    # but not set to 0.
    J1.vector().set_local(J1.vector().array().clip(min=1E-32))
    J1.vector().apply("insert")
    # Apply ramp function on J2 too; bound at initial value
    J2.vector().set_local(J2.vector().array().clip(min=1))
    J2.vector().apply("insert")

def tophatfilter(G_matr, G_under, dummy,
        assigners, assigners_rev, unfiltered=None, filtered=None,
        N=1, **NS_namespace):
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
        vec_ = (G_matr*uf.vector())*G_under.vector()
        uf.vector().zero()
        uf.vector().axpy(1,vec_)
        exec(code)

def compute_uiuj(F_uiuj, uiuj_pairs, tensdim, dummy, G_matr, G_under,
        assigners_rev, u=None, **NS_namespace):
    """
    Manually compute the term

    F(uiuj)

    and assign to tensor.

    The terms uiuj are computed, then the filter function
    is called for each term. 
    """
    
    # Loop over each tensor component
    for i in xrange(tensdim):
        # Extract velocity pair
        j,k = uiuj_pairs[i]
        dummy.vector().zero()
        # Add ujuk to dummy
        dummy.vector().axpy(1.0, u[j].vector()*u[k].vector())
        # Assign to tensor
        assigners_rev[i].assign(F_uiuj.sub(i), dummy)

def compute_magSSij(F_SSij, G_matr, CG1, dim, tensdim, assigners_rev, Sijforms,
        Sijcomps, lag_sol, u_=None, **NS_namespace):
    """
    Solve for 
    
    sqrt(2*inner(Sij,Sij))*Sij
    
    componentwise by applying a pre-assembled CG1
    mass matrix, and pre-assembled derivative matrices
    Ax, Ay and Az. Array operations are applied
    when removing the trace and computing |S|
    """
    Sij = Sijcomps
    
    # Apply pre-assembled matrices and compute right hand sides
    if tensdim == 3:
        Ax, Ay = Sijforms
        u = u_[0].vector()
        v = u_[1].vector()
        b = [2*Ax*u, Ay*u + Ax*v, 2*Ay*v]
    else:
        Ax, Ay, Az = Sijforms
        u = u_[0].vector()
        v = u_[1].vector()
        w = u_[2].vector()
        b = [2*Ax*u, Ay*u + Ax*v, Az*u + Ay*w, 2*Ay*v, Az*v + Ay*w, 2*Az*w]

    # First we need to solve for the different components of Sij
    for i in xrange(tensdim):
        solve(G_matr, Sij[i].vector(), 0.5*b[i], "cg", "default")

    # Compute |S| = sqrt(2*Sij:Sij)
    if tensdim == 3:
        # Extract Sij vectors
        S00 = Sij[0].vector().array()
        S01 = Sij[1].vector().array()
        S11 = Sij[2].vector().array()
        # Compute |S|
        magS = np.sqrt(2*(S00*S00 + 2*S01*S01 + S11*S11))
    else:
        # Extract Sij vectors
        S00 = Sij[0].vector().array()
        S01 = Sij[1].vector().array()
        S02 = Sij[2].vector().array()
        S11 = Sij[3].vector().array()
        S12 = Sij[4].vector().array()
        S22 = Sij[5].vector().array()
        # Compute |S|
        magS = np.sqrt(2*(S00*S00 + 2*S01*S01 + 2*S02*S02 + S11*S11 +
            2*S12*S12+ S22*S22))

    # Multiply each component of Sij by magS and assign to F_SSij
    for i in xrange(tensdim):
        Sij[i].vector().set_local(Sij[i].vector().array()*magS)
        Sij[i].vector().apply("insert")
        assigners_rev[i].assign(F_SSij.sub(i), Sij[i])
