__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import TrialFunction, TestFunction, assemble, inner, dx, grad,\
                    plot, interactive, solve, project
import numpy as np

def lagrange_average(eps, T_, u_, dt, G_matr, dummy, CG1, lag_sol, 
        bcJ1, bcJ2, Sijcomps, Sijfcomps, assigners, tensdim, J1=None, J2=None, 
        Aij=None, Bij=None, **NS_namespace):
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
    eps = ((J1.vector().array()*J2.vector().array())**0.125)*T_.vector().array()
    # Update eps to (dt/T)/(1+dt/T)
    eps = eps/(1+eps)
    
    J1_back = J1.vector().array()
    J2_back = J2.vector().array()
    # Compute tensor contractions
    AijBij = tensor_inner(A=Aij, B=Bij, **vars())
    BijBij = tensor_inner(A=Bij, B=Bij, **vars())

    J1.vector().set_local(eps*AijBij + (1-eps)*J1_back)
    J1.vector().apply("insert")
    J2.vector().set_local(eps*BijBij + (1-eps)*J2_back)
    J2.vector().apply("insert")

    # Apply ramp function on J1 to remove negative values,
    # but not set to 0.
    J1.vector().set_local(J1.vector().array().clip(min=1E-32))
    J1.vector().apply("insert")
    ## Apply ramp function on J2 too; bound at initial value
    J2.vector().set_local(J2.vector().array().clip(min=10))
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
        uf.vector().axpy(1.0, vec_)
        exec(code)

def compute_Lij(Lij, dummyTFS, uiuj_pairs, tensdim, dummy, G_matr, G_under,
        assigners_rev, assigners, u=None, uf=None, **NS_namespace):
    """
    Manually compute the term

    F(uiuj)

    and assign to tensor.

    The terms uiuj are computed, then the filter function
    is called for each term. 
    """
    trace = np.zeros(len(dummy.vector().array()))
    # Loop over each tensor component
    for i in xrange(tensdim):
        # Extract velocity pair
        j, k = uiuj_pairs[i]
        dummy.vector().zero()
        # Add ujuk to dummy
        dummy.vector().axpy(1.0, u[j].vector()*u[k].vector())
        # Filter dummy -> F(ujuk)
        tophatfilter(unfiltered=dummy, filtered=dummy, **vars())
        # Axpy - F(uj)F(uk)
        dummy.vector().axpy(-1.0, uf[j].vector()*uf[k].vector())
        # Assign to Lij
        assigners_rev[i].assign(Lij.sub(i), dummy)

        # Add to trace if j == k
        if j == k:
            trace += dummy.vector().array()
    
    # The deviatoric part of Lij must now be removed
    if tensdim == 3:
        # Compute trace and add to dummy
        trace = 0.5*trace
        dummy.vector().set_local(trace)
        dummy.vector().apply("insert")
        # Assign the trace to the diagonal of dummyTFS
        assigners_rev[0].assign(dummyTFS.sub(0), dummy)
        assigners_rev[2].assign(dummyTFS.sub(2), dummy)
        # Axpy trace from Lij
        Lij.vector().axpy(-1.0, dummyTFS.vector())
    else:
        # Compute trace and add to dummy
        trace = 1./3.*trace
        dummy.vector().set_local(trace)
        dummy.vector().apply("insert")
        # Assign the trace to the diagonal of dummyTFS
        assigners_rev[0].assign(dummyTFS.sub(0), dummy)
        assigners_rev[3].assign(dummyTFS.sub(3), dummy)
        assigners_rev[5].assign(dummyTFS.sub(5), dummy)
        # Axpy trace from Lij
        Lij.vector().axpy(-1.0, dummyTFS.vector())

def compute_Mij(Mij, G_matr, CG1, dim, tensdim, assigners_rev, Sijmats,
        Sijcomps, Sijfcomps, delta, dt, T_, alphaval=None, u_nf=None, u_f=None, **NS_namespace):
    """
    Solve for 
    
    sqrt(2*inner(Sij,Sij))*Sij
    
    componentwise by applying a pre-assembled CG1
    mass matrix, and pre-assembled derivative matrices
    Ax, Ay and Az. Array operations are applied
    when removing the trace and computing |S|
    """

    Sij = Sijcomps
    Sijf = Sijfcomps
    alpha = alphaval
    deltasq = 2*(dt/(1.5*T_.vector().array()))**2
    
    # Apply pre-assembled matrices and compute right hand sides
    if tensdim == 3:
        Ax, Ay = Sijmats
        u = u_nf[0].vector()
        v = u_nf[1].vector()
        uf = u_f[0].vector()
        vf = u_f[1].vector()
        bu = [2*Ax*u, Ay*u + Ax*v, 2*Ay*v]
        buf = [2*Ax*uf, Ay*uf + Ax*vf, 2*Ay*vf]
    else:
        Ax, Ay, Az = Sijmats
        u = u_nf[0].vector()
        v = u_nf[1].vector()
        w = u_nf[2].vector()
        uf = uf_f[0].vector()
        vf = uf_f[1].vector()
        wf = uf_f[0].vector()
        bu = [2*Ax*u, Ay*u + Ax*v, Az*u + Ay*w, 2*Ay*v, Az*v + Ay*w, 2*Az*w]
        buf = [2*Ax*uf, Ay*uf + Ax*vf, Az*uf + Ay*wf, 2*Ay*vf, Az*vf + Ay*wf, 2*Az*wf]

    # First we need to solve for the different components of Sij
    for i in xrange(tensdim):
        solve(G_matr, Sij[i].vector(), 0.5*bu[i], "cg", "default")
    # Second we need to solve for the diff. components of F(Sij)
    for i in xrange(tensdim):
        solve(G_matr, Sijf[i].vector(), 0.5*buf[i], "cg", "default")

    # Compute |S| = sqrt(2*Sij:Sij) and F(|S|) = sqrt(2*F(Sij):F(Sij))
    if tensdim == 3:
        # Extract Sij vectors
        S00 = Sij[0].vector().array()
        S01 = Sij[1].vector().array()
        S11 = Sij[2].vector().array()
        S00f = Sijf[0].vector().array()
        S01f = Sijf[1].vector().array()
        S11f = Sijf[2].vector().array()
        # Remove trace from Sij
        trace = 0.5*(S00+S11)
        S00 = S00-trace
        S11 = S11-trace
        Sij[0].vector().set_local(S00)
        Sij[0].vector().apply("insert")
        Sij[2].vector().set_local(S11)
        Sij[2].vector().apply("insert")
        # Remove trace from F(Sij)
        trace = 0.5*(S00f+S11f)
        S00f = S00f-trace
        S11f = S11f-trace
        Sijf[0].vector().set_local(S00f)
        Sijf[0].vector().apply("insert")
        Sijf[2].vector().set_local(S11f)
        Sijf[2].vector().apply("insert")
        # Compute |S|
        magS = np.sqrt(2*(S00*S00 + 2*S01*S01 + S11*S11))
        # Compute F(|S|)
        magSf = np.sqrt(2*(S00f*S00f + 2*S01f*S01f + S11f*S11f))
    else:
        # Extract Sij vectors
        S00 = Sij[0].vector().array()
        S01 = Sij[1].vector().array()
        S02 = Sij[2].vector().array()
        S11 = Sij[3].vector().array()
        S12 = Sij[4].vector().array()
        S22 = Sij[5].vector().array()
        S00f = Sijf[0].vector().array()
        S01f = Sijf[1].vector().array()
        S02f = Sijf[2].vector().array()
        S11f = Sijf[3].vector().array()
        S12f = Sijf[4].vector().array()
        S22f = Sijf[5].vector().array()
        # Remove trace from Sij
        trace = (1./3.)*(S00+S11+S22)
        S00 = S00-trace
        S11 = S11-trace
        S22 = S22-trace
        Sij[0].vector().set_local(S00)
        Sij[0].vector().apply("insert")
        Sij[3].vector().set_local(S11)
        Sij[3].vector().apply("insert")
        Sij[5].vector().set_local(S22)
        Sij[5].vector().apply("insert")
        # Remove trace from F(Sij)
        trace = (1./3.)*(S00f+S11f+S22f)
        S00f = S00f-trace
        S11f = S11f-trace
        S22f = S22f-trace
        Sijf[0].vector().set_local(S00f)
        Sijf[0].vector().apply("insert")
        Sijf[3].vector().set_local(S11f)
        Sijf[3].vector().apply("insert")
        Sijf[5].vector().set_local(S22f)
        Sijf[5].vector().apply("insert")
        # Compute |S|
        magS = np.sqrt(2*(S00*S00 + 2*S01*S01 + 2*S02*S02 + S11*S11 +
            2*S12*S12+ S22*S22))
        # Compute F(|S|)
        magSf = np.sqrt(2*(S00f*S00f + 2*S01f*S01f + 2*S02f*S02f + S11f*S11f +
            2*S12f*S12f+ S22f*S22f))

    # Multiply each component of Sij by magS, and each comp of F(Sij) by magSf
    for i in xrange(tensdim):
        # Compute 2*delta**2(F(|S|Sij) - alpha**2*F(|S|)F(Sij)) and add to Sij[i]
        Sij[i].vector().set_local(deltasq*(magS*Sij[i].vector().array()-(alpha**2)*magSf*Sijf[i].vector().array()))
        Sij[i].vector().apply("insert")
        # Last but not least, assign to Mij
        assigners_rev[i].assign(Mij.sub(i), Sij[i])

def tensor_inner(Sijcomps, Sijfcomps, assigners, tensdim, 
        A=None, B=None, **NS_namespace):
    """
    Compute tensor inner product of two symmetric tensors Aij and Bij
    """
    
    # Apply dummy functions
    dummiesA = Sijcomps
    dummiesB = Sijfcomps
    [dummiesA[i].vector().zero() for i in xrange(len(dummiesA))]
    [dummiesB[i].vector().zero() for i in xrange(len(dummiesB))]

    for i in xrange(tensdim):
        assigners[i].assign(dummiesA[i], A.sub(i))
        assigners[i].assign(dummiesB[i], B.sub(i))
    
    if tensdim == 3:
        contr = dummiesA[0].vector().array()*dummiesB[0].vector().array() +\
                2*dummiesA[1].vector().array()*dummiesB[1].vector().array() +\
                dummiesA[2].vector().array()*dummiesB[2].vector().array()
    else:
        contr = dummiesA[0].vector().array()*dummiesB[0].vector().array() +\
                2*dummiesA[1].vector().array()*dummiesB[1].vector().array() +\
                2*dummiesA[2].vector().array()*dummiesB[2].vector().array() +\
                dummiesA[3].vector().array()*dummiesB[3].vector().array() +\
                2*dummiesA[4].vector().array()*dummiesB[4].vector().array() +\
                dummiesA[5].vector().array()*dummiesB[5].vector().array()
    return contr
