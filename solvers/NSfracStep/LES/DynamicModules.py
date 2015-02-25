__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import TrialFunction, TestFunction, assemble, inner, dx, grad,\
                    Function, dot, solve, plot, interactive
import numpy as np

def lagrange_average(u_ab, dt, dummy, CG1, bcJ1, bcJ2, Sijcomps, 
        Sijfcomps, assigners, tensdim, delta_CG1, J1=None, J2=None, 
        Aij=None, Bij=None, **NS_namespace):
    """
    Function for Lagrange Averaging two tensors
    AijBij and BijBij, PDE's are solved implicitly.

    d/dt(J1) + u*grad(J2) = 1/T(AijBij - J1)
    d/dt(J2) + u*grad(J1) = 1/T(BijBij - J2)

    Cs**2 = J1/J2

    - eps = (dt/T)/(1+dt/T) is computed.
    - The bacckward terms are assembled.
    - Tensor contractions of AijBij and BijBij are computed manually.
    - Two equations are solved implicitly and easy, no linear system.
    - J1 is clipped at 1E-32 (not zero, will lead to problems).
    """

    # Update eps
    eps = dt*(J1.vector().array()*J2.vector().array())**(1./8.)/(1.5*delta_CG1.vector().array())
    eps = eps/(1.0+eps)

    # Compute tensor contractions
    AijBij = tensor_inner(A=Aij, B=Bij, **vars())
    BijBij = tensor_inner(A=Bij, B=Bij, **vars())

    # Compute backward convective terms J(x-dt*u) (!! NOT STABLE !!)
    J1_back = Function(CG1)
    J2_back = Function(CG1)
    #b = assemble(dt*dot(u_ab,grad(TrialFunction(CG1)))*TestFunction(CG1)*dx)
    #solve(A_lag, J1_back.vector(), b*J1.vector(), "cg", "additive_schwarz")
    #solve(A_lag, J2_back.vector(), b*J2.vector(), "cg", "additive_schwarz")
    J1_back = J1.vector().array()-J1_back.vector().array()
    J2_back = np.abs(J2.vector().array()-J2_back.vector().array())

    # Update J1
    J1.vector().set_local(eps*AijBij + (1-eps)*J1_back)
    J1.vector().apply("insert")
    # Update J2
    J2.vector().set_local(eps*BijBij + (1-eps)*J2_back)
    J2.vector().apply("insert")

    # Apply ramp function on J1 to remove negative values, but not set to 0.
    J1.vector().set_local(J1.vector().array().clip(min=1E-32))
    J1.vector().apply("insert")

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

def compute_Lij(Lij, uiuj_pairs, tensdim, dummy, G_matr, G_under,
        assigners_rev, assigners, u=None, uf=None, Qij=None, **NS_namespace):
    """
    Manually compute the tensor Lij = F(uiuj)-F(ui)F(uj)
    """
    
    # Loop over each tensor component
    for i in xrange(tensdim):
        # Extract velocity pair
        j, k = uiuj_pairs[i]
        dummy.vector().zero()
        # Add ujuk to dummy
        dummy.vector().axpy(1.0, u[j].vector()*u[k].vector())
        # Filter dummy -> F(ujuk)
        tophatfilter(unfiltered=dummy, filtered=dummy, **vars())
        # Add to Qij if ScaleDep model
        if Qij != None:
            assigners_rev[i].assign(Qij.sub(i), dummy)
        # Axpy - F(uj)F(uk)
        dummy.vector().axpy(-1.0, uf[j].vector()*uf[k].vector())
        # Assign to Lij
        assigners_rev[i].assign(Lij.sub(i), dummy)

def compute_Mij(Mij, G_matr, CG1, dim, tensdim, assigners_rev, Sijmats,
        Sijcomps, Sijfcomps, dt, delta_CG1, dummy, G_under, assigners,
        alphaval=None, u_nf=None, u_f=None, Nij=None, **NS_namespace):
    """
    Manually compute the tensor Mij = 2*delta**2*(F(|S|Sij)-alpha**2*F(|S|)F(Sij)
    """

    Sij = Sijcomps
    Sijf = Sijfcomps
    alpha = alphaval
    deltasq = 2*(delta_CG1.vector().array())**2
    
    # Apply pre-assembled matrices and compute right hand sides
    if tensdim == 3:
        Ax, Ay = Sijmats
        u = u_nf[0].vector()
        v = u_nf[1].vector()
        uf = u_f[0].vector()
        vf = u_f[1].vector()
        # Unfiltered rhs
        bu = [2*Ax*u, Ay*u + Ax*v, 2*Ay*v]
        # Filtered rhs
        buf = [2*Ax*uf, Ay*uf + Ax*vf, 2*Ay*vf]
    else:
        Ax, Ay, Az = Sijmats
        u = u_nf[0].vector()
        v = u_nf[1].vector()
        w = u_nf[2].vector()
        uf = u_f[0].vector()
        vf = u_f[1].vector()
        wf = u_f[0].vector()
        bu = [2*Ax*u, Ay*u + Ax*v, Az*u + Ay*w, 2*Ay*v, Az*v + Ay*w, 2*Az*w]
        buf = [2*Ax*uf, Ay*uf + Ax*vf, Az*uf + Ay*wf, 2*Ay*vf, Az*vf + Ay*wf, 2*Az*wf]

    for i in xrange(tensdim):
        # First we need to solve for the different components of Sij
        solve(G_matr, Sij[i].vector(), 0.5*bu[i], "cg", "default")
        # Second we need to solve for the diff. components of F(Sij)
        solve(G_matr, Sijf[i].vector(), 0.5*buf[i], "cg", "default")
    
    # Compute magnitudes of Sij and Sijf
    magS = compute_magSij(Sij, tensdim)
    magSf = compute_magSij(Sijf, tensdim)

    # Multiply each component of Sij by magS, and each comp of F(Sij) by magSf
    for i in xrange(tensdim):
        # Compute |S|*Sij
        Sij[i].vector().set_local(magS*Sij[i].vector().array())
        Sij[i].vector().apply("insert")
        # Compute F(|S|*Sij)
        tophatfilter(unfiltered=Sij[i], filtered=Sij[i], **vars())
        # Check if Nij
        if Nij != None:
            assigners_rev[i].assign(Nij.sub(i), Sij[i])
        # Compute 2*delta**2*(F(|S|Sij) - alpha**2*F(|S|)F(Sij)) and add to Sij[i]
        Sij[i].vector().set_local(deltasq*(Sij[i].vector().array()-(alpha**2)*magSf*Sijf[i].vector().array()))
        Sij[i].vector().apply("insert")
        # Last but not least, assign to Mij
        assigners_rev[i].assign(Mij.sub(i), Sij[i])

    # Return magS for use when updating nut_
    return magS

def compute_Qij(Qij, uiuj_pairs, tensdim, dummy, G_matr, G_under,
        assigners_rev, assigners, uf=None, **NS_namespace):
    """
    Function for computing Qij in ScaleDepLagrangian
    """
    # Compute Qij
    for i in range(tensdim):
        j, k = uiuj_pairs[i]
        # Assign Qij comp to dummy
        assigners[i].assign(dummy, Qij.sub(i))
        # Filter component
        tophatfilter(unfiltered=dummy, filtered=dummy, **vars())
        # Axpy outer(uf,uf)
        dummy.vector().axpy(-1.0, uf[j].vector()*uf[k].vector())
        # Assign to Qij
        assigners_rev[i].assign(Qij.sub(i), dummy)

def compute_Nij(Mij, G_matr, CG1, dim, tensdim, assigners_rev, Sijmats,
        Sijcomps, Sijfcomps, dt, delta_CG1, dummy, G_under, assigners,
        alphaval=None, u_nf=None, u_f=None, Nij=None, **NS_namespace):
    """
    Function for computing Nij in ScaleDepLagrangian
    """
    
    Sijf = Sijfcomps
    alpha = alphaval
    # Compute 2*delta**2 from T_ = dt/(1.5*delta)
    deltasq = 2*(delta_CG1.vector().array())**2
    
    # Need to compute F(F(Sij)), set up right hand sides
    if tensdim == 3:
        Ax, Ay = Sijmats
        uf = u_f[0].vector()
        vf = u_f[1].vector()
        # Filtered rhs
        buf = [2*Ax*uf, Ay*uf + Ax*vf, 2*Ay*vf]
    else:
        Ax, Ay, Az = Sijmats
        uf = u_f[0].vector()
        vf = u_f[1].vector()
        wf = u_f[0].vector()
        buf = [2*Ax*uf, Ay*uf + Ax*vf, Az*uf + Ay*wf, 2*Ay*vf, Az*vf + Ay*wf, 2*Az*wf]
    
    for i in xrange(tensdim):
        # Solve for the diff. components of F(F(Sij)))
        solve(G_matr, Sijf[i].vector(), 0.5*buf[i], "cg", "default")
    
    # Compute magSf
    magSf = compute_magSij(Sijf, tensdim)
    
    for i in range(tensdim):
        # Extract F(|S|Sij) from Nij
        assigners[i].assign(dummy, Nij.sub(i))
        # Filter dummy
        tophatfilter(unfiltered=dummy, filtered=dummy, **vars())
        # Axpy - alpha**2*magSf*Sijf
        dummy.vector().set_local(deltasq*(dummy.vector().array()-alpha**2*magSf*Sijf[i].vector().array()))
        dummy.vector().apply("insert")
        # Assign to Nij
        assigners_rev[i].assign(Nij.sub(i), dummy)

def tensor_inner(Sijcomps, Sijfcomps, assigners, tensdim, 
        A=None, B=None, **NS_namespace):
    """
    Compute tensor contraction Aij:Bij of two symmetric tensors Aij and Bij.
    A numpy array is returned.
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
        contraction = dummiesA[0].vector().array()*dummiesB[0].vector().array() +\
                2*dummiesA[1].vector().array()*dummiesB[1].vector().array() +\
                dummiesA[2].vector().array()*dummiesB[2].vector().array()
    else:
        contraction = dummiesA[0].vector().array()*dummiesB[0].vector().array() +\
                2*dummiesA[1].vector().array()*dummiesB[1].vector().array() +\
                2*dummiesA[2].vector().array()*dummiesB[2].vector().array() +\
                dummiesA[3].vector().array()*dummiesB[3].vector().array() +\
                2*dummiesA[4].vector().array()*dummiesB[4].vector().array() +\
                dummiesA[5].vector().array()*dummiesB[5].vector().array()

    return contraction

def compute_magSij(Sij, tensdim):
    """
    Compute |S| = magS = 2*sqrt(inner(Sij,Sij))
    """
    if tensdim == 3:
        # Extract Sij vectors
        S00 = Sij[0].vector().array()
        S01 = Sij[1].vector().array()
        S11 = Sij[2].vector().array()
        # Compute |S|
        magS = np.sqrt(2*(S00*S00 + 2*S01*S01 + S11*S11))
    elif tensdim == 6:
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
    
    return magS
