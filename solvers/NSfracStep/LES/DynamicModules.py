__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

import numpy as np

def dyn_u_ops(u_ab, u_components, u_CG1, u_filtered, ll, bcs_u_CG1,
        G_matr, G_under, **NS_namespace):
    """
    Function for interpolating u to CG1, apply BCS, then filter,
    then apply BCS to filtered.
    """
    # Loop over u_components
    for i, ui in enumerate(u_components):
        # Interpolate to CG1
        ll.interpolate(u_CG1[i], u_ab[i])
        # Apply BCS
        [bc.apply(u_CG1[i].vector()) for bc in bcs_u_CG1[ui]]
        # Filter
        tophatfilter(unfiltered=u_CG1[i].vector(), filtered=u_filtered[i].vector(), **vars())
        # Apply BCS
        [bc.apply(u_filtered[i].vector()) for bc in bcs_u_CG1[ui]]

def lagrange_average(u_CG1, dt, CG1, tensdim, delta_CG1_sq, dim,
        Sijmats, G_matr, dummy, J1=None, J2=None, Aij=None, Bij=None, **NS_namespace):
    """
    Function for Lagrange Averaging two tensors
    AijBij and BijBij, PDE's are solved implicitly.

    d/dt(J1) + u*grad(J2) = 1/T(AijBij - J1)
    d/dt(J2) + u*grad(J1) = 1/T(BijBij - J2)
    Cs**2 = J1/J2

    - eps = (dt/T)/(1+dt/T) is computed.
    - The backward terms are assembled (UNSTABLE)
    - Tensor contractions of AijBij and BijBij are computed manually.
    - Two equations are solved implicitly and easy, no linear system.
    - J1 is clipped at 1E-32 (not zero, will lead to problems).
    """
    
    # Update eps and assign to dummy
    eps = dt*(J1.array()*J2.array())**(1./8.)/(1.5*np.sqrt(delta_CG1_sq.array()))
    dummy.set_local(eps/(1.0 + eps))
    dummy.apply("insert")

    # Compute tensor contractions
    AijBij = tensor_inner(A=Aij, B=Bij, **vars())
    BijBij = tensor_inner(A=Bij, B=Bij, **vars())

    J1_back = J1
    J2_back = J2

    # Update J1 and clip
    J1.axpy(1.0, dummy*(AijBij-J1_back))
    J1.set_local(J1.array().clip(min=1E-32))
    J1.apply("insert")
    # Update J2
    J2.axpy(1.0, dummy*(BijBij-J2_back))
    
def tophatfilter(G_matr, G_under, unfiltered=None, filtered=None,
        weight=1.0, **NS_namespace):
    """
    Filtering a CG1 function for applying a generalized top hat filter.
    uf = int(G*u)/int(G).

    G = CG1-basis functions.

    both unfiltered and filtered must be GenericVectors
    """

    # Filter to vec_
    vec = weight*((G_matr*unfiltered)*G_under.vector()) + (1-weight)*unfiltered
    # Zero filtered vector
    filtered.zero()
    # Axpy weighted filter operation to filtered
    filtered.axpy(1.0, vec)

def compute_Lij(Lij, uiuj_pairs, tensdim, G_matr, G_under, CG1,
        u=None, uf=None, Qij=None, **NS_namespace):
    """
    Manually compute the tensor Lij = F(uiuj)-F(ui)F(uj)
    """
    
    # Loop over each tensor component
    for i in xrange(tensdim):
        Lij[i].zero()
        # Extract velocity pair
        j, k = uiuj_pairs[i]
        # Add ujuk to Lij[i]
        Lij[i].axpy(1.0, u[j].vector()*u[k].vector())
        # Filter Lij[i] -> F(ujuk)
        tophatfilter(unfiltered=Lij[i], filtered=Lij[i], **vars())
        # Add to Qij if ScaleDep model
        if Qij != None:
            Qij[i].zero()
            Qij[i].axpy(1.0, Lij[i])
        # Axpy - F(uj)F(uk)
        Lij[i].axpy(-1.0, uf[j].vector()*uf[k].vector())

def compute_Mij(Mij, G_matr, G_under, Sijmats, Sijcomps, Sijfcomps, delta_CG1_sq,
        tensdim, Sij_sol, dummy, CG1, alphaval=None, u_nf=None, u_f=None, Nij=None, **NS_namespace):
    """
    Manually compute the tensor Mij = 2*delta**2*(F(|S|Sij)-alpha**2*F(|S|)F(Sij)
    """

    Sij = Sijcomps
    Sijf = Sijfcomps
    alpha = alphaval
    deltasq = delta_CG1_sq

    # Apply pre-assembled matrices and compute right hand sides
    if tensdim == 3:
        Ax, Ay = Sijmats
        u = u_nf[0].vector()
        v = u_nf[1].vector()
        uf = u_f[0].vector()
        vf = u_f[1].vector()
        # Unfiltered rhs
        bu = [Ax*u, 0.5*(Ay*u + Ax*v), Ay*v]
        # Filtered rhs
        buf = [Ax*uf, 0.5*(Ay*uf + Ax*vf), Ay*vf]
    else:
        Ax, Ay, Az = Sijmats
        u = u_nf[0].vector()
        v = u_nf[1].vector()
        w = u_nf[2].vector()
        uf = u_f[0].vector()
        vf = u_f[1].vector()
        wf = u_f[2].vector()
        bu = [Ax*u, 0.5*(Ay*u + Ax*v), 0.5*(Az*u + Ax*w), Ay*v, 0.5*(Az*v + Ay*w), Az*w]
        buf = [Ax*uf, 0.5*(Ay*uf + Ax*vf), 0.5*(Az*uf + Ax*wf), Ay*vf, 0.5*(Az*vf + Ay*wf), Az*wf]

    for i in xrange(tensdim):
        # Solve for the different components of Sij
        Sij_sol.solve(G_matr, Sij[i], bu[i])
        # Solve for the different components of F(Sij)
        Sij_sol.solve(G_matr, Sijf[i], buf[i])

    # Compute magnitudes of Sij and Sijf
    magS = mag(Aij=Sij, **vars())
    magSf = mag(Aij=Sijf, **vars())

    # Loop over components and add to Mij
    for i in xrange(tensdim):
        # Zero out Mij[i]
        Mij[i].zero()
        # Compute |S|*Sij
        Mij[i].axpy(1.0, magS*Sij[i])
        # Compute F(|S|*Sij)
        tophatfilter(unfiltered=Mij[i], filtered=Mij[i], **vars())
        
        # Check if Nij, assign F(|S|Sij) if not None
        if Nij != None:
            Nij[i].zero()
            Nij[i].axpy(1.0, Mij[i])
        
        # Compute 2*delta**2*(F(|S|Sij) - alpha**2*F(|S|)F(Sij)) and add to Mij[i]
        Mij[i].axpy(-1.0, (alpha**2)*magSf*Sijf[i])
        Mij[i] *= 2*deltasq
    
    # Return magS for use when updating nut_
    return magS

def compute_Qij(Qij, uiuj_pairs, tensdim, G_matr, G_under, uf=None, **NS_namespace):
    """
    Function for computing Qij in ScaleDepLagrangian
    """
    # Compute Qij
    for i in xrange(tensdim):
        j, k = uiuj_pairs[i]
        # Filter component of Qij
        tophatfilter(unfiltered=Qij[i], filtered=Qij[i], **vars())
        # Axpy outer(uf,uf) to Qij
        Qij[i].axpy(-1.0, uf[j].vector()*uf[k].vector())

def compute_Nij(Nij, G_matr, G_under, tensdim, Sijmats, Sijfcomps, delta_CG1_sq,
        Sij_sol, dummy, alphaval=None, u_f=None, **NS_namespace):
    """
    Function for computing Nij in ScaleDepLagrangian
    """
    
    Sijf = Sijfcomps
    alpha = alphaval
    deltasq = 2*delta_CG1_sq
    
    # Need to compute F(F(Sij)), set up right hand sides
    if tensdim == 3:
        Ax, Ay = Sijmats
        uf = u_f[0].vector()
        vf = u_f[1].vector()
        # Filtered rhs
        buf = [Ax*uf, 0.5*(Ay*uf + Ax*vf), Ay*vf]
    else:
        Ax, Ay, Az = Sijmats
        uf = u_f[0].vector()
        vf = u_f[1].vector()
        wf = u_f[2].vector()
        buf = [Ax*uf, 0.5*(Ay*uf + Ax*vf), 0.5*(Az*uf + Ax*wf), Ay*vf, 0.5*(Az*vf + Ay*wf), Az*wf]
    
    for i in xrange(tensdim):
        # Solve for the diff. components of F(F(Sij)))
        Sij_sol.solve(G_matr, Sijf[i], buf[i])
    
    # Compute magSf
    magSf = mag(Aij=Sijf, **vars())
    
    for i in xrange(tensdim):
        # Filter Nij = F(|S|Sij) --> F(F(|S|Sij))
        tophatfilter(unfiltered=Nij[i], filtered=Nij[i], **vars())
        # Compute 2*delta**2*(F(F(|S|Sij)) - alpha**2*F(F(|S))F(F(Sij)))
        Nij[i].axpy(-1.0, (alpha**2)*magSf*Sijf[i])
        Nij[i] *= deltasq

def compute_Hij(Hij, uiuj_pairs, dummy, dummy2, tensdim, G_matr, G_under, CG1,
        u=None, uf=None, **NS_namespace):
    """
    Scale similarity tensor Hij for use with the mixed dynamic sgs-model
    DMM2 by Vreman et.al.
    """

    w = 0.75
    
    # Loop over tensor components
    for i in range(tensdim):
        # Compute 
        # Hij = F(G(F(ui)F(uj))) - F(G(F(ui)))F(G(F(uj))) - F(G(uiuj)) + F(G(ui)G(uj))

        # Zero Hij component
        Hij[i].zero()
        # Extract uiuj_pair
        j,k = uiuj_pairs[i]
        
        # Compute and add F(G(F(ui)F(uj)))
        dummy.zero()
        dummy.axpy(1.0, uf[j].vector()*uf[k].vector())
        # Filter grid filter
        tophatfilter(unfiltered=dummy, filtered=dummy, weight=w, **vars())
        # Filter test filter
        tophatfilter(unfiltered=dummy, filtered=dummy, **vars())
        # Add to Hij
        Hij[i].axpy(1.0, dummy)

        # Compute and add F(G(F(ui)))F(G(F(uj)))
        dummy.zero()
        dummy.axpy(1.0, uf[j].vector())
        # Filter uf[j] twice, first grid then test
        tophatfilter(unfiltered=dummy, filtered=dummy, weight=w, **vars())
        tophatfilter(unfiltered=dummy, filtered=dummy, **vars())
        dummy2.zero()
        dummy2.axpy(1.0, uf[k].vector())
        # Filter uf[k] twice, first grid then test
        tophatfilter(unfiltered=dummy2, filtered=dummy2, weight=w, **vars())
        tophatfilter(unfiltered=dummy2, filtered=dummy2, **vars())
        # Add to Hij
        Hij[i].axpy(-1.0, dummy*dummy2)

        # Compute and add F(G(uiuj))
        dummy.zero()
        dummy.axpy(1.0, u[j].vector()*u[k].vector())
        # Filter twice, grid then test
        tophatfilter(unfiltered=dummy, filtered=dummy, weight=w, **vars())
        tophatfilter(unfiltered=dummy, filtered=dummy, **vars())
        # Add to Hij
        Hij[i].axpy(-1.0, dummy)

        # Compute and add F(G(ui)G(uj))
        dummy.zero()
        dummy.axpy(1.0, u[j].vector())
        # Filter u[j]
        tophatfilter(unfiltered=dummy, filtered=dummy, weight=w, **vars())
        dummy2.zero()
        dummy2.axpy(1.0, u[k].vector())
        # Filter u[k]
        tophatfilter(unfiltered=dummy2, filtered=dummy2, weight=w, **vars())
        # Filter dummy2*dummy and add to dummy
        tophatfilter(unfiltered=dummy*dummy2, filtered=dummy, **vars())
        # Add to Hij
        Hij[i].axpy(1.0, dummy)

def compute_Hij_DMM1(Hij, uiuj_pairs, dummy, dummy2, tensdim, G_matr, G_under, CG1,
        u=None, uf=None, **NS_namespace):
    """
    Tensor applied in the DMM1 model by Zang et.al.
    """
    
    w = 0.75
    
    for i in xrange(tensdim):

        Hij[i].zero()
        j,k = uiuj_pairs[i]

        # Compute and add F(G(ui)G(uj))
        dummy.zero()
        dummy.axpy(1.0, u[j].vector())
        # Grid filter u[j]
        tophatfilter(unfiltered=dummy, filtered=dummy, weight=w, **vars())
        dummy2.zero()
        dummy2.axpy(1.0, u[k].vector())
        # Grid filter u[k]
        tophatfilter(unfiltered=dummy2, filtered=dummy2, weight=w, **vars())
        # Axpy to Hij
        Hij[i].axpy(1.0, dummy*dummy2)
        # Filter dummy
        tophatfilter(unfiltered=Hij[i], filtered=Hij[i], **vars())

        # Compute and add F(G(ui))F(G(uj))
        tophatfilter(unfiltered=dummy, filtered=dummy, **vars())
        tophatfilter(unfiltered=dummy2, filtered=dummy2, **vars())
        # Axpy to Hij
        Hij[i].axpy(-1.0, dummy*dummy2)

def compute_Leonard(Lij, uiuj_pairs, dummy, dummy2, tensdim, G_matr, G_under, CG1,
        u=None, **NS_namespace):
    """
    Leonard tensor for rhs of NS when mixed dynamic SGS-model applied.
    """

    w = 0.75

    # Loop over components
    for i in range(tensdim):
        j,k = uiuj_pairs[i]
        Lij[i].zero()
        # Add uiuj
        Lij[i].axpy(1.0, u[j].vector()*u[k].vector())
        # Grid filter --> G(uiuj)
        tophatfilter(unfiltered=Lij[i], filtered=Lij[i], weight=w, **vars())
        # Filter u velocities once through grid filter
        tophatfilter(unfiltered=u[j].vector(), filtered=dummy, weight=w, **vars())
        tophatfilter(unfiltered=u[k].vector(), filtered=dummy2, weight=w, **vars())
        # Axpy -G(ui)G(uj) to Lij
        Lij[i].axpy(-1.0, dummy*dummy2)

    # Remove trace from Lij
    remove_trace(Aij=Lij, **vars())

def update_mixedLESSource(u_components, u_CG1, mixedmats, Lij, tensdim, dummy2,
        uiuj_pairs, G_matr, G_under, CG1, mixedLESSource, dummy, **NS_namespace):

    # Compute Leonard Tensor for velocity, added to Lij
    compute_Leonard(u=u_CG1, **vars())
    # Update components of mixedLESSource
    if tensdim == 3:
        Ax, Ay = mixedmats
        for i, ui in enumerate(u_components):
            mixedLESSource[ui] = Ax*Lij[i] + Ay*Lij[i+1]

    elif tensdim == 6:
        Ax, Ay, Az = mixedmats
        k = 0
        for i, ui in enumerate(u_components):
            mixedLESSource[ui] = Ax*Lij[i] + Ay*Lij[i+k+1] + Az*Lij[i+k+1]
            k = 1

def remove_trace(tensdim, dummy, Aij=None, **NS_namespace):
    """
    Remove trace from a symetric tensor Aij.
    """
    trace = dummy
    trace.zero()
    if tensdim == 3:
        trace.axpy(0.5, (Aij[0] + Aij[2]))
        Aij[0].axpy(-1.0, trace)
        Aij[2].axpy(-1.0, trace)
    elif tensdim == 6:
        trace.axpy((1./3.), (Aij[0]+Aij[3]+Aij[5]))
        Aij[0].axpy(-1.0, trace)
        Aij[3].axpy(-1.0, trace)
        Aij[5].axpy(-1.0, trace)

def tensor_inner(tensdim, A=None, B=None, **NS_namespace):
    """
    Compute tensor contraction Aij:Bij of two symmetric tensors Aij and Bij.
    A GenericVector is returned.
    """
    if tensdim == 3:
        contraction = A[0]*B[0] +\
                    2*A[1]*B[1] +\
                      A[2]*B[2]
    else:
        contraction = A[0]*B[0] +\
                    2*A[1]*B[1] +\
                    2*A[2]*B[2] +\
                      A[3]*B[3] +\
                    2*A[4]*B[4] +\
                      A[5]*B[5]
    return contraction

def mag(tensdim, dummy, Aij=None, **NS_namespace):
    """
    Compute |A| = magA = 2*sqrt(inner(Aij,Aij))
    """
    if tensdim == 3:
        # Compute |S|
        magA = 2*(Aij[0]*Aij[0] + 2*Aij[1]*Aij[1] + Aij[2]*Aij[2])
    elif tensdim == 6:
        # Compute |S|
        magA = 2*(Aij[0]*Aij[0] + 2*Aij[1]*Aij[1] + 2*Aij[2]*Aij[2] + Aij[3]*Aij[3] +
            2*Aij[4]*Aij[4] + Aij[5]*Aij[5])

    magA.set_local(np.sqrt(magA.array()))
    magA.apply("insert")

    return magA
