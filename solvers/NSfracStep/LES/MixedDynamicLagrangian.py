__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-28'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, assemble, TestFunction, dx, solve, Constant,\
        FacetFunction, DirichletBC, TestFunction, as_vector, div,\
        TrialFunction
from DynamicModules import tophatfilter, lagrange_average, compute_Lij,\
        compute_Mij, compute_Qij, compute_Nij
import DynamicLagrangian
import numpy as np

__all__ = ['les_setup', 'les_update']

def les_setup(u_, mesh, dt, krylov_solvers, V, assemble_matrix, CG1Function, nut_krylov_solver, 
        bcs, u_components, **NS_namespace):
    """
    Set up for solving the mixed Germano Dynamic LES model applying
    Lagrangian Averaging.
    """
    
    # The setup is 99% equal to DynamicLagrangian, hence use its les_setup
    dyn_dict = DynamicLagrangian.les_setup(**vars())
    
    # Set up functions for scale similarity tensor Hij
    Hij = [Function(dyn_dict["CG1"]) for i in range(dyn_dict["dim"]**2)]
    mixedmats = [assemble_matrix(TrialFunction(dyn_dict["CG1"]).dx(i)*TestFunction(V)*dx)
        for i in range(dyn_dict["dim"])]

    dyn_dict.update(Hij=Hij, mixedmats=mixedmats)

    return dyn_dict

def les_update(u_ab, nut_, nut_form, dt, CG1, delta, tstep, u_components, V,
            DynamicSmagorinsky, Cs, u_CG1, u_filtered, Lij, Mij, Hij,
            JLM, JMM, dim, tensdim, G_matr, G_under, ll, mixedLESSource,
            dummy, uiuj_pairs, Sijmats, Sijcomps, Sijfcomps, delta_CG1_sq, 
            mixedmats, **NS_namespace):

    # Check if Cs is to be computed, if not update nut_ and break
    if tstep%DynamicSmagorinsky["Cs_comp_step"] != 0:
        # Update nut_
        nut_()
        # Break function
        return
    
    # All velocity components must be interpolated to CG1 then filtered
    for i in xrange(dim):
        # Interpolate to CG1
        ll.interpolate(u_CG1[i], u_ab[i])
        # Filter
        tophatfilter(unfiltered=u_CG1[i], filtered=u_filtered[i], **vars())

    # Compute Lij applying dynamic modules function
    compute_Lij(u=u_CG1, uf=u_filtered, **vars())

    # Compute Mij applying dynamic modules function
    alpha = 2.0
    magS = compute_Mij(alphaval=alpha, u_nf=u_CG1, u_f=u_filtered, **vars())
    
    # Compute Hij applying dynamic modules function
    compute_Hij(u=u_CG1, uf=u_filtered, **vars())
    
    # Compute Aij = Lij-Hij and add to Hij
    for i in xrange(tensdim):
        Hij[i].vector().set_local(Lij[i].vector().array()-Hij[i].vector().array())
        Hij[i].vector().apply("insert")

    # Lagrange average Lij and Mij
    lagrange_average(J1=JLM, J2=JMM, Aij=Hij, Bij=Mij, **vars())

    # Update Cs = sqrt(JLM/JMM) and filter/smooth Cs, then clip at 0.3. 
    """
    Important that the term in nut_form is Cs**2 and not Cs
    since Cs here is stored as sqrt(JLM/JMM).
    """
    Cs.vector().set_local(np.sqrt(JLM.vector().array()/JMM.vector().array()))
    Cs.vector().apply("insert")
    tophatfilter(unfiltered=Cs, filtered=Cs, N=2, weight=1., **vars())
    Cs.vector().set_local(Cs.vector().array().clip(max=0.3))
    Cs.vector().apply("insert")

    # Update nut_
    nut_.vector().set_local(Cs.vector().array()**2 * delta_CG1_sq.vector().array() * magS)
    nut_.vector().apply("insert")
    
    # Update mixedSource, first remove trace from Lij
    remove_trace(Aij=Lij, **vars())
    # Update components of mixedLESSource (div(Lij
    if tensdim == 3:
        Ax, Ay = mixedmats
        for i, ui in enumerate(u_components):
            mixedLESSource[ui] = -(Ax*Lij[i].vector()+Ay*Lij[i+1].vector())
    elif tensdim == 6:
        Ax, Ay, Az = mixedmats
        for i, ui in enumerate(u_components):
            mixedLESSource[ui] = -(Ax*Lij[i].vector()+Ay*Lij[i+1].vector()+Az*Lij[i+2].vector())

def remove_trace(tensdim, Aij=None, **NS_namespace):
    
    if tensdim == 3:
        trace = 0.5*(Aij[0].vector().array()+Aij[2].vector().array())
        Aij[0].vector().set_local(Aij[0].vector().array()-trace)
        Aij[0].vector().apply("insert")
        Aij[2].vector().set_local(Aij[2].vector().array()-trace)
        Aij[2].vector().apply("insert")
    elif tensdim == 6:
        trace = (1./3.)*(Aij[0].vector().array()+Aij[3].vector().array()+Aij[5].vector().array())
        Aij[0].vector().set_local(Aij[0].vector().array()-trace)
        Aij[0].vector().apply("insert")
        Aij[3].vector().set_local(Aij[3].vector().array()-trace)
        Aij[3].vector().apply("insert")
        Aij[5].vector().set_local(Aij[5].vector().array()-trace)
        Aij[5].vector().apply("insert")

def compute_Hij(Hij, uiuj_pairs, dummy, tensdim, G_matr, G_under, CG1,
        u=None, uf=None, **NS_namespace):
    
    # Loop over tensor components
    for i in range(tensdim):
        # Compute and add F(F(F(ui)F(uj))) - F(F(F(ui)))F(F(F(uj))) -
        # (F(F(uiuj)) - F(F(ui)F(uj))) to Hij
        
        # Zero Hij component
        Hij[i].vector().zero()
        # Extract uiuj_pair
        j,k = uiuj_pairs[i]

        # Compute and add F(F(F(ui)F(uj)))
        dummy.vector().zero()
        dummy.vector().axpy(1.0, uf[j].vector()*uf[k].vector())
        # Filter twice
        tophatfilter(unfiltered=dummy, filtered=dummy, N=2, **vars())
        # Add to Hij
        Hij[i].vector().axpy(1.0, dummy.vector())

        # Compute and add F(F(F(ui)))F(F(F(uj)))
        dummy.vector().zero()
        dummy.vector().axpy(1.0, uf[j].vector())
        # Filter uf[j] twice
        tophatfilter(unfiltered=dummy, filtered=dummy, N=2, **vars())
        dummy2 = Function(CG1)
        dummy2.vector().zero()
        dummy2.vector().axpy(1.0, uf[k].vector())
        # Filter uf[k] twice
        tophatfilter(unfiltered=dummy2, filtered=dummy2, N=2, **vars())
        # Add to Hij
        Hij[i].vector().axpy(-1.0, dummy.vector()*dummy2.vector())
        
        # Compute and add F(F(uiuj))
        dummy.vector().zero()
        dummy.vector().axpy(1.0, u[j].vector()*u[k].vector())
        # Filter twice
        tophatfilter(unfiltered=dummy, filtered=dummy, N=2, **vars())
        # Add to Hij
        Hij[i].vector().axpy(-1.0, dummy.vector())

        # Compute and add F(F(ui)F(uj))
        dummy.vector().zero()
        dummy.vector().axpy(1.0, u[j].vector())
        # Filter u[j]
        tophatfilter(unfiltered=dummy, filtered=dummy, **vars())
        dummy2.vector().zero()
        dummy2.vector().axpy(1.0, u[k].vector())
        # Filter u[k]
        tophatfilter(unfiltered=dummy2, filtered=dummy2, **vars())
        # Axpy to dummy
        vec_ = dummy.vector()*dummy2.vector()
        dummy.vector().zero()
        dummy.vector().axpy(1.0, vec_)
        # Filter dummy
        tophatfilter(unfiltered=dummy, filtered=dummy, **vars())
        # Add to Hij
        Hij[i].vector().axpy(1.0, dummy.vector())
