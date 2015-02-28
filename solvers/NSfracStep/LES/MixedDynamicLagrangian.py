__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-28'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, assemble, TestFunction, dx, solve, Constant,\
        FacetFunction, DirichletBC, TestFunction, as_vector, div,\
        TrialFunction
from DynamicModules import tophatfilter, lagrange_average, compute_Lij,\
        compute_Mij, compute_Hij, compute_Leonard
import DynamicLagrangian
import numpy as np

__all__ = ['les_setup', 'les_update']

def les_setup(u_, mesh, dt, krylov_solvers, V, assemble_matrix, CG1Function, nut_krylov_solver, 
        bcs, u_components, **NS_namespace):
    """
    Set up for solving the mixed scale similar Germano Dynamic LES model applying
    Lagrangian Averaging. The implementation is based on the work of
    Vreman et.al. 1994, "On the Formulation of the Dynamic Mixed Subgrid-scale
    Model" and their so called DMM2 model. 
    
    Results showed that both the DMM1 and DMM2 models performed better than 
    the Germano Dynamic Model. However an inconsistency present in DMM1 is 
    fixed in DMM2, resulting in even better results.

    Cs**2 = avg((Lij-Hij)Mij)/avg(MijMij)
    
    where

    Lij = F(uiuj) - F(ui)F(uj)
    Mij = 2*delta**2*(F(|S|Sij)-alpha**2*F(|S|)F(Sij))
    Hij = F(G(F(ui)F(uj)))-F(G(F(ui)))F(G(F(uj))) - F(G(uiuj)) + F(G(ui)G(uj))
    
    and the Leonard tensor is

    Lij_L = dev(G(uiuj)-G(ui)G(uj))
    
    SGS stress modeled as

    tau_ij = Lij_L - 2*Cs**2*delta**2*|S|*Sij

    - Test filter F = 0.5 iterations on filter
    - Grid filter G = 0.25 iteration on filter
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
            mixedmats, Sij_sol, **NS_namespace):

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
    
    # Compute Hij
    compute_Hij(u=u_CG1, uf=u_filtered, **vars())
    
    # Compute Aij = Lij-Hij and add to Hij
    for i in xrange(tensdim):
        Hij[i].vector().set_local(Lij[i].vector().array()-Hij[i].vector().array())
        Hij[i].vector().apply("insert")

    # Lagrange average (Lij-Hij) and Mij
    lagrange_average(J1=JLM, J2=JMM, Aij=Hij, Bij=Mij, **vars())

    # Update Cs = JLM/JMM and filter/smooth Cs, then clip at 0.09 
    """
    Important that the term in nut_form is Cs and not Cs**2
    since Cs here is stored as JLM/JMM.
    """
    Cs.vector().set_local(JLM.vector().array()/JMM.vector().array())
    Cs.vector().apply("insert")
    tophatfilter(unfiltered=Cs, filtered=Cs, N=2, weight=1., **vars())
    Cs.vector().set_local(Cs.vector().array().clip(max=0.09))
    Cs.vector().apply("insert")

    # Update nut_
    nut_.vector().set_local(Cs.vector().array() * delta_CG1_sq.vector().array() * magS)
    nut_.vector().apply("insert")
    
    # Compute Leonard Tensor
    compute_Leonard(u=u_CG1, **vars())
    # Update components of mixedLESSource
    if tensdim == 3:
        Ax, Ay = mixedmats
        for i, ui in enumerate(u_components):
            mixedLESSource[ui] = -(Ax*Lij[i].vector()+Ay*Lij[i+1].vector())
    elif tensdim == 6:
        Ax, Ay, Az = mixedmats
        mixedLESSource["u0"] = -(Ax*Lij[0].vector()+Ay*Lij[1].vector()+Az*Lij[2].vector())
        mixedLESSource["u1"] = -(Ax*Lij[1].vector()+Ay*Lij[3].vector()+Az*Lij[4].vector())
        mixedLESSource["u2"] = -(Ax*Lij[2].vector()+Ay*Lij[4].vector()+Az*Lij[5].vector())
