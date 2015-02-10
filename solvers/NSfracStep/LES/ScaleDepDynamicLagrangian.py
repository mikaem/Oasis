__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, FunctionSpace, assemble, TestFunction, sym, grad,\
        dx, inner, as_backend_type, TrialFunction, project, CellVolume, sqrt,\
        TensorFunctionSpace, assign, solve, lhs, rhs, LagrangeInterpolator,\
        dev, outer, as_vector, FunctionAssigner, KrylovSolver, DirichletBC
from DynamicModules import tophatfilter, lagrange_average, compute_uiuj,\
        compute_magSSij
import DynamicLagrangian
import numpy as np

__all__ = ['les_setup', 'les_update']

def les_setup(u_, mesh, dt, krylov_solvers, V, assemble_matrix, **NS_namespace):
    """
    Set up for solving the Germano Dynamic LES model applying
    scale dependent Lagrangian Averaging.
    """
    
    # The setup is 99% equal to DynamicLagrangian, hence use its les_setup
    dyn_dict = DynamicLagrangian.les_setup(**vars())
    
    # Add scale dep specific parameters
    JQN = Function(dyn_dict["CG1"])
    JQN.vector()[:] += 1e-7
    JNN = Function(dyn_dict["CG1"])
    JNN.vector()[:] += 10.
    
    # Update and return dict
    dyn_dict.update(JQN=JQN, JNN=JNN)

    return dyn_dict

def les_update(u_, nut_, nut_form, v_dg, dg_diag, dt, CG1, delta, tstep, 
            DynamicSmagorinsky, Cs, u_CG1, u_filtered, F_uiuj, F_SSij, 
            JLM, JMM, JQN, JNN, bcJ1, bcJ2, eps, T_, dim, tensdim, G_matr, 
            G_under, dummy, assigners, assigners_rev, lag_sol, 
            uiuj_pairs, Sijmats, Sijcomps, **NS_namespace):

    # Check if Cs is to be computed, if not update nut_ and break
    if tstep%DynamicSmagorinsky["Cs_comp_step"] != 0:
        ##################
        # Solve for nut_ #
        ##################
        nut_.vector().set_local(assemble(nut_form*v_dg*dx).array()/dg_diag)
        nut_.vector().apply("insert")
        # BREAK FUNCTION
        return

    # Ratio between filters, such that delta_tilde = 2*delta,
    # where delta is the implicit mesh filter.
    alpha = 2.

    #############################
    # Filter the velocity field #
    #############################
    # All velocity components must be interpolated to CG1
    # then filtered
    for i in xrange(dim):
        # Interpolate to CG1
        u_CG1[i].interpolate(u_[i])
        # Filter
        tophatfilter(unfiltered=u_CG1[i], filtered=u_filtered[i], **vars())

    ##############
    # SET UP Lij #
    ##############
    # Compute outer product of uiuj
    compute_uiuj(u=u_CG1, **vars())
    # Compute F(uiuj) and add to F_uiuj
    tophatfilter(unfilterd=F_uiuj, filtered=F_uiuj, N=tensdim, **vars())
    # Define Lij = dev(F(uiuj)-F(ui)F(uj))
    Lij = dev(F_uiuj - outer(u_filtered, u_filtered))

    ##############
    # SET UP Mij #
    ##############
    # Compute |S|Sij and add to F_SSij
    compute_magSSij(u=u_, **vars())
    # Compute F(|S|Sij) and add to F_SSij
    tophatfilter(unfilterd=F_SSij, filtered=F_SSij, N=tensdim, **vars())
    # Define F(Sij)
    Sijf = sym(grad(u_filtered))
    # Define F(|S|) = sqrt(2*Sijf:Sijf)
    magSf = sqrt(2*inner(Sijf,Sijf))
    # Define Mij = 2*delta**2(F(|S|Sij) - alpha**2F(|S|)F(Sij))
    Mij = 2*(delta**2)*(F_SSij - (alpha**2)*magSf*Sijf)

    ##################################################
    # Solve Lagrange Equations for LijMij and MijMij #
    ##################################################
    lagrange_average(J1=JLM, J2=JMM, Aij=Lij, Bij=Mij, **vars())

    # Now u needs to be filtered once more
    for i in xrange(dim):
        # Filter
        tophatfilter(unfiltered=u_filtered[i], filtered=u_filtered[i], **vars())
    
    ##############
    # SET UP Qij #
    ##############
    # Filter F(uiuj) --> F(F(uiuj))
    tophatfilter(unfilterd=F_uiuj, filtered=F_uiuj, N=tensdim, **vars())
    # Define Qij
    Qij = dev(F_uiuj - outer(u_filtered, u_filtered))
    
    ##############
    # SET UP Nij #
    ##############
    # F(|S|Sij) has allready been computed, filter once more
    tophatfilter(unfilterd=F_SSij, filtered=F_SSij, N=tensdim, **vars())
    # Define F(Sij)
    Sijf = sym(grad(u_filtered))
    # Define F(|S|) = sqrt(2*Sijf:Sijf)
    magSf = sqrt(2*inner(Sijf,Sijf))
    # Define Mij = 2*delta**2(F(|S|Sij) - alpha**2F(|S|)F(Sij))
    Nij = 2*(delta**2)*(F_SSij - ((2*alpha)**2)*magSf*Sijf)

    ##################################################
    # Solve Lagrange Equations for QijNij and NijNij #
    ##################################################
    lagrange_average(J1=JQN, J2=JNN, Aij=Qij, Bij=Nij, **vars())

    #################################
    # UPDATE Cs**2 = (JLM*JMM)/beta #
    # beta = JQN/JNN                #
    #################################
    beta = JQN.vector().array()/JNN.vector().array()
    beta = beta.clip(min=0.5)
    Cs.vector().set_local(np.sqrt((JLM.vector().array()/JMM.vector().array())/beta))
    Cs.vector().apply("insert")

    ##################
    # Solve for nut_ #
    ##################
    nut_.vector().set_local(assemble(nut_form*v_dg*dx).array()/dg_diag)
    nut_.vector().apply("insert")
