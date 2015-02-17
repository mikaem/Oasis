__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, FunctionSpace, assemble, TestFunction, sym, grad,\
        dx, inner, as_backend_type, TrialFunction, project, CellVolume, sqrt,\
        TensorFunctionSpace, assign, solve, lhs, rhs, LagrangeInterpolator,\
        dev, outer, as_vector, FunctionAssigner, KrylovSolver, DirichletBC
from DynamicModules import tophatfilter, lagrange_average, compute_Lij,\
        compute_Mij
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
        
        # Update nut_
        nut_.vector().set_local(assemble(nut_form*v_dg*dx).array()/dg_diag)
        nut_.vector().apply("insert")

        # Break function
        return

    # All velocity components must be interpolated to CG1 then filtered
    for i in xrange(dim):
        # Interpolate to CG1
        u_CG1[i].interpolate(u_[i])
        # Filter
        tophatfilter(unfiltered=u_CG1[i], filtered=u_filtered[i], **vars())

    # Compute Lij from dynamic modules function
    compute_Lij(u=u_CG1, uf=u_filtered, **vars())

    # Compute Mij from dynamic modules function
    alpha = 2.
    compute_Mij(alphaval=alpha, u_nf=u_CG1, u_f=u_filtered, **vars())

    # Lagrange average Lij and Mij
    lagrange_average(J1=JLM, J2=JMM, Aij=Lij, Bij=Mij, **vars())

    # Now u needs to be filtered once more
    for i in xrange(dim):
        # Filter
        tophatfilter(unfiltered=u_filtered[i], filtered=u_filtered[i], **vars())
    
    # Compute Qij from dynamic modules function
    compute_Lij(u=u_CG1, uf=u_filtered, **vars())

    # Compute Nij from dynamic modules function
    alpha = 4.
    compute_Mij(alphaval=alpha, u_nf=u_CG1, u_f=u_filtered, **vars())

    # Lagrange average Lij and Mij
    lagrange_average(J1=JQN, J2=JNN, Aij=Qij, Bij=Nij, **vars())

    # UPDATE Cs**2 = (JLM*JMM)/beta, beta = JQN/JNN
    beta = JQN.vector().array()/JNN.vector().array()
    beta = beta.clip(min=0.5)
    Cs.vector().set_local((np.sqrt((JLM.vector().array()/JMM.vector().array())/beta)).clip(max=0.4))
    Cs.vector().apply("insert")

    # Update nut_
    nut_.vector().set_local(assemble(nut_form*v_dg*dx).array()/dg_diag)
    nut_.vector().apply("insert")
