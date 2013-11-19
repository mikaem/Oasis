__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-10-14"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from NavierStokes import *

NS_parameters['low_memory_version'] = True

def get_solvers(use_krylov_solvers, use_lumping_of_mass_matrix, 
                krylov_solvers, sys_comp, bcs, x_, Q, 
                scalar_components, **NS_namespace):
    """Return linear solvers. 
    
    We are solving for
       - tentative velocity
       - pressure correction
       - velocity update (unless lumping is switched on)
       
       and possibly:       
       - scalars
            
    """
    if use_krylov_solvers:
        ## tentative velocity solver ##
        u_sol = KrylovSolver('bicgstab', 'jacobi')
        u_sol.parameters.update(krylov_solvers)
        if "structure" in u_sol.parameters['preconditioner']:
            u_sol.parameters['preconditioner']['structure'] = "same_nonzero_pattern"
        else:
            u_sol.parameters['preconditioner']['reuse'] = False
            u_sol.parameters['preconditioner']['same_nonzero_pattern'] = True

        ## velocity correction solver
        du_sol = None
        
        ## pressure solver ##
        #p_prec = PETScPreconditioner('petsc_amg')
        #p_prec.parameters['report'] = True
        #p_prec.parameters['reuse'] = True
        #p_prec.parameters['gamg']['num_aggregation_smooths'] = 1
        #p_sol = PETScKrylovSolver('gmres', p_prec)
        #p_sol.p_prec = p_prec
        if bcs['p'] == []:
            p_sol = KrylovSolver('minres', 'hypre_amg')
        else:
            p_sol = KrylovSolver('gmres', 'hypre_amg')
        if "structure" in p_sol.parameters['preconditioner']:
            p_sol.parameters['preconditioner']['structure'] = "same"
        else:
            p_sol.parameters['preconditioner']['reuse'] = True    
        
        p_sol.parameters.update(krylov_solvers)
        if bcs['p'] == []:
            attach_pressure_nullspace(p_sol, x_, Q)
        sols = [u_sol, p_sol, du_sol]
        ## scalar solver ##
        if len(scalar_components) > 0:
            #c_sol = KrylovSolver('bicgstab', 'hypre_euclid')
            c_sol = KrylovSolver('bicgstab', 'jacobi')
            c_sol.parameters.update(krylov_solvers)
            if "structure" in c_sol.parameters['preconditioner']:
                c_sol.parameters['preconditioner']['structure'] = "same_nonzero_pattern"
            else:
                c_sol.parameters['preconditioner']['reuse'] = False
                c_sol.parameters['preconditioner']['same_nonzero_pattern'] = True    
            sols.append(c_sol)
        else:
            sols.append(None)
    else:
        ## tentative velocity solver ##
        u_sol = LUSolver()
        ## velocity correction ##
        du_sol = None
        ## pressure solver ##
        p_sol = LUSolver()
        p_sol.parameters['reuse_factorization'] = True
        if bcs['p'] == []:
            p_sol.normalize = True
        sols = [u_sol, p_sol, du_sol]
        ## scalar solver ##
        if len(scalar_components) > 0:
            c_sol = LUSolver()
            sols.append(c_sol)
        else:
            sols.append(None)
        
    return sols

def add_pressure_gradient_rhs(b, p_, i, ui, M, dp, lp, **NS_namespace):
    """Add pressure gradient on rhs of tentative velocity equation."""
    dp.vector().zero()
    lp.solve(dp, Form(p_.dx(i)*dx))
    b[ui].axpy(-1., M*dp.vector())
    
def assemble_pressure_rhs(b, Q, x_, dt, q, u_, Ap, Mp, lpq, **NS_namespace):
    """Assemble rhs of pressure equation."""
    b['p'].zero()
    divu = Function(Q)
    lpq.solve(divu, Form(div(u_)*dx))
    #lpq.solve_lumping(divu, Form(q*div(u_)*dx))
    [bc.apply(divu.vector()) for bc in lpq.bcs]
    b['p'].axpy(-1./dt, Mp*divu.vector())
    b['p'].axpy(1., Ap*x_['p'])

def update_velocity_lao(dp_, lp, dp, V, dt, x_, u_components, bcs, **NS_namespace):
    for i, ui in enumerate(u_components):
        dp.vector().zero()
        lp.solve(dp, Form(dp_.dx(i)*dx))
        x_[ui].axpy(-dt, dp.vector())
        [bc.apply(x_[ui]) for bc in bcs[ui]]
