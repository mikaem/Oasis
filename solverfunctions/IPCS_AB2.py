__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-10-14"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

#from common.default_hooks import *
from NavierStokes import *

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
        u_sol.parameters['preconditioner']['reuse'] = True
        u_sol.parameters['preconditioner']['same_nonzero_pattern'] = True
        ## velocity correction solver
        if use_lumping_of_mass_matrix:
            du_sol = None
        else:
            du_sol = KrylovSolver('bicgstab', 'hypre_euclid')
            du_sol.parameters.update(krylov_solvers)
            du_sol.parameters['preconditioner']['reuse'] = True
            du_sol.parameters['preconditioner']['same_nonzero_pattern'] = True
            du_sol.parameters['preconditioner']['ilu']['fill_level'] = 1
            #PETScOptions.set("pc_hypre_euclid_print_statistics", True)

        ## pressure solver ##
        if bcs['p'] == []:
            p_sol = KrylovSolver('minres', 'hypre_amg')
        else:
            p_sol = KrylovSolver('gmres', 'hypre_amg')
        p_sol.parameters['preconditioner']['reuse'] = True
        p_sol.parameters['preconditioner']['same_nonzero_pattern'] = True
        p_sol.parameters.update(krylov_solvers)
        if bcs['p'] == []:
            attach_pressure_nullspace(p_sol, x_, Q)
        sols = [u_sol, p_sol, du_sol]
        ## scalar solver ##
        if len(scalar_components) > 0:
            #c_sol = KrylovSolver('bicgstab', 'hypre_euclid')
            c_sol = KrylovSolver('bicgstab', 'jacobi')
            c_sol.parameters.update(krylov_solvers)
            c_sol.parameters['preconditioner']['reuse'] = False
            c_sol.parameters['preconditioner']['same_nonzero_pattern'] = True
            sols.append(c_sol)
        else:
            sols.append(None)
    else:
        ## tentative velocity solver ##
        u_sol = LUSolver()
        u_sol.parameters['reuse_factorization'] = True
        ## velocity correction ##
        if use_lumping_of_mass_matrix:
            du_sol = None
        else:
            du_sol = LUSolver()
            du_sol.parameters['reuse_factorization'] = True
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

def velocity_tentative_hook(ui, use_krylov_solvers, u_sol, **NS_namespace):
    """Called just prior to solving for tentative velocity."""
    pass
