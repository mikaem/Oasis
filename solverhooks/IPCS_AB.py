__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-10-14"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from common.default_hooks import *

def assemble_lumped_P1_diagonal(Vv, M, **NS_namespace):
    ones = Function(Vv)
    ones.vector()[:] = 1.
    ML = M * ones.vector()
    ML.set_local(1. / ML.array())
    return ML

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
        if "structure" in u_sol.parameters['preconditioner']:
            u_sol.parameters['preconditioner']['structure'] = "same_nonzero_pattern"
        else:
            u_sol.parameters['preconditioner']['reuse'] = False
            u_sol.parameters['preconditioner']['same_nonzero_pattern'] = True
        u_sol.parameters.update(krylov_solvers)

        ## velocity correction solver
        if use_lumping_of_mass_matrix:
            du_sol = None
        else:
            du_sol = KrylovSolver('bicgstab', 'hypre_euclid')
            if "structure" in du_sol.parameters['preconditioner']:
                du_sol.parameters['preconditioner']['structure'] = "same"
            else:
                du_sol.parameters['preconditioner']['reuse'] = True
            du_sol.parameters.update(krylov_solvers)

        ## pressure solver ##
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

def add_pressure_gradient_rhs(b, x_, P, p_, v, **NS_namespace):
    """Add pressure gradient on rhs of tentative velocity equation."""
    if P:
        b['u'].axpy(-1., P*x_['p'])
    else:
        b['u'].axpy(-1., assemble(dot(v, grad(p_))*dx))

def add_pressure_gradient_rhs_update(b, dt, P, dp_, v, **NS_namespace):
    """Add pressure gradient on rhs of velocity update equation."""
    if P:
        b['u'].axpy(-dt, P * dp_.vector())
    else:
        b['u'].axpy(-dt, assemble(dot(v, grad(dp_))*dx))
        
def assemble_pressure_rhs(b, Rx, x_, dt, q, u_, Ap, **NS_namespace):
    """Assemble rhs of pressure equation."""
    b['p'].zero()
    if Rx:
        b['p'].axpy(-1./dt, Rx*x_['u'])
    else:
        b['p'].axpy(-1./dt, assemble(div(u_)*q*dx))
    b['p'].axpy(1., Ap*x_['p'])

def update_velocity_lumping(P, dp_, ML, dt, x_, v, bcs, **NS_namespace):
    if P:
        x_['u'].axpy(-dt, (P * dp_.vector()) * ML)
    else:
        x_['u'].axpy(-dt, (assemble(dot(v, grad(dp_))*dx)) * ML)

    [bc.apply(x_['u']) for bc in bcs['u']]

def velocity_tentative_hook(use_krylov_solvers, u_sol, **NS_namespace):
    """Called just prior to solving for tentative velocity."""
    pass