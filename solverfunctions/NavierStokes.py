__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-10-14"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from NSdefault_hooks import *

def assemble_lumped_P1_diagonal(V, M, **NS_namespace):
    ones = Function(V)
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
        u_sol.parameters.update(krylov_solvers)
        u_sol.parameters['preconditioner']['reuse'] = False
        u_sol.parameters['preconditioner']['same_nonzero_pattern'] = True
        ## velocity correction solver
        if use_lumping_of_mass_matrix:
            du_sol = None
        else:
            du_sol = KrylovSolver('bicgstab', 'hypre_euclid')
            du_sol.parameters.update(krylov_solvers)
            du_sol.parameters['preconditioner']['reuse'] = True
        ## pressure solver ##
        #p_prec = PETScPreconditioner('petsc_amg')
        #p_prec.parameters['report'] = True
        #p_prec.parameters['same_nonzero_pattern'] = True
        #p_prec.parameters['gamg']['verbose'] = 20
        #p_prec.parameters['gamg']['num_aggregation_smooths'] = 2
        #p_sol = PETScKrylovSolver('gmres', p_prec)
        #p_sol.p_prec = p_prec
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

def add_pressure_gradient_rhs(b, x_, P, p_, v, i, ui, **NS_namespace):
    """Add pressure gradient on rhs of tentative velocity equation."""
    if P:
        b[ui].axpy(-1., P[ui]*x_['p'])
    else:
        b[ui].axpy(-1., assemble(v*p_.dx(i)*dx))

def add_pressure_gradient_rhs_update(b, dt, P, dp_, v, i, ui, **NS_namespace):
    """Add pressure gradient on rhs of velocity update equation."""
    if P:
        b[ui].axpy(-dt, P[ui]*dp_.vector())
    else:
        b[ui].axpy(-dt, assemble(v*dp_.dx(i)*dx))
        
def assemble_pressure_rhs(b, Rx, x_, dt, q, u_, Ap, **NS_namespace):
    """Assemble rhs of pressure equation."""
    b['p'].zero()
    if Rx:
        for ui in Rx:
            b['p'].axpy(-1./dt, Rx[ui]*x_[ui])
    else:
        b['p'].axpy(-1./dt, assemble(div(u_)*q*dx))
    b['p'].axpy(1., Ap*x_['p'])

def update_velocity_lumping(P, dp_, ML, dt, x_, v, u_components, bcs, **NS_namespace):
    for i, ui in enumerate(u_components):
        if P:
            x_[ui].axpy(-dt, (P[ui] * dp_.vector()) * ML)
        else:
            x_[ui].axpy(-dt, (assemble(v*dp_.dx(i)*dx)) * ML)
        [bc.apply(x_[ui]) for bc in bcs[ui]]

def velocity_tentative_hook(ui, use_krylov_solvers, u_sol, **NS_namespace):
    """Called just prior to solving for tentative velocity."""
    if use_krylov_solvers:
        if ui == "u0":
            u_sol.parameters['preconditioner']['reuse'] = False
        else:
            u_sol.parameters['preconditioner']['reuse'] = True
