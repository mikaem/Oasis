__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-10-14"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from common.default_hooks import *

def get_solvers(use_krylov_solvers, krylov_solvers, bcs, x_, Q, 
                scalar_components, velocity_update_type, 
                **NS_namespace):
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
        if velocity_update_type != "default":
            du_sol = None
        else:
            #du_sol = KrylovSolver('bicgstab', 'jacobi')
            du_sol = KrylovSolver('bicgstab', 'hypre_euclid')
            if "structure" in du_sol.parameters['preconditioner']:
                du_sol.parameters['preconditioner']['structure'] = "same"
            else:
                du_sol.parameters['preconditioner']['reuse'] = True
            du_sol.parameters.update(krylov_solvers)
            du_sol.parameters['preconditioner']['ilu']['fill_level'] = 2

            #PETScOptions.set("pc_hypre_euclid_bj", True)
            #PETScOptions.set("pc_hypre_euclid_print_statistics", True)
            PETScOptions.set("pc_factor_reuse_ordering", True)
            PETScOptions.set("pc_factor_reuse_fill", True)

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
        u_sol.parameters["same_nonzero_pattern"] = True
        ## velocity correction ##
        if velocity_update_type != "default":
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

def attach_pressure_nullspace(p_sol, x_, Q):
    """Create null space basis object and attach to Krylov solver."""
    null_vec = Vector(x_['p'])
    Q.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0/null_vec.norm("l2")
    null_space = VectorSpaceBasis([null_vec])
    p_sol.set_nullspace(null_space)
    p_sol.null_space = null_space

def solve_pressure(dp_, x_, Ap, b, p_sol, bcs, **NS_namespace):
    """Solve pressure equation."""
    [bc.apply(b['p']) for bc in bcs['p']]
    dp_.vector().zero()
    dp_.vector().axpy(1., x_['p'])
    # KrylovSolvers use nullspace for normalization of pressure
    if hasattr(p_sol, 'null_space'):
        p_sol.null_space.orthogonalize(b['p']);

    t1 = Timer("Pressure Linear Algebra Solve")
    p_sol.solve(Ap, x_['p'], b['p'])
    t1.stop()
    # LUSolver use normalize directly for normalization of pressure
    if hasattr(p_sol, 'normalize'):
        normalize(x_['p'])

    #dp_.vector()[:] = x_['p'][:] - dp_.vector()[:]
    dp_.vector().axpy(-1., x_['p'])
    dp_.vector()._scale(-1.)

def add_pressure_gradient_rhs(b, x_, P, p_, v, i, ui, **NS_namespace):
    """Add pressure gradient on rhs of tentative velocity equation."""
    if P:
        b[ui].axpy(-1., P[ui]*x_['p'])
    else:
        b[ui].axpy(-1., assemble(v*p_.dx(i)*dx))

def assemble_pressure_rhs(b, Rx, x_, dt, q, u_, Ap, **NS_namespace):
    """Assemble rhs of pressure equation."""
    b['p'].zero()
    if Rx:
        for ui in Rx:
            b['p'].axpy(-1./dt, Rx[ui]*x_[ui])
    else:
        b['p'].axpy(-1./dt, assemble(div(u_)*q*dx))
    b['p'].axpy(1., Ap*x_['p'])

def update_velocity(u_components, b, Mu, bcs, print_solve_info, du_sol, 
                    P, dp_, dt, v, x_, **NS_namespace):
    for i, ui in enumerate(u_components):
        b[ui].zero()
        b[ui].axpy(1.0, Mu*x_[ui])
        if P:
            b[ui].axpy(-dt, P[ui]*dp_.vector())
        else:
            b[ui].axpy(-dt, assemble(v*dp_.dx(i)*dx))

        [bc.apply(b[ui]) for bc in bcs[ui]]
        info_blue('Updating velocity '+ui, print_solve_info)
        t1 = Timer("Update Linear Algebra Solve")
        du_sol.solve(Mu, x_[ui], b[ui])
        t1.stop()

def velocity_tentative_hook(ui, use_krylov_solvers, u_sol, **NS_namespace):
    """Called just prior to solving for tentative velocity."""
    if use_krylov_solvers:
        if ui == "u0":
            if "structure" in u_sol.parameters['preconditioner']:
                u_sol.parameters['preconditioner']['structure'] = "same_nonzero_pattern"
            else:
                u_sol.parameters['preconditioner']['reuse'] = False
                u_sol.parameters['preconditioner']['same_nonzero_pattern'] = True    
        else:
            if "structure" in u_sol.parameters['preconditioner']:
                u_sol.parameters['preconditioner']['structure'] = "same"
            else:
                u_sol.parameters['preconditioner']['reuse'] = True
