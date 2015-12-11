__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-09-15'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, FunctionSpace, assemble, TestFunction, dx, \
        inner,DirichletBC, Constant, CellVolume, TrialFunction, KrylovSolver
from common import derived_bcs
import numpy as np

__all__ = ['les_setup', 'les_update']

def les_setup(U_AB, mesh, Sigma, CG1Function, nut_krylov_solver, bcs, V,
        constrained_domain, MPI, mpi_comm_world, **NS_namespace):
    """
    Set up for solving Sigma SVD LES model.
    """
    CG1 = FunctionSpace(mesh, "CG", 1)
    p,q = TrialFunction(CG1), TestFunction(CG1)
    u = TrialFunction(V)

    dim = mesh.geometry().dim()
    tensdim = dim*dim
    if dim == 2:
        if MPI.rank(mpi_comm_world()) == 0:
            print "\nWARNING: Sigma LES model only valid for 3D. Proceeding...\n"
    ij_pairs = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]

    # Define matrices
    A_mat = assemble(inner(p,q)*dx)
    b_mats = [assemble(inner(u.dx(i),q)*dx) for i in range(dim)]
    # Set up solver
    grad_sol = KrylovSolver("bicgstab", "jacobi")
    grad_sol.parameters["preconditioner"]["structure"] = "same_nonzero_pattern"
    grad_sol.parameters["error_on_nonconvergence"] = False
    grad_sol.parameters["monitor_convergence"] = False
    grad_sol.parameters["report"] = False
    
    dummy = Function(CG1).vector()
    gij = [dummy.copy() for i in range(dim**2)]

    delta = pow(CellVolume(mesh), 1./dim)

    sigma = [Function(CG1),Function(CG1),Function(CG1)] #[sigma_1, sigma_2, sigma_3]
    sigma[0].vector()[:] = 1.   # Set equal to one to avoid zero-divison first steps
    D_sigma = (sigma[2]*(sigma[0]-sigma[1])*(sigma[1]-sigma[2]))/(sigma[0]**2)
    
    nut_form = Sigma['Cs']**2 * delta**2 * D_sigma
    nut_ = CG1Function(nut_form, mesh, method=nut_krylov_solver,
            bcs=[],name="nut", bounded=True)

    return dict(nut_=nut_, delta=delta, CG1=CG1, sigma=sigma, A_mat=A_mat, 
                b_mats=b_mats, grad_sol=grad_sol, gij=gij, dim=dim,
                ij_pairs=ij_pairs, tensdim=tensdim)

def les_update(nut_, u_ab, tstep, mesh, ij_pairs, sigma, Sigma, 
                A_mat, b_mats, grad_sol, gij, dim, tensdim, **NS_namespace):
    
    # Start computing when u_ab != 0, hence skip first steps,
    # eventually compute new sigmas each nth timestep.
    if tstep < 3 or tstep%Sigma["comp_step"] != 0:
        # Update nut_
        nut_()
        return

    # Compute all velocity gradients
    for k in xrange(tensdim):
        i,j = ij_pairs[k] 
        grad_sol.solve(A_mat, gij[k], b_mats[j]*u_ab[i].vector())
    
    # Create array of local gradient matrices by fast numpy vectorization
    G = np.concatenate(tuple([gij[k].array() for k in xrange(tensdim)])).reshape(dim,dim,-1).transpose(2,0,1)
    # Solve for Singular values of all matrices in G simultaneously
    sigmas = np.linalg.svd(G, compute_uv=False)
    
    # Set_local and apply sigma arrays to sigma functions
    [(sigma[i].vector().set_local(sigmas[:,i]),sigma[i].vector().apply("insert")) for i in range(dim)]

    # Update nut_
    nut_()
