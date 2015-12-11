__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-11-14'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, FunctionSpace, assemble, TestFunction, dx, \
        inner,DirichletBC, Constant, CellVolume, TrialFunction, KrylovSolver
import DynamicLagrangian
from common import derived_bcs
from DynamicModules import tensor_inner, dyn_u_ops, compute_Lij, \
        compute_Mij_Sigma
import numpy as np

__all__ = ['les_setup', 'les_update']

def les_setup(U_AB, mesh, Sigma, CG1Function, nut_krylov_solver, bcs, V,
        constrained_domain, MPI, mpi_comm_world, u_, dt, assemble_matrix,
        u_components, DynamicSmagorinsky, **NS_namespace):

    """
    Set up for solving Sigma SVD LES model includning a scale similar mixed
    term.
    """
    CG1 = FunctionSpace(mesh, "CG", 1, constrained_domain=constrained_domain)
    p,q = TrialFunction(CG1), TestFunction(CG1)
    u = TrialFunction(V)

    dim = mesh.geometry().dim()
    tensdim = dim*dim
    if dim == 2:
        if MPI.rank(mpi_comm_world()) == 0:
            print "\nWARNING: Sigma LES model only valid for 3D. Proceeding...\n"
    ij_pairs = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]

    # Ryn DynamicLagrangian setup to obtain some values
    sig_dict = DynamicLagrangian.les_setup(**vars())
    vol = float(assemble(Constant(1)*dx(mesh)))
    LM = Function(CG1)
    MM = Function(CG1)
    Sigma["Cs_arr"] = []
    Sigma["t_arr"] = []

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
    dummy2 = dummy.copy()
    gij = [dummy.copy() for i in range(dim**2)]

    delta = pow(CellVolume(mesh), 1./dim)

    sigma = [Function(CG1),Function(CG1),Function(CG1)] #[sigma_1, sigma_2, sigma_3]
    sigmaf = [Function(CG1),Function(CG1),Function(CG1)] #[sigma_1, sigma_2, sigma_3]
    sigma[0].vector()[:] = 1.   # Set equal to one to avoid zero-divison first steps
    sigmaf[0].vector()[:] = 1.
    D_sigma = (sigma[2]*(sigma[0]-sigma[1])*(sigma[1]-sigma[2]))/(sigma[0]**2)

    nut_form = Sigma['Cs']**2 * delta**2 * D_sigma
    nut_ = CG1Function(nut_form, mesh, method=nut_krylov_solver,
            bcs=[],name="nut", bounded=True)

    sig_dict.update(nut_=nut_, delta=delta, CG1=CG1, sigma=sigma, A_mat=A_mat,
                b_mats=b_mats, grad_sol=grad_sol, gij=gij, dim=dim, LM=LM, MM=MM,
                ij_pairs=ij_pairs, tensdim_f=tensdim, vol=vol, dummy2=dummy2,
                sigmaf=sigmaf)
    return sig_dict

def les_update(nut_, u_ab, tstep, dt, mesh, ij_pairs, sigma, Sigma, V,
               A_mat, b_mats, grad_sol, gij, dim, tensdim, tensdim_f,
	       u_components, u_CG1, Lij, Mij, dummy2, Sijmats, Sijcomps, Sijfcomps,
               uiuj_pairs, G_matr, row_mean , CG1, dummy, vol, Sij_sol,
               delta_CG1_sq, LM, MM, delta, u_, u_filtered, ll, bcs_u_CG1,
               vdegree, u_filtered_CG2, sigmaf, **NS_namespace):

    # Start computing when u_ab != 0, hence skip first steps,
    # eventually compute new sigmas each nth timestep.
    if tstep < 3 or tstep%Sigma["comp_step"] != 0:
        # Update nut_
        nut_()
        return

    # Compute all velocity gradients
    for k in xrange(tensdim_f):
        i,j = ij_pairs[k]
        grad_sol.solve(A_mat, gij[k], b_mats[j]*u_ab[i].vector())

    # Create array of local gradient matrices by fast numpy vectorization
    G = np.concatenate(tuple([gij[k].array() for k in xrange(tensdim_f)])).reshape(dim,dim,-1).transpose(2,0,1)
    # Solve for Singular values of all matrices in G simultaneously
    sigmas = np.linalg.svd(G, compute_uv=False)

    # Set_local and apply sigma arrays to sigma functions
    [(sigma[i].vector().set_local(sigmas[:,i]),sigma[i].vector().apply("insert")) for i in range(dim)]

    # Compute diff operator
    s1,s2,s3 = sigmas[:,0], sigmas[:,1], sigmas[:,2]
    Ds = (s3*(s1-s2)*(s2-s3))/(s1**2)

    # All velocity components must be interpolated to CG1 then filtered, also apply bcs
    dyn_u_ops(**vars())

    # D_sigma must now be computed for the filtered u_ab
    # Compute all velocity gradients
    if vdegree == 2:
        uf_ = u_filtered_CG2
    else:
        uf_ = u_filtered
    for k in xrange(tensdim_f):
        i,j = ij_pairs[k]
        grad_sol.solve(A_mat, gij[k], b_mats[j]*uf_[i].vector())

    # Create array of local gradient matrices by fast numpy vectorization
    G = np.concatenate(tuple([gij[k].array() for k in xrange(tensdim_f)])).reshape(dim,dim,-1).transpose(2,0,1)
    # Solve for Singular values of all matrices in G simultaneously
    sigmas = np.linalg.svd(G, compute_uv=False)

    # Compute diff operator
    s1,s2,s3 = sigmas[:,0], sigmas[:,1], sigmas[:,2]
    Dsf = (s3*(s1-s2)*(s2-s3))/(s1**2)

    # Compute Lij applying dynamic modules function
    compute_Lij(u=u_CG1, uf=u_filtered, **vars())

    # Compute Mij applying dynamic modules function
    alpha = 2.0
    compute_Mij_Sigma(alphaval=alpha, u_nf=u_, u_f=u_filtered_CG2, D_sigma=Ds, D_sigma_f=Dsf, **vars())

    # Contractions of LijMij and MklMkl
    LM.vector().zero()
    LM.vector().axpy(1.0, tensor_inner(A=Lij, B=Mij, **vars()))
    MM.vector().zero()
    MM.vector().axpy(1.0, tensor_inner(A=Mij, B=Mij, **vars()))

    # Avg. contractions over entire domain
    A = assemble(LM*dx(mesh))/vol
    B = assemble(MM*dx(mesh))/vol

    # Update Sigma and nut.form
    Sigma["Cs"] = float(A)/B
    nut_.form = Sigma["Cs"] * delta**2 * (sigma[2]*(sigma[0]-sigma[1])*(sigma[1]-sigma[2]))/(sigma[0]**2)

    # Update nut_
    nut_()
