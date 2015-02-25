__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function, FunctionSpace, assemble, TestFunction, sym, grad,\
        dx, inner, as_backend_type, TrialFunction, project, CellVolume, sqrt,\
        TensorFunctionSpace, FunctionAssigner, DirichletBC, as_vector, solve,\
        plot, interactive
from DynamicModules import tophatfilter, lagrange_average, compute_Lij,\
        compute_Mij, tensor_inner
import numpy as np

__all__ = ['les_setup', 'les_update']

def les_setup(u_, mesh, dt, V, assemble_matrix, **NS_namespace):
    """
    Set up for solving the Germano Dynamic LES model applying
    Lagrangian Averaging.
    """
    
    # Create function spaces
    DG = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)
    p,q = TrialFunction(CG1), TestFunction(CG1)
    p2 = TrialFunction(V)
    TFS = TensorFunctionSpace(mesh, "CG", 1, symmetry=True)
    dim = mesh.geometry().dim()

    # Create delta in DG0 and CG1
    delta = project(pow(CellVolume(mesh), 1./dim), DG)
    delta_CG1 = project(delta, CG1)
    
    # Define nut_
    nut_ = Function(CG1)
    Sij = sym(grad(u_))
    magS = sqrt(2*inner(Sij,Sij))
    Cs = Function(CG1)
    nut_form = Cs**2 * delta**2 * magS

    # Create functions for holding the different velocities
    u_CG1 = as_vector([Function(CG1) for i in range(dim)])
    u_filtered = as_vector([Function(CG1) for i in range(dim)])
    dummy = Function(CG1)
    
    # Assemble required filter matrices and functions
    G_under = Function(CG1, assemble(TestFunction(CG1)*dx))
    G_under.vector().set_local(1./G_under.vector().array())
    G_under.vector().apply("insert")
    G_matr = assemble(inner(p,q)*dx)

    # Assemble some required matrices for solving for rate of strain terms
    Lij = Function(TFS)
    Mij = Function(TFS)
    # Check if case is 2D or 3D and set up uiuj product pairs and 
    # Sij forms, assemble required matrices
    Sijcomps = [Function(CG1) for i in range(dim*dim)]
    Sijfcomps = [Function(CG1) for i in range(dim*dim)]
    Sijmats = [assemble_matrix(p.dx(i)*q*dx) for i in range(dim)]
    if dim == 3:
        tensdim = 6
        uiuj_pairs = ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
    else:
        tensdim = 3
        uiuj_pairs = ((0,0),(0,1),(1,1))
    
    # Set up function assigners
    # From TFS.sub(i) to CG1
    assigners = [FunctionAssigner(CG1, TFS.sub(i)) for i in range(tensdim)]
    # From CG1 to TFS.sub(i)
    assigners_rev = [FunctionAssigner(TFS.sub(i), CG1) for i in range(tensdim)]
    
    # Set up Lagrange Equations
    JLM = Function(CG1)
    JLM.vector()[:] += 1E-32
    JMM = Function(CG1)
    JMM.vector()[:] += 1
    # These DirichletBCs are needed for the stability 
    # when solving the Lagrangian PDEs
    bcJ1 = DirichletBC(CG1, 0, "on_boundary")
    bcJ2 = DirichletBC(CG1, 1, "on_boundary")
    
    return dict(Sij=Sij, nut_form=nut_form, nut_=nut_, delta=delta, delta_CG1=delta_CG1,
                DG=DG, CG1=CG1,
                Cs=Cs, u_CG1=u_CG1, u_filtered=u_filtered,
                Lij=Lij, Mij=Mij, Sijcomps=Sijcomps, Sijfcomps=Sijfcomps, Sijmats=Sijmats, 
                JLM=JLM, JMM=JMM, bcJ1=bcJ1, bcJ2=bcJ2, TFS=TFS,
                dim=dim, tensdim=tensdim, G_matr=G_matr, G_under=G_under, 
                dummy=dummy, assigners=assigners, assigners_rev=assigners_rev, 
                uiuj_pairs=uiuj_pairs) 
    
def les_update(u_, u_ab, nut_, nut_form, dt, CG1, delta, tstep, 
            DynamicSmagorinsky, Cs, u_CG1, u_filtered, Lij, Mij,
            JLM, JMM, bcJ1, bcJ2, dim, tensdim, G_matr, G_under,
            dummy, assigners, assigners_rev, uiuj_pairs, Sijmats,
            Sijcomps, Sijfcomps, delta_CG1, **NS_namespace):

    # Check if Cs is to be computed, if not update nut_ and break
    if tstep%DynamicSmagorinsky["Cs_comp_step"] != 0:
        # Update nut_
        solve(G_matr, nut_.vector(), assemble(nut_form*TestFunction(CG1)*dx), "cg", "default")
        # Remove negative values
        nut_.vector().set_local(nut_.vector().array().clip(min=0))
        nut_.vector().apply("insert")
        # Break function
        return

    # All velocity components must be interpolated to CG1 then filtered
    for i in xrange(dim):
        # Interpolate to CG1
        u_CG1[i].interpolate(u_ab[i])
        # Filter
        tophatfilter(unfiltered=u_CG1[i], filtered=u_filtered[i], **vars())

    # Compute Lij applying dynamic modules function
    compute_Lij(u=u_CG1, uf=u_filtered, **vars())

    # Compute Mij applying dynamic modules function
    alpha = 2.0
    magS = compute_Mij(alphaval=alpha, u_nf=u_CG1, u_f=u_filtered, **vars())

    # Lagrange average Lij and Mij
    lagrange_average(J1=JLM, J2=JMM, Aij=Lij, Bij=Mij, **vars())

    # Update Cs = sqrt(JLM/JMM) and filter/smooth Cs, then clip at 0.3. 
    """
    Important that the term in nut_form is Cs**2 and not Cs
    since Cs here is stored as sqrt(JLM/JMM).
    """
    Cs.vector().set_local(np.sqrt(JLM.vector().array()/JMM.vector().array()))
    Cs.vector().apply("insert")
    tophatfilter(unfiltered=Cs, filtered=Cs, **vars())
    tophatfilter(unfiltered=Cs, filtered=Cs, **vars())
    Cs.vector().set_local(Cs.vector().array().clip(max=0.3))
    Cs.vector().apply("insert")

    # Update nut_
    nut_.vector().set_local(Cs.vector().array()**2 * delta_CG1.vector().array()**2 * magS)
    nut_.vector().apply("insert")
