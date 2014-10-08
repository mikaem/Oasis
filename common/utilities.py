__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-10-03"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import assemble, KrylovSolver, LUSolver,  Function, TrialFunction, \
    TestFunction, dx, Vector, Matrix, GenericMatrix, FunctionSpace, Timer, div, Form

# Create some dictionaries to hold work matrices
class Mat_cache_dict(dict):
    def __missing__(self, key):
        form, bcs = key
        A, sol = (assemble(form), KrylovSolver("minres", "petsc_amg"))
        sol.parameters["report"] = False
        #sol.parameters["relative_tolerance"] = 1e-8
        sol.parameters["preconditioner"]["structure"] = "same"
        sol.parameters["error_on_nonconvergence"] = False
        #sol.parameters["monitor_convergence"] = True
        #A, sol = (assemble(key), LUSolver("mumps"))
        #sol.parameters["reuse_factorization"] = True
        for bc in bcs:
            bc.apply(A)

        self[key] = (A, sol)
        return self[key]

A_cache = Mat_cache_dict()

def assemble_matrix(form, bcs=[]):
    """Assemble matrix using cache register.
    """
    assert Form(form).rank() == 2
    return A_cache[(form, tuple(bcs))][0]

class OasisFunction(Function):
    """Function with more or less efficient projection methods 
    of associated linear form.
    
    The matvec option is provided for letting the right hand side 
    be computed through a fast matrix vector product. Both the matrix 
    and the Coefficient of the required vector must be provided.
    
      method = "default"
        Solve projection with regular linear algebra
        
      method = "lumping"
        Solve through lumping of mass matrix 
      
    """
    def __init__(self, form, Space, bcs=[], name="x", matvec=[None, None], 
                 method="default"):
        Function.__init__(self, Space, name=name)
        self.form = form
        self.method = method
        self.bcs = bcs
        self.matvec = matvec
        trial = self.trial = TrialFunction(Space)
        test = self.test = TestFunction(Space)
        Mass = trial*test*dx()
        self.bf = form*test*dx()
        self.rhs = Vector(self.vector())
        
        if method.lower() == "default":
            self.A, self.sol = A_cache[(Mass, tuple(bcs))]
                
        elif method.lower() == "lumping":
            assert Space.ufl_element().degree() < 2
            self.A, dummy = A_cache[(Mass, tuple(bcs))]
            ones = Function(Space)
            ones.vector()[:] = 1.
            self.ML = self.A * ones.vector()
            self.ML.set_local(1. / self.ML.array())
                        
    def assemble_rhs(self, u=None):
        """
        Assemble right hand side (form*test*dx) in projection 
        """
        if u: self.matvec[1] = u
        if not self.matvec[0] is None:
            mat, func = self.matvec
            self.rhs.zero()
            self.rhs.axpy(1.0, mat*func.vector())
        else:
            assemble(self.bf, tensor=self.rhs)
                
    def __call__(self, u=None, assemb_rhs=True):
        """
        Compute the projection
        """
        timer = Timer("Projecting {}".format(self.name()))
        if u: self.matvec[1] = u
        if assemb_rhs: self.assemble_rhs()            
        for bc in self.bcs:
            bc.apply(self.rhs)
            
        if self.method.lower() == "default":
            self.sol.solve(self.A, self.vector(), self.rhs)
            
        else:
            self.vector()[:] = self.rhs * self.ML

class GradFunction(OasisFunction):
    """
    Function used for projecting gradients.
    
    Typically used for computing pressure gradient on velocity function space.
    
    """
    def __init__(self, p_, Space, i=0, bcs=[], name="grad", method="default",
                 low_memory_version=False):
        
        assert p_.rank() == 0
        assert i >= 0 and i < Space.mesh().geometry().dim()
        
        OasisFunction.__init__(self, p_.dx(i), Space, bcs=bcs, name=name, 
                               method=method)
        
        Source = p_.function_space()
        if not low_memory_version:
            self.matvec = [A_cache[(self.test*TrialFunction(Source).dx(i)*dx, ())][0], p_]
        
        if method.lower() == "gradient_matrix":
            from fenicstools import compiled_gradient_module
            DG = FunctionSpace(Space.mesh(), 'DG', 0)
            G = assemble(TrialFunction(DG)*self.test*dx())
            dg = Function(DG)
            dP = assemble(TrialFunction(p_.function_space()).dx(i)*TestFunction(DG)*dx())
            self.WGM = compiled_gradient_module.compute_weighted_gradient_matrix(G, dP, dg)
           
    def __call__(self, u=None, assemb_rhs=True):
        if u: self.matvec[1] = u
        if self.method.lower() == "gradient_matrix":    
            self.vector()[:] = self.WGM * self.matvec[1].vector()
        else:
            OasisFunction.__call__(self, assemb_rhs=assemb_rhs)

class DivFunction(OasisFunction):
    """
    Function used for projecting divergence of vector.
    
    Typically used for computing divergence of velocity on pressure function space.
    
    """
    def __init__(self, u_, Space, bcs=[], name="div", method="default",
                 low_memory_version=False):
        
        OasisFunction.__init__(self, div(u_), Space, bcs=bcs, name=name, 
                               method=method)
        
        Source = u_[0].function_space()
        if not low_memory_version:
            self.matvec = [[A_cache[(self.test*TrialFunction(Source).dx(i)*dx, ())][0], u_[i]] 
                           for i in range(Space.mesh().geometry().dim())]
        
        if method.lower() == "gradient_matrix":
            from fenicstools import compiled_gradient_module
            DG = FunctionSpace(Space.mesh(), 'DG', 0)
            G = assemble(TrialFunction(DG)*self.test*dx())
            dg = Function(DG)
            self.WGM = []
            st = TrialFunction(Source)
            for i in range(Space.mesh().geometry().dim()):
                dP = assemble(st.dx(i)*TestFunction(DG)*dx)
                A = Matrix(G)
                self.WGM.append(compiled_gradient_module.compute_weighted_gradient_matrix(A, dP, dg))

    def assemble_rhs(self, use_matvec=True):
        """
        Assemble right hand side (form*test*dx) in projection 
        """
        if use_matvec:
            self.rhs.zero()
            for mat, vec in self.matvec:
                self.rhs.axpy(1.0, mat*vec.vector())
                
        else:
            assemble(self.bf, tensor=self.rhs)     
     
    def __call__(self, assemb_rhs=True):
        if self.method.lower() == "gradient_matrix":
            if assemb_rhs: self.assemble_rhs()
            self.vector().zero()
            for i in range(self.function_space().mesh().geometry().dim()):
                self.vector().axpy(1., self.WGM[i] * self.matvec[i][1].vector())

        else:
            OasisFunction.__call__(self, assemb_rhs=assemb_rhs)
       