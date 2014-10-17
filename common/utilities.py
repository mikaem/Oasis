__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-10-03"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import assemble, KrylovSolver, LUSolver,  Function, TrialFunction, \
    TestFunction, dx, Vector, Matrix, GenericMatrix, FunctionSpace, Timer, div, \
    Form, Coefficient, inner

# Create some dictionaries to hold work matrices
class Mat_cache_dict(dict):
    """Items in dictionary are matrices and solvers for efficient
    reuse during, e.g., multiple projections to the same space.
    """
    def __missing__(self, key):
        form, bcs = key
        A = assemble(form)        
        for bc in bcs:
            bc.apply(A)

        self[key] = A
        return self[key]

A_cache = Mat_cache_dict()

def assemble_matrix(form, bcs=[]):
    """Assemble matrix using cache register.
    """
    assert Form(form).rank() == 2
    return A_cache[(form, tuple(bcs))]

class OasisFunction(Function):
    """Function with more or less efficient projection methods 
    of associated linear form.
    
    The matvec option is provided for letting the right hand side 
    be computed through a fast matrix vector product. Both the matrix 
    and the Coefficient of the required vector must be provided.
    
      method = "default"
        Solve projection with regular linear algebra using solver_type
        and preconditioner_type
        
      method = "lumping"
        Solve through lumping of mass matrix 
      
    """
    def __init__(self, form, Space, bcs=[], 
                 name="x", 
                 matvec=[None, None], 
                 method="default", 
                 solver_type="cg", 
                 preconditioner_type="default"):
        
        Function.__init__(self, Space, name=name)
        self.form = form
        self.method = method
        self.bcs = bcs
        self.matvec = matvec
        self.trial = trial = TrialFunction(Space)
        self.test = test = TestFunction(Space)
        Mass = inner(trial, test)*dx()
        self.bf = inner(form, test)*dx()
        self.rhs = Vector(self.vector())
        
        if method.lower() == "default":
            self.A = A_cache[(Mass, tuple(bcs))]
            self.sol = KrylovSolver(solver_type, preconditioner_type)
            self.sol.parameters["preconditioner"]["structure"] = "same"
            self.sol.parameters["error_on_nonconvergence"] = False
            self.sol.parameters["monitor_convergence"] = False
            self.sol.parameters["report"] = False
                
        elif method.lower() == "lumping":
            assert Space.ufl_element().degree() < 2
            self.A = A_cache[(Mass, tuple(bcs))]
            ones = Function(Space)
            ones.vector()[:] = 1.
            self.ML = self.A * ones.vector()
            self.ML.set_local(1. / self.ML.array())
                        
    def assemble_rhs(self):
        """
        Assemble right hand side (form*test*dx) in projection 
        """
        if not self.matvec[0] is None:
            mat, func = self.matvec
            self.rhs.zero()
            self.rhs.axpy(1.0, mat*func.vector())
        
        else:
            assemble(self.bf, tensor=self.rhs)
                
    def __call__(self, assemb_rhs=True):
        """
        Compute the projection
        """
        timer = Timer("Projecting {}".format(self.name()))
        
        if assemb_rhs: 
            self.assemble_rhs()
        
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
    def __init__(self, p_, Space, i=0, bcs=[], name="grad", method={}):
        
        assert p_.rank() == 0
        assert i >= 0 and i < Space.mesh().geometry().dim()
        
        solver_type = method.get('solver_type', 'cg')
        preconditioner_type = method.get('preconditioner_type', 'default')
        solver_method = method.get('method', 'default')
        low_memory_version = method.get('low_memory_version', False)
        
        OasisFunction.__init__(self, p_.dx(i), Space, bcs=bcs, name=name, 
                               method=solver_method, solver_type=solver_type,
                               preconditioner_type=preconditioner_type)
        
        self.i = i
        Source = p_.function_space()
        if not low_memory_version:
            self.matvec = [A_cache[(self.test*TrialFunction(Source).dx(i)*dx, ())], p_]
        
        if solver_method.lower() == "gradient_matrix":
            from fenicstools import compiled_gradient_module
            DG = FunctionSpace(Space.mesh(), 'DG', 0)
            G = assemble(TrialFunction(DG)*self.test*dx())
            dg = Function(DG)
            dP = assemble(TrialFunction(p_.function_space()).dx(i)*TestFunction(DG)*dx())
            self.WGM = compiled_gradient_module.compute_weighted_gradient_matrix(G, dP, dg)

    def assemble_rhs(self, u=None):
        """
        Assemble right hand side trial.dx(i)*test*dx.
        
        Possible Coefficient u may replace p_ and makes it possible
        to use this Function to compute both grad(p) and grad(dp), i.e., 
        the gradient of pressure correction.
        
        """
        if isinstance(u, Coefficient): 
            self.matvec[1] = u
            self.bf = u.dx(self.i)*self.test*dx()
            
        if not self.matvec[0] is None:
            mat, func = self.matvec
            self.rhs.zero()
            self.rhs.axpy(1.0, mat*func.vector())
        else:
            assemble(self.bf, tensor=self.rhs)
           
    def __call__(self, u=None, assemb_rhs=True):
        if isinstance(u, Coefficient): 
            self.matvec[1] = u
            self.bf = u.dx(self.i)*self.test*dx()

        if self.method.lower() == "gradient_matrix":    
            self.vector()[:] = self.WGM * self.matvec[1].vector()
        else:
            OasisFunction.__call__(self, assemb_rhs=assemb_rhs)

class DivFunction(OasisFunction):
    """
    Function used for projecting divergence of vector.
    
    Typically used for computing divergence of velocity on pressure function space.
    
    """
    def __init__(self, u_, Space, bcs=[], name="div", method={}):

        solver_type = method.get('solver_type', 'cg')
        preconditioner_type = method.get('preconditioner_type', 'default')
        solver_method = method.get('method', 'default')
        low_memory_version = method.get('low_memory_version', False)
        
        OasisFunction.__init__(self, div(u_), Space, bcs=bcs, name=name, 
                               method=solver_method, solver_type=solver_type,
                               preconditioner_type=preconditioner_type)
        
        Source = u_[0].function_space()
        if not low_memory_version:
            self.matvec = [[A_cache[(self.test*TrialFunction(Source).dx(i)*dx, ())], u_[i]] 
                           for i in range(Space.mesh().geometry().dim())]
        
        if solver_method.lower() == "gradient_matrix":
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

    def assemble_rhs(self):
        """
        Assemble right hand side (form*test*dx) in projection 
        """
        if not self.matvec[0] is None:
            self.rhs.zero()
            for mat, vec in self.matvec:
                self.rhs.axpy(1.0, mat*vec.vector())
                
        else:
            assemble(self.bf, tensor=self.rhs)     
     
    def __call__(self, assemb_rhs=True):
        
        if self.method.lower() == "gradient_matrix":
            # Note that assembling rhs is not necessary using gradient_matrix
            if assemb_rhs: self.assemble_rhs()
            self.vector().zero()
            for i in range(self.function_space().mesh().geometry().dim()):
                self.vector().axpy(1., self.WGM[i] * self.matvec[i][1].vector())

        else:
            OasisFunction.__call__(self, assemb_rhs=assemb_rhs)
       