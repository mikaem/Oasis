from io import *
from dolfin import assemble, KrylovSolver, LUSolver,  Function, TrialFunction, TestFunction, dx, Vector, GenericMatrix
import sys, json

# Parse command-line keyword arguments
def parse_command_line():
    commandline_kwargs = {}
    for s in sys.argv[1:]:
        if s.count('=') == 1:
            key, value = s.split('=', 1)
        else:
            raise TypeError(s+" Only kwargs separated with '=' sign allowed. See NSdefault_hooks for a range of parameters. Your problem file should contain problem specific parameters.")
        try:
            value = json.loads(value) 
        except ValueError:
            if value in ("True", "False"): # json understands true/false, but not True/False
                value = eval(value)
        commandline_kwargs[key] = value
    return commandline_kwargs

# Create some dictionaries to hold work matrices and functions
class Mat_cache_dict(dict):
    def __missing__(self, key):
        A, sol = (assemble(key), KrylovSolver("cg", "ilu"))
        sol.parameters["report"] = False
        sol.parameters["relative_tolerance"] = 1e-8
        sol.parameters["preconditioner"]["structure"] = "same"
        #A, sol = (assemble(key), LUSolver("mumps"))
        #sol.parameters["reuse_factorization"] = True
        self[key] = (A, sol)
        return self[key]

A_cache = Mat_cache_dict()

class OasisFunction(Function):
    """Function with efficient project method of associated form
    """
    def __init__(self, form, Space, bcs=[], name="x", matvec=None, method="regular"):
        Function.__init__(self, Space, name=name)
        self.form = form
        self.method = method
        self.bcs = bcs
        self.matvec = matvec
        Mass = TrialFunction(Space)*TestFunction(Space)*dx()
        self.bf = form*TestFunction(Space)*dx()
        if method == "regular":
            self.A, self.sol = A_cache[Mass]
            for bc in bcs:
                bc.apply(self.A)
                        
        self.rhs = Vector(self.vector())
        
    def assemble(self):
        if self.method == "regular":
            if self.matvec:
                mat, func = self.matvec
                self.rhs.zero()
                if isinstance(mat, dict):
                    for key, val in mat.iteritems():
                        self.rhs.axpy(1.0, val*func[key].vector())
                elif isinstance(mat, GenericMatrix):
                    self.rhs.axpy(1.0, mat*func)
            else:
                assemble(self.bf, tensor=self.rhs)
                
    def __call__(self):
        self.assemble()
        for bc in self.bcs:
            bc.apply(self.rhs)
        self.sol.solve(self.A, self.vector(), self.rhs)
