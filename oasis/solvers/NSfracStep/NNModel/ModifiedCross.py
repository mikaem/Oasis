__author__ = 'Anna Haley <ahaley@mie.utoronto.ca>'
__date__ = '2020-05-04'
__copyright__ = 'Copyright (C) 2020 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

from dolfin import (Function, FunctionSpace, assemble, TestFunction, sym, grad, inner, sqrt,
    DirichletBC, Constant)

from .common import derived_bcs

__all__ = ['nn_setup', 'nn_update']


def nn_setup(u_, mesh, ModifiedCross, CG1Function, nu_nn_krylov_solver, bcs, **NS_namespace):
    """
    Set up for solving Modified-Cross non-Newtonian model.
    """
    CG1 = FunctionSpace(mesh, "CG", 1)
    mu_inf = ModifiedCross['mu_inf']
    mu_o = ModifiedCross['mu_o']
    rho = ModifiedCross['rho']
    lam = ModifiedCross['lam'] 
    m_param = ModifiedCross['m_param'] 
    a_param = ModifiedCross['a_param']
    # Set up Modified Cross form
    Sij = sym(grad(u_))
    SII = sqrt(2 * inner(Sij, Sij))
    nu_nn_form = ((mu_o-mu_inf)/((1+(lam*SII)**m_param)**a_param) + mu_inf)/rho
    #bcs_nu_nn = derived_bcs(CG1, bcs['u0'], u_)
    nunn_ = CG1Function(nu_nn_form, mesh, method=nu_nn_krylov_solver, bounded=True, name="nu_nn")
    return dict(Sij=Sij, nunn_=nunn_, bcs_nu_nn=[])

def nn_update(nunn_, **NS_namespace):
    """Compute nunn_"""
    nunn_()
