__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-04"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..NSCoupled import *

# Reuse some code from NSfracStep case
from ..NSfracStep.Cylinder import mesh, H, L, D, center, case, Inlet, \
  Wall, Cyl, Outlet, post_import_problem

# Override some problem specific parameters
NS_parameters.update(
    omega = 1.0,
    max_iter = 100,
    plot_interval = 10,
    velocity_degree = 2)

def create_bcs(VQ, Um, **NS_namespace):
    inlet = Expression(("4.*{0}*x[1]*({1}-x[1])/pow({1}, 2)".format(Um, H), "0"))
    ux = Expression(("0.00*x[1]", "-0.00*(x[0]-{})".format(center)))
    bc0 = DirichletBC(VQ.sub(0), inlet, Inlet)
    bc1 = DirichletBC(VQ.sub(0), ux, Cyl)
    bc2 = DirichletBC(VQ.sub(0), (0, 0), Wall)
    return dict(up = [bc0, bc1, bc2])

def theend_hook(u_, p_, up_, mesh, ds, VQ, nu, Umean, **NS_namespace):
    plot(u_, title='Velocity')
    plot(p_, title='Pressure')

    R = VectorFunctionSpace(mesh, 'R', 0)
    c = TestFunction(R)
    tau = -p_*Identity(2)+nu*(grad(u_)+grad(u_).T)
    ff = FacetFunction("size_t", mesh, 0)
    Cyl.mark(ff, 1)
    n = FacetNormal(mesh)
    ds = ds[ff]
    forces = assemble(dot(dot(tau, n), c)*ds(1)).array()*2/Umean**2/D
    
    print "Cd = {}, CL = {}".format(*forces)

    from fenicstools import Probes
    from numpy import linspace, repeat, where, resize
    xx = linspace(0, L, 10000)
    x = resize(repeat(xx, 2), (10000, 2))
    x[:, 1] = 0.2
    probes = Probes(x.flatten(), VQ)
    probes(up_)
    nmax = where(probes.array()[:, 0] < 0)[0][-1]
    print "L = ", x[nmax, 0]-0.25
    print "dP = ", up_(Point(0.15, 0.2))[2] - up_(Point(0.25, 0.2))[2]
    print "Global divergence error ", assemble(dot(u_, n)*ds()), assemble(div(u_)*div(u_)*dx())
