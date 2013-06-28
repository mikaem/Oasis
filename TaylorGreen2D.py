__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from Oasis import *

# Create a mesh here
mesh = UnitSquareMesh(50, 50)
scale = 2*(mesh.coordinates() - 0.5)*pi
mesh.coordinates()[:, :] = scale
del scale

class PeriodicDomain(SubDomain):
    
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], -pi) or near(x[1], -pi)) and 
                (not ((near(x[0], -pi) and near(x[1], pi)) or 
                        (near(x[0], pi) and near(x[1], -pi)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], pi) and near(x[1], pi):
            y[0] = x[0] - 2.0*pi
            y[1] = x[1] - 2.0*pi
        elif near(x[0], pi):
            y[0] = x[0] - 2.0*pi
            y[1] = x[1]
        else: #near(x[1], pi)
            y[0] = x[0]
            y[1] = x[1] - 2.0*pi

constrained_domain = PeriodicDomain()

# Override some problem specific parameters and put the variables in DC_dict
T = 10.
dt = 0.1
folder = "taylorgreen2D_results"
newfolder = create_initial_folders(folder, dt)
NS_parameters.update(dict(
    nu = 0.01,
    T = T,
    dt = dt,
    folder = folder,
    max_iter = 5,
    newfolder = newfolder,
    use_krylov_solvers = True,
    use_lumping_of_mass_matrix = False,
    monitor_convergence = False,
    krylov_report = False    
  )
)

if NS_parameters['velocity_degree'] > 1:
    NS_parameters['use_lumping_of_mass_matrix'] = False

def pre_solve_hook(q_, p_, **NS_namespace):    
    velocity_plotter0 = VTKPlotter(q_['u0'])
    velocity_plotter1 = VTKPlotter(q_['u1'])
    pressure_plotter = VTKPlotter(p_) 
    return dict(velocity_plotter0=velocity_plotter0, 
                velocity_plotter1=velocity_plotter1, 
                pressure_plotter=pressure_plotter)

initial_fields = dict(
        u0='2./sqrt(3.)*sin(2.*pi/3.)*sin(x[0])*cos(x[1])',
        u1='2./sqrt(3.)*sin(-2.*pi/3.)*cos(x[0])*sin(x[1])', 
        p='0')
    
def initialize(q_, q_1, q_2, VV, sys_comp, **NS_namespace):
    for ui in sys_comp:
        vv = project(Expression((initial_fields[ui])), VV[ui])
        q_[ui].vector()[:] = vv.vector()[:]
        if not ui == 'p':
            q_1[ui].vector()[:] = q_[ui].vector()[:]
            q_2[ui].vector()[:] = q_[ui].vector()[:]

def temporal_hook(velocity_plotter0, velocity_plotter1,
                           pressure_plotter, **NS_namespace):
    pressure_plotter.plot()
    velocity_plotter0.plot()
    velocity_plotter1.plot()
