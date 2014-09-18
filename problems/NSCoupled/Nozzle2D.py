from ..NSCoupled import *
from ..Nozzle2D import *

from math import sqrt, pi
from fenicstools import StructuredGrid, StatisticsProbes
import sys
from numpy import array, linspace
from scitools.std import plot as sciplot

# Override some problem specific parameters
re_high = False
NS_parameters.update(
    omega = 0.4,
    nu = 0.0035 / 1056.,     
    folder = "nozzle_results",
    max_error = 1e-13,
    velocity_degree = 2,
    max_iter = 25,
    re_high = re_high)

def create_bcs(VQ, mesh, sys_comp, re_high, **NS_namespce):
    #Q = 5.21E-6 if not re_high else 6.77E-5  # From FDA
    Q = 5.21E-6 if not re_high else 3E-5  # From FDA
    r_0 = 0.006
    u_maks = Q / (4.*r_0*r_0*(1.-2./pi))  # Analytical, could be more exact numerical, different r_0
    #inn = Expression(("u_maks * cos(sqrt(pow(x[1],2))/r_0/2.*pi)", "0"), u_maks=u_maks, r_0=r_0)
    inn = Expression(("u_maks * (1-x[1]*x[1]/r_0/r_0)", "0"), u_maks=u_maks, r_0=r_0)
    
    bc0 = DirichletBC(VQ.sub(0),    inn,  inlet)
    bc1 = DirichletBC(VQ.sub(0), (0, 0),  walls)
    bc2 = DirichletBC(VQ.sub(0).sub(1), 0, centerline)
    
    return dict(up=[bc0, bc1, bc2])  
      
def pre_solve_hook(mesh, V, **NS_namespace):
   
    # Normals and facets to compute flux at inlet and outlet
    normal = FacetNormal(mesh)
    Inlet = AutoSubDomain(inlet)
    Outlet = AutoSubDomain(outlet)
    Walls = AutoSubDomain(walls)
    Centerline = AutoSubDomain(centerline)
    facets = FacetFunction('size_t', mesh, 0)
    Inlet.mark(facets, 1)
    Outlet.mark(facets, 2)
    Walls.mark(facets, 3)
    Centerline.mark(facets, 4)
    
    z_senterline = linspace(-0.18269,0.320,1000)
    x = array([[i, 0.0] for i in z_senterline])
    senterline = StatisticsProbes(x.flatten(), V)

    return dict(uv=Function(V), senterline=senterline, facets=facets,
                normal=normal)
    
def temporal_hook(**NS_namespace):
    pass
    #TODO: check stats with fenicstools
    # - Compute velocity at z: -0.088, -0.064, -0.048, -0.02, 0.008, 0.0, 0.008, 0.016, 0.024, 0.032, 0.06, 0.08
    # - Chech volumetric folow rate trough this points
    # - Axial velocity 
    # - Wall pressure
    # - Axial velocity(r) at each point 
    # - Shear stress(r) at each point
    # - Wall shear stress
    # - Mean error metric
    # - Use kill_oasis when the stats don't change
    # - Start update stats after some time T?
    # - What is the lowest mesh that gives ok results?
    # - Lowest dt with out crashing
    
    ## save plot
    #if tstep % plot_interval == 0:
        #uv.assign(project(u_, Vv))
        #file = File(newfolder + "/VTK/nozzle_velocity_%05d.pvd" % (tstep / plot_interval))
    #file << uv
    
    ## print flux
    #if tstep % check_flux == 0:
        #inlet_flux = assemble(dot(u_, normal)*ds(1), exterior_facet_domains=facets)
    #outlet_flux = assemble(dot(u_, normal)*ds(2), exterior_facet_domains=facets)
    #rel_err = (abs(inlet_flux) - abs(outlet_flux)) / abs(inlet_flux)
        #if MPI.process_number() == 0:
           #info_red("Flux in:         %e\nFlux out:        %e\nRelativ error:   %e\ntstep:           %d" % (inlet_flux, outlet_flux, rel_err, tstep))
    
    ## save stats
    #if tstep % save_statistics == 0:
    #statsfolder = path.join(newfolder, "Stats")
    #for i in range(len(stats)):
        #stats[i].toh5(0, tstep, filename=statsfolder+"/dump_mean_%d_%d.h5" % (i, tstep))
    
    ##senterline.toh5(0, tstep, filename=statsfolder+"/dump_mean_senterline_%d.h5" % tstep)
    
    ## update stats
    #if tstep % update_statistics == 0: # and tstep/dt > 4 (or some time):
    #for i in range(len(stats)):
        #stats[i](q_['u0'], q_['u1'], q_['u2'])
    #senterline(q_['u2'])
    
