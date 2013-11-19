__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

# This is a possible extension of the channel case using one additional scalar
# You don't have to use two separate files to use scalars, I just resuse
# as much as possible of the 'clean' Navier Stokes setup here.

from Channel import *

NS_parameters['folder'] = 'channelscalar_results'

# Declare two scalar fields with different diffusivities
scalar_components = ['c']
Schmidt['c'] = 1.
    
# Specify additional boundary conditions for scalar
create_bcs0 = create_bcs
def create_bcs(V, q_, q_1, q_2, sys_comp, u_components, **NS_namespace):
    bcs = create_bcs0(V, q_, q_1, q_2, sys_comp, u_components, **NS_namespace)
    bcs['c'] = [DirichletBC(V, Constant(1), walls)]        
    return bcs
    
initialize0 = initialize
def initialize(V, Vv, q_, q_1, q_2, bcs, restart_folder, **NS_namespace):
    kw = initialize0(V, Vv, q_, q_1, q_2, bcs, restart_folder, **NS_namespace)
    if restart_folder is None:
        bcs['c'][0].apply(q_['c'].vector())
    return kw

temporal_hook0 = temporal_hook
def temporal_hook(q_, u_, V, Vv, tstep, uv, voluviz, stats,
                  update_statistics, check_save_h5, newfolder, **NS_namespace):
    temporal_hook0(q_, u_, V, Vv, tstep, uv, voluviz, stats,
                   update_statistics, check_save_h5, newfolder, **NS_namespace)
    if tstep % check_save_h5 == 0:
        voluviz(q_['c'])
        h5folder = path.join(newfolder, "Voluviz")
        voluviz.toh5(0, tstep, filename=h5folder+"/snapshot_c_{}.h5".format(tstep))
        voluviz.probes.clear()
        
        plot(q_['c'], title='scalar')
