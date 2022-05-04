from fenics import MPI, Mesh, MeshValueCollection, XDMFFile, cpp, Measure, assemble, Constant
import numpy as np

"""
Author : Kei Yamamoto <keiya@math.uio.no>
Here, we assume that mesh files are placed inside mesh folder. 
You can download mesh file from https://drive.google.com/drive/folders/1YWCEOJ5vnuNpcLkpiofD54xi56QH1cLQ?usp=sharing
"""

mesh = Mesh()
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile(MPI.comm_world, "mesh/StraightPipe/mesh.xdmf") as infile:
   infile.read(mesh)
   infile.read(mvc, "name_to_read")

mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile(MPI.comm_world, "mesh/StraightPipe/mf.xdmf") as infile:
    infile.read(mvc, "name_to_read")

mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
dx = Measure("dx")(subdomain_data=mf)

# create mesh functions for the outlet mesh
outlet_mesh = Mesh()
outlet_mvc = MeshValueCollection("size_t", outlet_mesh, outlet_mesh.topology().dim())
with XDMFFile(MPI.comm_world, "mesh/Outlet/mesh.xdmf") as outlet_infile:
   outlet_infile.read(outlet_mesh)
   outlet_infile.read(outlet_mvc, "name_to_read")

outlet_mvc = MeshValueCollection("size_t", outlet_mesh, outlet_mesh.topology().dim()-1)
with XDMFFile(MPI.comm_world, "mesh/Outlet/mf.xdmf") as outlet_infile:
    outlet_infile.read(outlet_mvc, "name_to_read")

outlet_mf = cpp.mesh.MeshFunctionSizet(outlet_mesh, outlet_mvc)

comm = MPI.comm_world
local_xmin = mesh.coordinates()[:, 0].min()
local_xmax = mesh.coordinates()[:, 0].max()
local_ymin = mesh.coordinates()[:, 1].min()
local_ymax = mesh.coordinates()[:, 1].max()
local_zmin = mesh.coordinates()[:, 2].min()
local_zmax = mesh.coordinates()[:, 2].max()
xmin = comm.gather(local_xmin, 0)
xmax = comm.gather(local_xmax, 0)
ymin = comm.gather(local_ymin, 0)
ymax = comm.gather(local_ymax, 0)
zmin = comm.gather(local_zmin, 0)
zmax = comm.gather(local_zmax, 0)

local_num_cells = mesh.num_cells()
local_num_edges = mesh.num_edges()
local_num_faces = mesh.num_faces()
local_num_facets = mesh.num_facets()
local_num_vertices = mesh.num_vertices()
num_cells = comm.gather(local_num_cells, 0)
num_edges = comm.gather(local_num_edges, 0)
num_faces = comm.gather(local_num_faces, 0)
num_facets = comm.gather(local_num_facets, 0)
num_vertices = comm.gather(local_num_vertices, 0)
volume = assemble(Constant(1) * dx(mesh))

if MPI.rank(MPI.comm_world) == 0:
    print("=== Mesh information ===")
    print("X range: {} to {} (delta: {:.4f})".format(min(xmin), max(xmax), max(xmax) - min(xmin)))
    print("Y range: {} to {} (delta: {:.4f})".format(min(ymin), max(ymax), max(ymax) - min(ymin)))
    print("Z range: {} to {} (delta: {:.4f})".format(min(zmin), max(zmax), max(zmax) - min(zmin)))
    print("Number of cells: {}".format(sum(num_cells)))
    print("Number of cells per processor: {}".format(int(np.mean(num_cells))))
    print("Number of edges: {}".format(sum(num_edges)))
    print("Number of faces: {}".format(sum(num_faces)))
    print("Number of facets: {}".format(sum(num_facets)))
    print("Number of vertices: {}".format(sum(num_vertices)))
    print("Volume: {:.4f}".format(volume))
    print("Number of cells per volume: {:.4f}".format(sum(num_cells) / volume))
    print("========================")