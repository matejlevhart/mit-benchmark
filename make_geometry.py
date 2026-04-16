import firedrake as fd
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import netgen
from netgen.occ import *

print_= print
print = PETSc.Sys.Print
comm = MPI.COMM_WORLD

# Corners of the box.
x0 = 0.0
x1 = 1.0
y0 = 0.0
y1 = 8.0

def marks_netgen_to_firedrake(ngmsh):
    bndry=dict()
    region_names=ngmsh.GetRegionNames(codim=1)
    names=set(region_names)
    arr=np.array(region_names)
    for name in names:
        indices=np.where(arr==name)[0]+1
        bndry[name]=indices.tolist()
    return(bndry)

def dfg_bench(h=0.01, degree=1):
    ngmsh = netgen.libngpy._meshing.Mesh(2)  #2-dim
    bndry = {}
    if comm.rank == 0:
        rec = Rectangle(x1,y1).Face()
        rec.edges.Min(X).name = "heat"
        rec.edges.Max(X).name = "cool"
        rec.edges.Min(Y).name = "wall"
        rec.edges.Max(Y).name = "wall"

        # Points for measuring (see paper by M.A. Christon 2001)
        p1 = Vertex(Pnt(0.181,7.37,0.0))
        p2 = Vertex(Pnt(0.819,0.63,0.0))
        p3 = Vertex(Pnt(0.181,0.63,0.0))
        p4 = Vertex(Pnt(0.819,7.37,0.0))
        p5 = Vertex(Pnt(0.181,4.0,0.0))

        geo = OCCGeometry(rec, dim=2)

        ngmsh = geo.GenerateMesh(maxh=h)
        bndry = marks_netgen_to_firedrake(ngmsh)        

    # distribute the bndry info to all cpus
    bndry=comm.bcast(bndry, root=0)

    mesh = fd.Mesh(ngmsh, name="dfg")
    if degree>1 :
        cf = mesh.curve_field(degree)
        mesh = fd.Mesh(cf, name=f"dfg_{degree}")
    return(mesh, bndry)

if __name__ == "__main__":
    import matplotlib.pyplot as plt


    mesh, bndry = dfg_bench(h=0.1, degree=1)

    print(f"{bndry=}")

    fig, axes = plt.subplots()
    fig.set_size_inches(10, 4)

    axes.set_aspect(1)
    fd.triplot(mesh, axes=axes)
    
    plt.savefig(f'fig_dfg_mesh.pdf', bbox_inches='tight')
