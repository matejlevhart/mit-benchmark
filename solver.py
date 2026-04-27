from petsc4py import PETSc
print = PETSc.Sys.Print
import numpy as np
import firedrake as fd
import make_geometry
from firedrake import grad, div, inner, dx

# TODO: add measurement of relevant data
#mesh, bndry = make_geometry.dfg_bench(h=0.01, degree=1)
mesh = fd.RectangleMesh(10, 80, 1.0, 8,0) # simpler, doesn't have points for measuring
bndry = {"heat":[1], "cool":[2], "wall":[3,4]}

Ra = 3.4e6             # value from paper = 3.4e5
Pr = 0.71 # value from paper
K1 = fd.Constant(np.sqrt(Pr/Ra))
K2 = fd.Constant(1/np.sqrt((Ra*Pr)))
T_left = fd.Constant(0.5)
T_right = fd.Constant(-0.5)
e_y = fd.Constant((0, 1))
t = 0.0
t_end = 5.0
dt = 0.01


theta = fd.Constant(0.5) # Crank-Nicolson time stepping parameter

# Define function spaces
Ep = fd.FiniteElement("CG", mesh.ufl_cell(), 1)
Ev = fd.VectorElement("CG", mesh.ufl_cell(), 2)
ET = fd.FiniteElement("CG", mesh.ufl_cell(), 2)
W   = fd.FunctionSpace(mesh, fd.MixedElement([Ev, Ep, ET]))

# define test functions:
(v, q, S) = fd.TestFunctions(W)

# current unknown time step
w = fd.Function(W)
(u, p, T) = fd.split(w)
# previous known time step
w0 = fd.Function(W)
(u0, p0, T0) = fd.split(w0)

# initial conditions: u = 0, T = 0
# variational form without time derivative in current time
def a(u,v):
    return (inner(grad(u)*u, v)*dx
        + K1 * inner(grad(u), grad(v))*dx
        - T*inner(e_y, v)*dx)
def b(u,q):
    return inner(div(u), q)*dx
def c(u, T, S):
    return ( inner(u, grad(T))*S*dx + inner(K2*grad(T), grad(S))*dx) # - konstanta *inner(sym(grad(v),sym(grad(v))))*S


Feq1 = a(u,v) + c(u, T, S)
Feq2 = b(u, q) - b(v, p)
# part of the equation without Lagrange multipliers
F = Feq1

# variational form without time derivative in previous time
F0 = fd.replace(F, {w: w0})
udot = fd.Constant(1.0/dt) * inner(u - u0, v)*dx
Tdot = fd.Constant(1.0/dt) * inner(T - T0, S)*dx

F = udot + Tdot + theta*F + (1-theta)*F0 + Feq2
J = fd.derivative(F, w)
nullsp = fd.MixedVectorSpaceBasis(W, [W.sub(0), fd.VectorSpaceBasis(constant=True, comm=mesh.comm), W.sub(2)])
# Boundary conditions
bcs = [fd.DirichletBC(W.sub(0), fd.Constant((0, 0)), [1, 2, 3, 4]),
    fd.DirichletBC(W.sub(2), T_left, [1]),
    fd.DirichletBC(W.sub(2), T_right, [2])]

lu = {
    "snes_monitor": "",
    "snes_type": "newtonls",
    "snes_max_it": 40,
    "snes_rtol": 1e-10,
    "snes_atol": 1e-10,
    "snes_linesearch_type": "basic",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}

problem = fd.NonlinearVariationalProblem(F, w, bcs,J)
solver = fd.NonlinearVariationalSolver(problem,nullspace=nullsp, solver_parameters=lu, options_prefix="")

bcs[1].apply(w)
bcs[2].apply(w)

(u, p, T) = w.subfunctions
u.rename("velocity")
p.rename("pressure")
T.rename("temperature")

rfile = fd.VTKFile("results/rayleigh_benard.pvd")
rfile.write(u,p,T, time=t)
#time stepping

while t<t_end :

    w0.assign(w)
    t += dt
    print(f"{t=}")
    solver.solve()

    rfile.write(u,p,T, time=t)

