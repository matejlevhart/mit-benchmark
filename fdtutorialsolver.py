from firedrake import *
from petsc4py import PETSc

N = 128
M = UnitSquareMesh(N, N)

V = VectorFunctionSpace(M, "CG", 2)
W = FunctionSpace(M, "CG", 1)
Q = FunctionSpace(M, "CG", 1)
Z = V * W * Q

upT = Function(Z)
u, p, T = split(upT)
v, q, S = TestFunctions(Z)

Ra = Constant(200.0)
Pr = Constant(6.8)
g = Constant((0, -9.81))

F = (inner(grad(u), grad(v))*dx
    + inner(dot(grad(u), u), v)*dx
    - inner(p, div(v))*dx
    - (Ra/Pr)*inner(T*g, v)*dx
    + inner(div(u), q)*dx
    + inner(dot(grad(T), u), S)*dx
    + 1/Pr * inner(grad(T), grad(S))*dx
     )

bcs = [
    DirichletBC(Z.sub(0), Constant((0,0)), (1, 2, 3, 4)),
    DirichletBC(Z.sub(2), Constant(1.0), (1, )),
    DirichletBC(Z.sub(2), Constant(0.0), (2, ))
        ]

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant = True), Z.sub(2)])
solve(F == 0, upT, bcs=bcs, nullspace=nullspace,
      solver_parameters={"mat_type":"aij"
                         "snes_monitor": None,
                         "ksp_type":"gmres",
                         "pc_type":"lu",
                         "pc_factor_mat_solver_type":"mumps"
                         })

