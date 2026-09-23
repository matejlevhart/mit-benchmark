from petsc4py import PETSc
print = PETSc.Sys.Print
import numpy as np
import firedrake as fd
import make_geometry
import matplotlib.pyplot as plt
from firedrake import grad, div, inner, dx, sym, ds


n = 10

mesh, bndry = make_geometry.dfg_bench(h=1/(10*n), degree=1) 
print(f"Number of faces: {mesh.num_cells()}")
#mesh = fd.RectangleMesh(n*10, n*80, 1.0, 8,0) # simpler, doesn't have points for measuring
#bndry = {"heat":[1], "cool":[2], "wall":[3,4]}

Ra = 3.4e5             # value from paper = 3.4e5
Pr = 0.71              # value from paper
K1 = fd.Constant(np.sqrt(Pr/Ra))
K2 = fd.Constant(1/np.sqrt((Ra*Pr)))
T_left = fd.Constant(0.5)
T_right = fd.Constant(-0.5)
e_y = fd.Constant((0, 1))
t = 0.0
t_end = 180.0
dt = 0.1
t_measure = 100.0



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
    return ( inner(u, grad(T))*S*dx + inner(K2*grad(T), grad(S))*dx)


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
# boundary conditions
bcs = [fd.DirichletBC(W.sub(0), fd.Constant((0, 0)), [1, 2, 3, 4]),
    fd.DirichletBC(W.sub(2), T_left, [4]),
    fd.DirichletBC(W.sub(2), T_right, [2])]

lu = {
    "snes_monitor": "",
    "snes_type": "newtonls",
    "snes_max_it": 40,
    "snes_rtol": 1e-8,
    "snes_atol": 1e-8,
    "snes_linesearch_type": "bt",
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

n_vec = fd.FacetNormal(mesh)

def compute_nusselt(T_func, wall_tag, H, sign=1.0):
     integrand = inner(grad(T_func), n_vec) * ds(wall_tag)
     return sign / H * fd.assemble(integrand)
     

# measuring
a1 = [0.181, 7.37]
a2 = [0.819, 0.63]
a3 = [0.181, 0.63]
a4 = [0.819, 7.37]
a5 = [0.181, 4.0]

T_eval = fd.PointEvaluator(mesh, [a1,a2])
vel_eval = fd.PointEvaluator(mesh, [a1])

Ts = []
Eps = []
us = []
vs = []
Nu_hot = []
Nu_cold = []


#time stepping
while t<t_end :

    w0.assign(w)
    t += dt
    print(f"{t=}")
    solver.solve()

    if t >= t_measure:  
            TT = T_eval.evaluate(T)
            Ts.append(TT[0])

            Epss = TT[0] + TT[1]
            Eps.append(Epss)

            u_vec = vel_eval.evaluate(u)[0]

            u1 = u_vec[0]
            v1 = u_vec[1]

            us.append(u1)
            vs.append(v1)

            Nu_h = compute_nusselt(T, wall_tag=4, H=8, sign=1.0)
            Nu_c = compute_nusselt(T, wall_tag=2, H=8, sign=-1.0)
            Nu_hot.append(Nu_h)
            Nu_cold.append(Nu_c)
            #print(f"Ts: {Ts}")
            #print(f"u1: {us}")
            #print(f"u2: {vs}")

    #rfile.write(u,p,T, time=t)


# Constants for measuring/comparing:
T_goal = 0.2647
v_goal = 0.0572
Nu_goal = 4.5791

T_avg = sum(Ts)/len(Ts)
T_deviation = [T - T_avg for T in Ts]

u_avg = sum(us) / len(us)
v_avg = sum(vs) / len(vs)

Nu_hot_avg  = sum(Nu_hot)  / len(Nu_hot)
Nu_cold_avg = sum(Nu_cold) / len(Nu_cold)
Nu_avg = 0.5 * (Nu_hot_avg + Nu_cold_avg)

# plots
print(f"\n--- Results ---")
print(f"average temperature at point-1     = {T_avg:.4f}")
print(f"average temperature relative error = {abs(T_avg - T_goal)/T_goal:.4f}")
print(f"average velocity magnitude 1       = {u_avg:.4f}")
print(f"average velocity magnitude 2       = {v_avg:.4f}")
print(f"average velocity relative error    = {abs(u_avg - v_goal)/v_goal:.4f}")
print(f"average Nu (hot wall)              = {Nu_hot_avg:.4f}")
print(f"average Nu (cold wall)             = {Nu_cold_avg:.4f}")
print(f"average Nu (mean of walls)         = {Nu_avg:.4f}")
print(f"Nu relative error                  = {abs(Nu_avg - Nu_goal)/Nu_goal:.4f}")
 
# --- Plot: temperature ---
plt.figure()
plt.plot(Ts, label="T at point 1")
plt.axhline(T_goal, color="k", linestyle="--", label=f"T_goal = {T_goal}")
plt.xlabel("time step (after t_measure)")
plt.ylabel("temperature at point-1")
plt.legend()
plt.savefig("temperature_at_point1.pdf", bbox_inches='tight')
 
# --- Plot: velocity ---
plt.figure()
plt.plot(us, label="velocity in the x-direction at point 1")
plt.xlabel("time step (after t_measure)")
plt.ylabel("velocity in the x-direction at point-1")
plt.axhline(u_avg, color="k", linestyle="--", label=f"average velocity = {u_avg}")
plt.legend()
plt.savefig("velocity_magnitude_at_point1.pdf", bbox_inches='tight')
 
# --- Plot: Nusselt numbers ---
plt.figure()
plt.plot(Nu_hot,  label="Nusselt number")
plt.axhline(Nu_goal, color="k", linestyle="--", label=f"Nu goal = {Nu_goal}")
plt.xlabel("time step (after t_measure)")
plt.ylabel("Nusselt number")
plt.legend()
plt.savefig("nusselt_number.pdf", bbox_inches='tight')
