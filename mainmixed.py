# Load required libraries
#########################
import dolfinx
from dolfinx      import fem, mesh, plot, cpp
from dolfinx.fem  import FunctionSpace, Function, locate_dofs_geometrical, dirichletbc, Constant
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.io   import XDMFFile
import dolfinx_mpc
import matplotlib.pyplot as plt
from ufl import dx, grad, inner, div

import numpy as np
from mpi4py import MPI
import ufl
from typing import List
import basix
from utils import *
import ufl.integral
# import pytest
from pbcs import *
from petsc4py import PETSc
import time
import pyvista
import dolfinx.la as _la
import NLS
import BlockNLS
import BlockMPC
import NESTMPC
import scifem
##############################################################################
#                                     Define mesh                            #
##############################################################################
theta=0.5
N=1
GAMMA=1
q0=1
dt=0.1
eps=+0.2
g=1
avg = +0.3
# N=5

xell_ = np.pi/np.sqrt(3)

yell_ = np.pi


# L = 1; H = 1
# print(L,H)
# cell_sizex = xell_/5
# cell_sizey = yell_/5
# nx = 2
# ny = 2
# comm = MPI.COMM_WORLD
# domain = mesh.create_rectangle(comm, [(0.0, 0.0), (L, H)], [nx, ny],mesh.CellType.quadrilateral)
# ndim = domain.geometry.dim
L = 20*xell_; H = 20*yell_
cell_sizex = xell_/5
cell_sizey = yell_/5

print(L,H)
nx = int(L/cell_sizex)
ny = int(H/cell_sizey)
print(L,H)
comm = MPI.COMM_WORLD
domain = mesh.create_rectangle(comm, [(0.0, 0.0), (L, H)], [nx, ny],mesh.CellType.quadrilateral)
ndim = domain.geometry.dim


####### Define functional space #######
V = dolfinx.fem.functionspace(domain, ("CG", 1))
W = ufl.MixedFunctionSpace(V, V)
dpsi,dmu = ufl.TrialFunctions(W)
q, v = ufl.TestFunctions(W)

psi = fem.Function(V)
mu = fem.Function(V)
psi0 = fem.Function(V)
mu0 = fem.Function(V)
P = basix.ufl.element("Lagrange", domain.basix_cell(), 1) ## Interpolation
print(P)
ME = fem.functionspace(domain, basix.ufl.mixed_element([P, P])) # A mixed space FE
# SP = fem.functionspace(domain, P) # A mixed space FE
# q, v = ufl.TestFunctions(ME)
# Chi = fem.Function(ME)
# Chi0 = fem.Function(ME)
# psi, mu = ufl.split(Chi)
# psi0, mu0 = ufl.split(Chi0)
# target_psi=  fem.Function(SP)




# Define problem
A =  lambda g,avg,eps : -0.8422737685176483
avg = 0.1022341857169855

hex = lambda x : A(g,avg,eps)*(np.cos((np.sqrt(3)/2)*x[0])*np.cos(x[1]/2)-0.5*np.cos(x[1]))+avg
initialCmu = lambda x : -A(g,avg,eps)*(np.cos(x[1]/2)*np.cos(np.sqrt(3)*x[0]/2) - 2*np.cos(x[1]))/4 - 3*A(g,avg,eps)*np.cos(x[1]/2)*np.cos(np.sqrt(3)*x[0]/2)/4


# A*(cos(sqrt(3)*x/2)*cos(y/2)-0.5*cos(y))+p
filename = "mpcNLblocktesting"

# Interpolate initial condition
rng = np.random.default_rng(42)
initialCpsi = lambda x :hex(x)
initialCpsi = lambda x : (rng.random(x.shape[1])-0.5)+avg*0.8+hex(x)*0.2
psi.interpolate(initialCpsi)
mu.x.array[:]=0

# psi.x.array[:]=1
# mu.x.array[:]=3
# mu.interpolate(initialCmu)

psi0.x.array[:] = psi.x.array[:]
mu0.x.array[:] = mu.x.array[:]

# Chi.sub(1).interpolate(incleaitialCmu)
psi_mid = (1.0 - theta) * psi0 + theta * psi
mu_mid = (1.0 - theta) * mu0 + theta * mu
# psi_avg = fem.assemble_scalar(fem.form(psi*dx))/(L*H)

# print("Psi_average = ", psi_avg)

Fpsi = inner(psi,q)*dx -inner(psi0,q)*dx+dt*GAMMA*(((N*q0)**4-eps)*inner(psi_mid,q)*dx
                                                #    -g*inner(psi_mid**2,q)*dx
                                                   -(g/2)*inner(psi**2+psi0**2,q)*dx
                                                   +(1/2)*inner(psi**3+psi0**3,q)*dx
                                                #    +inner(psi_mid**3,q)*dx
                                                   -inner(grad(mu_mid),grad(q))*dx
                                                   +2*(N*q0)**2*inner(mu_mid,q)*dx)
Fmu = inner(mu,v)*dx+inner(grad(psi),grad(v))*dx ##Correct

# Fpsi = inner(psi**2,q)*dx-inner(3,q)*dx
# Fmu = inner(mu**2,v)*dx-inner(2,v)*dx
F_ = Fpsi + Fmu
F = list(ufl.extract_blocks(F_))
J = ufl.extract_blocks(ufl.derivative(F_, psi, dpsi) + ufl.derivative(F_, mu, dmu))

## Boundary conditions
mpc = PeriodicBC(domain, V,1)
# mpc = doclear
# lfinx_mpc.MultiPointConstraint(V)
# problem = NLS.NonlinearMPCProblem(F, [psi,mu], mpc, bcs=[])                # define nonlinear problem
# solver  = NLS.NewtonSolverMPC(comm, problem, mpc)                                   # define solver
t=0


# solver = BlockMPC.BlockedNewtonSolverMPC(F,[psi, mu],[mpc,mpc])
# solver = BlockNLS.BlockedNewtonSolver(F,[psi, mu])
solver = NESTMPC.NestedNewtonSolver(F,[psi, mu],[mpc,mpc])


# Create a grid
x_vals = np.linspace(0, L, nx)
y_vals = np.linspace(0, H, ny)
tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
# get2dplot(psi,x_vals, y_vals,domain,tree,filename+"InitialC",True)
# get2dplot(mu,x_vals, y_vals,domain,tree,ficlearlename+"InitialC",True)




# problem = NLS.NonlinearMPCProblem(F, Chi, mpc, bcs=[])                # define nonlinear problem
# solver  = NLS.NewtonSolverMPC(comm, problem, mpc)                                   # define solver



# # Set Solver Configurations
solver.convergence_criterion = "incremental"
solver.rtol = 1e-5
ksp = solver.krylov_solver
ksp.setOperators(solver._J,None)
opts = PETSc.Options()


if solver._J.getType()=="nest":
    opts.clear()
    ksp.setOptionsPrefix("ksp_")
    option_prefix = ksp.getOptionsPrefix()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("fieldsplit")
    nested_IS = solver._J.getNestISs()
    pc.setFieldSplitIS(("1", nested_IS[0][0]))
    pc.setFieldSplitIS(("2", nested_IS[0][1]))
    opts[f"{option_prefix}pc_fieldsplit_type"] = "schur"
    opts[f"{option_prefix}pc_fieldsplit_schur_fact_type"] = "full"
    opts[f"{option_prefix}pc_fieldsplit_schur_precondition"] = "selfp"
    opts[f"{option_prefix}fieldsplit_1_ksp_type"] = "gmres"
    opts[f"{option_prefix}fieldsplit_1_pc_type"] = "ilu"
    opts[f"{option_prefix}fieldsplit_1_ksp_rtol"] = 1e-10
    opts[f"{option_prefix}fieldsplit_2_ksp_type"] = "gmres"
    opts[f"{option_prefix}fieldsplit_2_pc_type"] = "ilu"
    opts[f"{option_prefix}fieldsplit_2_ksp_rtol"] = 1e-10
else:
    opts.clear()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
ksp.setFromOptions()

solver.solve()


get2dplot(psi,x_vals, y_vals,domain,tree,filename+"1iter",True)
get2dplot(mu,x_vals, y_vals,domain,tree,filename+"1iter",True)
exit()

# print(scifem.evaluate_function(psi, points))
# print(evaluate_function(mu, points))


file = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "./"+filename+".xdmf", "w")
file.write_mesh(domain)


# V0, dofs = W.sub(0).collapse()

topology, cell_types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
file.write_function(psi, 0)
# file.write_function(mu, 0)
file.close()
exit()



psi.x.array[:] = psi.x.array
print("SOlving...") ; t1=time.time()
t = 0.0
Average_ts=[]
Energy_ts=[]
Error_ts=[]
ts=[]
ts.append(t)
# Average_ts.append(psi_avg)
E = fem.assemble_scalar(fem.form(-((1/2)*(-eps+(q0)**4)*psi**2+(1/4)*psi**4-(g/3)*psi**3+(1/2)*(mu)**2-q0**2*inner(grad(psi),grad(psi)))*dx))


Energy_ts.append(E)
Error_ts.append(error_L2(psi, hex))

file.write_function(psi, t)

# print(type(psi))


while t < 20:
    t += dt
    # print("attemptin iteration...")
    r = solver.solve()
    print(r)
    # print(f"Step {int(t / dt)}: num iterations: {r[0]}","time=",t)
    psi0.x.array[:] = psi.x.array
    psi_avg = fem.assemble_scalar(fem.form(psi*dx))/(L*H)
    ts.append(t)
    Average_ts.append(psi_avg)
    file.write_function(psi, t)
    ### Compute total energy
    E = fem.assemble_scalar(fem.form(
        ((1/2)*(-eps+(N*q0)**4)*psi**2+(1/4)*psi**4-(g/3)*psi**3+(1/2)*(mu)**2-q0**2*inner(grad(psi),grad(psi)))*dx))
    Energy_ts.append(E)
    Error_ts.append(error_L2(psi, hex))
t2=time.time()
file.close()

exit()
print("Job done in ", t2-1)
grid.point_data["psi"] = Chi.x.array[dofs].real
grid.set_active_scalars("psi")
screenshot = None
screenshot = "c.png"
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False,cmap='seismic')
plotter.view_xy()
plotter.show()

file.close()
np.savetxt(filename+"ScalarOut.csv",np.column_stack((ts,Average_ts,Energy_ts,Error_ts)),delimiter="\t")


print("going to matploitlib")
get2dplot(psi,x_vals, y_vals,domain,tree,filename+"FinalC",False)

