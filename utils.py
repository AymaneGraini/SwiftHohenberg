import numpy as np
import matplotlib.pyplot as plt
import dolfinx.geometry as gm
import pandas as pd
from dolfinx.fem import (Expression, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological)
import ufl
from mpi4py import MPI

def get2dplot(psi,x_vals, y_vals,domain,tree,filename,show):
    X_parametric, Y_parametric = np.meshgrid(x_vals, y_vals)
    # Flatten the grid for efficient iteration
    grid_points = np.vstack([X_parametric.ravel(), Y_parametric.ravel(), np.zeros_like(X_parametric.ravel())]).T

    tree = gm.bb_tree(domain, domain.topology.dim)
    # index=cell_index.array[0]
    Z= np.full_like(X_parametric, np.nan)

    # Evaluate the function at grid points
    for idx, point in enumerate(grid_points):
        # Find the candidate cells for the point
        cell_candidates = gm.compute_collisions_points(tree, point)
        colliding_cells = gm.compute_colliding_cells(domain, cell_candidates, point)
        
        if len(colliding_cells) > 0:
            # Use the first valid cell containing the point
            cell_index = colliding_cells.array[0]
            value = psi.eval(point, cell_index)
            # Map back to the grid
            i, j = divmod(idx, X_parametric.shape[1])
            Z[i, j] = value
        else:
            # Handle points outside the mesh
            print(f"Point {point[:2]} is outside the mesh.")
    x_flat = X_parametric.ravel()
    y_flat = Y_parametric.ravel()
    z_flat = Z.ravel()

    # Create a pandas DataFrame
    data = {
        "x": x_flat,
        "y": y_flat,
        "value": z_flat
    }
    df = pd.DataFrame(data)

    # Save to a CSV file
    csv_filename = filename+".csv"
    df.to_csv(csv_filename, index=False)
    if show:
        plt.imshow(Z,extent=[0, x_vals[-1], 0, y_vals[-1]], origin="lower", cmap="seismic")
        plt.colorbar()
        plt.show()


def error_L2(uh, u_ex, degree_raise=3):
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree
    family = uh.function_space.ufl_element().family_name
    mesh = uh.function_space.mesh
    W = functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = Expression(u_ex, W.element.interpolation_points())
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)