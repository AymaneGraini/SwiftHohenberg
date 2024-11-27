
from __future__ import annotations

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import dolfinx.fem.petsc
import dolfinx.la as _la
import dolfinx.nls.petsc
import numpy as np
import ufl
from dolfinx.mesh import locate_entities_boundary, meshtags
import dolfinx_mpc


def periodic_bcs(domain, funcspace,nspace):

    ndir=2

    fdim = domain.topology.dim - 1
    pbc_directions = []
    pbc_slave_tags = []
    pbc_is_slave = []
    pbc_is_master = []
    pbc_meshtags = []
    pbc_slave_to_master_maps = []
    
    # number of subspaces in functionspace
    N_subspaces = funcspace.num_sub_spaces
    
    # Create MultiPointConstraint object
    mpc = dolfinx_mpc.MultiPointConstraint(funcspace)

    def generate_pbc_slave_to_master_map(i):
        def pbc_slave_to_master_map(x):
            out_x = x.copy() 
            out_x[i] = x[i] - domain.geometry.x.max()
            return out_x
        return pbc_slave_to_master_map

    def generate_pbc_is_slave(i):
        return lambda x: np.isclose(x[i], domain.geometry.x.max())

    def generate_pbc_is_master(i):
        return lambda x: np.isclose(x[i], domain.geometry.x.min())

    def parse_bcs():
        for i in range(ndir):
            pbc_directions.append(i)
            pbc_slave_tags.append(i + 2)
            pbc_is_slave.append(generate_pbc_is_slave(i))
            pbc_is_master.append(generate_pbc_is_master(i))
            pbc_slave_to_master_maps.append(generate_pbc_slave_to_master_map(i))

            facets = locate_entities_boundary(domain, fdim, pbc_is_slave[-1])
            arg_sort = np.argsort(facets)
            pbc_meshtags.append(meshtags(domain,
                                            fdim,
                                            facets[arg_sort],
                                            np.full(len(facets), pbc_slave_tags[-1], dtype=np.int32)))
    
        N_pbc = len(pbc_directions)
        for i in range(N_pbc):
            if N_pbc > 1:
                def pbc_slave_to_master_map(x):
                    out_x = pbc_slave_to_master_maps[i](x)
                    idx = pbc_is_slave[(i + 1) % N_pbc](x)
                    out_x[pbc_directions[i]][idx] = np.nan
                    return out_x
            else:
                pbc_slave_to_master_map = pbc_slave_to_master_maps[i]
    
            mpc.create_periodic_constraint_topological(functionspace, pbc_meshtags[i],
                                                        pbc_slave_tags[i],
                                                        pbc_slave_to_master_map,
                                                        [])
            
        if len(pbc_directions) > 1:
            # Map intersection(slaves_x, slaves_y) to intersection(masters_x, masters_y),
            # i.e. map the slave dof at (1, 1) to the master dof at (0, 0)
            def pbc_slave_to_master_map(x):
                out_x = x.copy()
                out_x[0] = x[0] - domain.geometry.x.max()
                out_x[1] = x[1] - domain.geometry.x.max()
                idx = np.logical_and(pbc_is_slave[0](x), pbc_is_slave[1](x))
                out_x[0][~idx] = np.nan
                out_x[1][~idx] = np.nan
                return out_x
            mpc.create_periodic_constraint_topological(functionspace, pbc_meshtags[1],
                                                        pbc_slave_tags[1],
                                                        pbc_slave_to_master_map,
                                                        [])

    if N_subspaces == 0:
        functionspace = funcspace
        parse_bcs()
    else:
        for j in range(nspace):
            functionspace = funcspace.sub(j)
            parse_bcs()
            
    mpc.finalize()
    
    return mpc
