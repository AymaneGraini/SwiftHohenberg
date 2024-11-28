
from __future__ import annotations
import numpy as np
from dolfinx.mesh import locate_entities_boundary, meshtags
import dolfinx_mpc


def PeriodicBC(domain,funcspace,nsub,boundary_condition = ["periodic", "periodic"]):
    mpc = dolfinx_mpc.MultiPointConstraint(funcspace)
    a=domain.geometry.x[-1,:] # array of the upper right corner (L,H)
    L = a[0]
    H = a[1]

    fdim = domain.topology.dim - 1
    bcs = []
    pbc_directions = []
    pbc_slave_tags = []
    pbc_is_slave = []
    pbc_is_master = []
    pbc_meshtags = []
    pbc_slave_to_master_maps = []

    Nsubspaces = funcspace.num_sub_spaces #THe number of available subspaces

    def generate_pbc_slave_to_master_map(i):
        maxd = L if i==0 else H
        def pbc_slave_to_master_map(x):
            out_x = x.copy()
            out_x[i] = x[i] - maxd
            return out_x

        return pbc_slave_to_master_map

    def generate_pbc_is_slave(i):
        maxd = L if i==0 else H
        return lambda x: np.isclose(x[i], maxd)

    def generate_pbc_is_master(i):
        return lambda x: np.isclose(x[i], 0)

    for i, bc_type in enumerate(boundary_condition):
        pbc_directions.append(i)
        pbc_slave_tags.append(i + 2)
        pbc_is_slave.append(generate_pbc_is_slave(i))
        pbc_is_master.append(generate_pbc_is_master(i))
        pbc_slave_to_master_maps.append(generate_pbc_slave_to_master_map(i))

        facets = locate_entities_boundary(domain, fdim, pbc_is_slave[-1])
        arg_sort = np.argsort(facets)
        pbc_meshtags.append(
            meshtags(
                domain,
                fdim,
                facets[arg_sort],
                np.full(len(facets), pbc_slave_tags[-1], dtype=np.int32),
            )
        )
    def applybcs(space):
        # Create MultiPointConstraint object
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

            mpc.create_periodic_constraint_topological(space, pbc_meshtags[i], pbc_slave_tags[i], pbc_slave_to_master_map, bcs)

        if len(pbc_directions) > 1:
            # Map intersection(slaves_x, slaves_y) to intersection(masters_x, masters_y),
            # i.e. map the slave dof at (1, 1) to the master dof at (0, 0)
            def pbc_slave_to_master_map(x):
                out_x = x.copy()
                out_x[0] = x[0] - L
                out_x[1] = x[1] - H
                idx = np.logical_and(pbc_is_slave[0](x), pbc_is_slave[1](x))
                out_x[0][~idx] = np.nan
                out_x[1][~idx] = np.nan
                return out_x

            mpc.create_periodic_constraint_topological(space, pbc_meshtags[1], pbc_slave_tags[1], pbc_slave_to_master_map, bcs)

    if Nsubspaces==0:
        applybcs(funcspace)
    else:
        for n in range(nsub):
            applybcs(funcspace.sub(n))
        mpc.finalize()

    return mpc