
import dolfinx.fem
import dolfinx_mpc
import dolfinx.fem.petsc
import dolfinx.la as _la
import dolfinx.nls.petsc
from mpi4py import MPI  
import dolfinx.la as _la
from scifem import BlockedNewtonSolver
from petsc4py import PETSc

##############################################################################
#              Classes for nonlinear MPC problems                            #
##############################################################################

"""
From https://github.com/jorgensd/dolfinx_mpc/blob/main/python/tests/test_nonlinear_assembly.py

"""



class NonlinearMPCProblem(dolfinx.fem.petsc.NonlinearProblem):
    def __init__(self, F, u, mpc, bcs=[], J=None, form_compiler_options={}, jit_options={}):
        self.mpc = mpc
        super().__init__(F, u, bcs=bcs, J=J, form_compiler_options=form_compiler_options, jit_options=jit_options)

    def F(self, x: PETSc.Vec, F: PETSc.Vec):  # type: ignore
        with F.localForm() as F_local:
            F_local.set(0.0)
        dolfinx_mpc.assemble_vector(self._L, self.mpc, b=F)

        # Apply boundary condition
        dolfinx_mpc.apply_lifting(
            F,
            [self._a],
            bcs=[self.bcs],
            constraint=self.mpc,
            x0=[x],
            scale=dolfinx.default_scalar_type(-1.0),
        )
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
        dolfinx.fem.petsc.set_bc(F, self.bcs, x, -1.0)

    def J(self, x: PETSc.Vec, A: PETSc.Mat):  # type: ignore
        A.zeroEntries()
        dolfinx_mpc.assemble_matrix(self._a, self.mpc, bcs=self.bcs, A=A)
        A.assemble()

class NewtonSolverMPC(dolfinx.cpp.nls.petsc.NewtonSolver):
    def __init__(
        self,
        comm: MPI.Intracomm,
        problem: NonlinearMPCProblem,
        mpc: dolfinx_mpc.MultiPointConstraint,
    ):
        """A Newton solver for non-linear MPC problems."""
        super().__init__(comm)
        self.mpc = mpc
        self.u_mpc = dolfinx.fem.Function(mpc.function_space)

        # Create matrix and vector to be used for assembly of the non-linear
        # MPC problem
        self._A = dolfinx_mpc.cpp.mpc.create_matrix(problem.a._cpp_object, mpc._cpp_object)
        self._b = _la.create_petsc_vector(mpc.function_space.dofmap.index_map, mpc.function_space.dofmap.index_map_bs)

        self.setF(problem.F, self._b)
        self.setJ(problem.J, self._A)
        self.set_form(problem.form)
        self.set_update(self.update)

    def update(self, solver: dolfinx.nls.petsc.NewtonSolver, dx: PETSc.Vec, x: PETSc.Vec):  # type: ignore
        # We need to use a vector created on the MPC's space to update ghosts
        self.u_mpc.x.petsc_vec.array = x.array_r
        self.u_mpc.x.petsc_vec.axpy(-1.0, dx)
        self.u_mpc.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,  # type: ignore
            mode=PETSc.ScatterMode.FORWARD,  # type: ignore
        )  # type: ignore
        self.mpc.homogenize(self.u_mpc)
        self.mpc.backsubstitution(self.u_mpc)
        x.array = self.u_mpc.x.petsc_vec.array_r
        x.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,  # type: ignore
            mode=PETSc.ScatterMode.FORWARD,  # type: ignore
        )  # type: ignore

    def solve(self, u: dolfinx.fem.Function):
        """Solve non-linear problem into function u. Returns the number
        of iterations and if the solver converged."""
        n, converged = super().solve(u.x.petsc_vec)
        u.x.scatter_forward()
        return n, converged

    @property
    def A(self) -> PETSc.Mat:  # type: ignore
        """Jacobian matrix"""
        return self._A

    @property
    def b(self) -> PETSc.Vec:  # type: ignore
        """Residual vector"""
        return self._b

