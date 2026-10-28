"""Expose the sparse heat recurrence to the ordinary GPyTorch ExactGP."""

import gpytorch
import torch
from linear_operator.operators import LinearOperator

from chebyshev import apply_chebyshev, heat_coefficients
from dense_gp import vertex_ids


class HeatOperator(LinearOperator):
    """Selected rows/columns of p_d(L), retaining only graph and coefficients.

    All tensor arguments enter the library representation tree so its standard
    matmul/solve autograd can differentiate the coefficient tensor. The graph is
    fixed. This example supports one unbatched graph and 2-D RHS matrices.
    """

    def __init__(
        self, laplacian: torch.Tensor, coefficients: torch.Tensor,
        rows: torch.Tensor, columns: torch.Tensor,
    ):
        super().__init__(laplacian, coefficients, rows, columns)
        self.laplacian = laplacian
        self.coefficients = coefficients
        self.rows = rows
        self.columns = columns

    def _size(self):
        return torch.Size([self.rows.numel(), self.columns.numel()])

    def _matmul(self, rhs: torch.Tensor) -> torch.Tensor:
        # S_columns^T scatters observations, including repeated vertex IDs.
        embedded = rhs.new_zeros(self.laplacian.shape[0], rhs.shape[-1])
        embedded = embedded.index_add(0, self.columns, rhs)
        product = apply_chebyshev(self.laplacian, self.coefficients, embedded)
        return product[self.rows]

    def _transpose_nonbatch(self):
        return HeatOperator(self.laplacian, self.coefficients, self.columns, self.rows)

    def _getitem(self, row_index, col_index, *batch_indices):
        # Preserve sparse products when ExactGP slices the joint prior into
        # training and test blocks, instead of using default interpolation.
        return HeatOperator(
            self.laplacian, self.coefficients, self.rows[row_index], self.columns[col_index]
        )

    def _get_indices(self, row_index, col_index, *batch_indices):
        # Exact requested entries cost one recurrence per entry. This simple
        # diagonal/indexing path never allocates an n-by-n graph identity.
        rows, columns = torch.broadcast_tensors(self.rows[row_index], self.columns[col_index])
        entries = []
        for row, column in zip(rows.reshape(-1), columns.reshape(-1)):
            basis = self.coefficients.new_zeros(self.laplacian.shape[0], 1)
            basis[column, 0] = 1
            product = apply_chebyshev(self.laplacian, self.coefficients, basis)
            entries.append(product[row, 0])
        return torch.stack(entries).reshape(rows.shape)

    def _diagonal(self):
        positions = torch.arange(min(self.shape), device=self.rows.device)
        return self._get_indices(positions, positions)


class SparseHeatKernel(gpytorch.kernels.Kernel):
    """Degree-20 heat covariance on [0, 2], with trainable tau in (0.05, 4)."""

    def __init__(self, laplacian: torch.Tensor, heat_time: float = 1.0):
        super().__init__()
        self.register_buffer("laplacian", laplacian)
        raw_time = torch.zeros((), dtype=laplacian.dtype, device=laplacian.device)
        self.register_parameter("raw_heat_time", torch.nn.Parameter(raw_time))
        # Match constraint buffers to the parameter, as in the dense reference.
        constraint = gpytorch.constraints.Interval(0.05, 4.0).to(dtype=laplacian.dtype)
        self.register_constraint("raw_heat_time", constraint)
        time = self.raw_heat_time.new_tensor(heat_time)
        self.initialize(raw_heat_time=self.raw_heat_time_constraint.inverse_transform(time))

    @property
    def heat_time(self) -> torch.Tensor:
        return self.raw_heat_time_constraint.transform(self.raw_heat_time)

    def forward(self, x1, x2, diag=False, **params):
        rows = vertex_ids(x1, self.laplacian.shape[0])
        columns = vertex_ids(x2, self.laplacian.shape[0])
        operator = HeatOperator(self.laplacian, heat_coefficients(self.heat_time), rows, columns)
        if diag:
            return operator.diagonal()
        return operator
