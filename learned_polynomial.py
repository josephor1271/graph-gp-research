"""A normalized two-parameter spectral filter, squared through HeatOperator."""

import gpytorch
import torch

from dense_gp import vertex_ids
from graph_gp import HeatOperator


class LearnedPolynomialKernel(gpytorch.kernels.Kernel):
    """K=q(L)^2, q=1+a1*(T1+1)+a2*(T2-1), for fixed symmetric L in [0,2].

    The unconstrained shape parameters preserve q(0)=1. Wrap in ScaleKernel
    for signal variance; GaussianLikelihood supplies observation noise.
    """

    def __init__(self, laplacian: torch.Tensor):
        super().__init__()
        if laplacian.layout != torch.sparse_coo or laplacian.shape[0] != laplacian.shape[1]:
            raise ValueError("Use a square sparse COO symmetric normalized Laplacian.")
        self.register_buffer("laplacian", laplacian.coalesce())
        beta = 2 * torch.sigmoid(torch.tensor(1.0, dtype=laplacian.dtype, device=laplacian.device))
        self.register_parameter("a1", torch.nn.Parameter(-beta / 2 + beta.square() / 4))
        self.register_parameter("a2", torch.nn.Parameter(beta.square() / 16))

    @property
    def coefficients(self) -> torch.Tensor:
        """Five coefficients of q^2, using Ti*Tj=(T(i+j)+T|i-j|)/2.

        Selection occurs after squaring the full spectral filter. Squaring an
        observed submatrix instead would discard paths through other vertices.
        """
        c0 = 1 + self.a1 - self.a2
        c1, c2 = self.a1, self.a2
        return torch.stack((
            c0.square() + (c1.square() + c2.square()) / 2,
            2 * c0 * c1 + c1 * c2,
            2 * c0 * c2 + c1.square() / 2,
            c1 * c2,
            c2.square() / 2,
        ))

    def forward(self, x1, x2, diag=False, **params):
        rows = vertex_ids(x1, self.laplacian.shape[0])
        columns = vertex_ids(x2, self.laplacian.shape[0])
        operator = HeatOperator(self.laplacian, self.coefficients, rows, columns)
        if diag:
            return operator.diagonal()
        return operator
