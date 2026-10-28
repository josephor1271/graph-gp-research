"""An ordinary ExactGP and an intentionally dense, small-graph reference."""

import gpytorch
import torch


def vertex_ids(inputs: torch.Tensor, size: int) -> torch.Tensor:
    """GPyTorch presents scalar inputs as an (m, 1) tensor."""
    if inputs.ndim != 2 or inputs.shape[1] != 1:
        raise ValueError("Use unbatched integer vertex IDs with shape (m, 1).")
    if inputs.dtype not in (torch.int32, torch.int64):
        raise ValueError("Vertex IDs must be integer tensors.")
    ids = inputs[:, 0].long()
    if ((ids < 0) | (ids >= size)).any():
        raise ValueError("Vertex ID outside the graph.")
    return ids


class DenseHeatKernel(gpytorch.kernels.Kernel):
    """K_tau = Q diag(exp(-tau lambda)) Q^T for a fixed scaled Laplacian.

    Heat time is restricted to (0.05, 4); amplitude is fixed at one.
    The eigendecomposition and covariance storage are deliberately dense.
    """

    def __init__(self, laplacian: torch.Tensor, heat_time: float = 1.0):
        super().__init__()
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian.to_dense())
        self.register_buffer("eigenvalues", eigenvalues)
        self.register_buffer("eigenvectors", eigenvectors)
        self.register_parameter("raw_heat_time", torch.nn.Parameter(eigenvalues.new_zeros(())))
        # GPyTorch constructs constraint buffers in the global default dtype;
        # match the parameter to avoid mixed-precision interval arithmetic.
        constraint = gpytorch.constraints.Interval(0.05, 4.0).to(dtype=eigenvalues.dtype)
        self.register_constraint("raw_heat_time", constraint)
        time = eigenvalues.new_tensor(heat_time)
        self.initialize(raw_heat_time=self.raw_heat_time_constraint.inverse_transform(time))

    @property
    def heat_time(self) -> torch.Tensor:
        return self.raw_heat_time_constraint.transform(self.raw_heat_time)

    def forward(self, x1, x2, diag=False, **params):
        left = self.eigenvectors[vertex_ids(x1, self.eigenvalues.numel())]
        right = self.eigenvectors[vertex_ids(x2, self.eigenvalues.numel())]
        weighted = left * torch.exp(-self.heat_time * self.eigenvalues)
        if diag:
            return (weighted * right).sum(dim=-1)
        return weighted @ right.T


class GraphGP(gpytorch.models.ExactGP):
    """The same familiar zero-mean model accepts either graph kernel."""

    def __init__(self, train_ids, train_y, likelihood, kernel):
        super().__init__(train_ids, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, ids):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(ids), self.covar_module(ids)
        )
