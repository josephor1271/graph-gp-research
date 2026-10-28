"""Run with .venv/bin/python demo.py; CPU float64, one fixed synthetic draw."""

from pathlib import Path

import gpytorch
import matplotlib
import numpy as np
import scipy.sparse as sparse
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dense_gp import DenseHeatKernel, GraphGP
from graph_gp import SparseHeatKernel


def grid_laplacian(side: int = 6) -> torch.Tensor:
    """Unweighted grid: L = (D - A) / max_degree, with spectrum in [0, 2]."""
    edges = np.ones(side - 1)
    adjacency = sparse.diags([edges, edges], [-1, 1])
    path = sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    grid = sparse.kronsum(path, path, format="coo")
    grid = grid / grid.diagonal().max()
    indices = torch.tensor(np.stack([grid.row, grid.col]), dtype=torch.long)
    # The scoped check also covers the sparse tensor created by coalesce().
    with torch.sparse.check_sparse_tensor_invariants():
        return torch.sparse_coo_tensor(
            indices, torch.tensor(grid.data), grid.shape, check_invariants=True
        ).coalesce()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    laplacian = grid_laplacian()
    ids = torch.arange(laplacian.shape[0])
    noise_variance = 0.04

    # This exact dense draw is data generation, independent of either GP kernel.
    truth_covariance = torch.matrix_exp(-1.4 * laplacian.to_dense())
    truth = torch.linalg.cholesky(truth_covariance) @ torch.randn(ids.numel(), dtype=torch.float64)
    observations = truth + noise_variance**0.5 * torch.randn_like(truth)
    train_ids = ids[ids % 3 != 1]
    test_ids = ids[ids % 3 == 1]
    train_y = observations[train_ids]

    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    likelihood.noise = torch.tensor(noise_variance, dtype=torch.float64)
    likelihood.raw_noise.requires_grad_(False)
    kernel = DenseHeatKernel(laplacian, heat_time=0.8)
    model = GraphGP(train_ids, train_y, likelihood, kernel)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    losses = []
    model.train()
    likelihood.train()
    for _ in range(40):
        optimizer.zero_grad()
        loss = -mll(model(train_ids), train_y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    final_loss = -mll(model(train_ids), train_y)
    print(f"Grid: 6 x 6; {train_ids.numel()} train / {test_ids.numel()} held out; CPU float64")
    print(f"Noise variance fixed at {noise_variance}; amplitude fixed at 1")
    print(f"Dense negative MLL: {losses[0]:.8f} -> {final_loss.item():.8f} (40 Adam steps)")
    print(f"Fitted heat time: {kernel.heat_time.item():.8f} (allowed interval 0.05 to 4)")

    # Swap only the kernel, then compare at identical fitted parameters.
    sparse_kernel = SparseHeatKernel(laplacian, heat_time=kernel.heat_time.item())
    sparse_model = GraphGP(train_ids, train_y, likelihood, sparse_kernel)
    sparse_mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, sparse_model)
    rhs = torch.linspace(-1, 1, ids.numel(), dtype=torch.float64)
    action_error = (kernel(ids) @ rhs - sparse_kernel(ids) @ rhs).abs().max()
    dense_value = -final_loss
    dense_gradient, = torch.autograd.grad(dense_value, kernel.raw_heat_time)
    sparse_value = sparse_mll(sparse_model(train_ids), train_y)
    sparse_gradient, = torch.autograd.grad(sparse_value, sparse_kernel.raw_heat_time)
    print(f"Degree 20 covariance action max error: {action_error.item():.3e}")
    print(f"Cholesky MLL / raw-tau gradient differences: "
          f"{abs(dense_value.item() - sparse_value.item()):.3e} / "
          f"{abs(dense_gradient.item() - sparse_gradient.item()):.3e}")

    # Dense iterative versus sparse iterative isolates the representation.
    # Each is also compared with the direct dense value: SLQ is stochastic.
    iterative_results = []
    for current_model, current_mll in [(model, mll), (sparse_model, sparse_mll)]:
        torch.manual_seed(71)
        with (
            gpytorch.settings.max_cholesky_size(0),
            gpytorch.settings.max_preconditioner_size(0),
            gpytorch.settings.cg_tolerance(1e-10),
            gpytorch.settings.num_trace_samples(32),
            gpytorch.settings.max_lanczos_quadrature_iterations(24),
        ):
            value = current_mll(current_model(train_ids), train_y)
            gradient, = torch.autograd.grad(value, current_model.covar_module.raw_heat_time)
            iterative_results.append(torch.stack([value.detach(), gradient]))
    matched_error = (iterative_results[0] - iterative_results[1]).abs()
    exact_result = torch.stack([dense_value.detach(), dense_gradient])
    inference_error = (iterative_results[1] - exact_result).abs()
    print("CG/SLQ: requested tolerance 1e-10, 32 probes, seed 71, 24 Lanczos steps; no preconditioner")
    print(f"Matched dense/sparse iterative MLL / gradient differences: "
          f"{matched_error[0]:.3e} / {matched_error[1]:.3e}")
    print(f"Sparse iterative versus dense direct MLL / gradient errors: "
          f"{inference_error[0]:.3e} / {inference_error[1]:.3e}")

    model.eval()
    sparse_model.eval()
    likelihood.eval()
    with torch.no_grad():
        prediction = model(test_ids)
        mean = prediction.mean
        std = prediction.variance.sqrt()
    with (
        torch.no_grad(),
        gpytorch.settings.max_cholesky_size(0),
        gpytorch.settings.max_preconditioner_size(0),
        gpytorch.settings.max_eager_kernel_size(0),
        gpytorch.settings.eval_cg_tolerance(1e-10),
    ):
        sparse_prediction = sparse_model(test_ids)
        sparse_mean = sparse_prediction.mean
        sparse_variance = sparse_prediction.variance
    print(f"Held-out mean / variance max differences (sparse CG versus dense direct): "
          f"{(sparse_mean - mean).abs().max():.3e} / {(sparse_variance - std.square()).abs().max():.3e}")
    rmse = (mean - truth[test_ids]).square().mean().sqrt()
    print(f"Held-out latent RMSE: {rmse.item():.8f}")

    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(ids, truth, color="0.6", label="Latent draw")
    axis.scatter(train_ids, train_y, s=18, color="black", label="Training observations")
    axis.errorbar(test_ids, mean, yerr=2 * std, fmt="o", label="Dense latent mean ± 2 sd")
    axis.scatter(test_ids, sparse_mean, marker="x", color="red", label="Sparse CG latent mean")
    axis.set(xlabel="Grid vertex ID (row order)", ylabel="Function value")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "prediction.png", dpi=150)
    plt.close(figure)
    np.savez(output / "prediction_inputs.npz", truth=truth.numpy(), observations=observations.numpy(),
             train_ids=train_ids.numpy(), test_ids=test_ids.numpy(), dense_mean=mean.numpy(),
             dense_variance=std.square().numpy(), sparse_mean=sparse_mean.numpy(),
             sparse_variance=sparse_variance.numpy(), heat_time=kernel.heat_time.item(),
             noise_variance=noise_variance)
    print(f"Figure and reproducible inputs: {output}")


if __name__ == "__main__":
    main()
