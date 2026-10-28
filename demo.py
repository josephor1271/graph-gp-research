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


def grid_laplacian(side: int = 6) -> torch.Tensor:
    """Unweighted grid: L = (D - A) / max_degree, with spectrum in [0, 2]."""
    edges = np.ones(side - 1)
    adjacency = sparse.diags([edges, edges], [-1, 1])
    path = sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    grid = sparse.kronsum(path, path, format="coo")
    grid = grid / grid.diagonal().max()
    indices = torch.tensor(np.stack([grid.row, grid.col]), dtype=torch.long)
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
    likelihood.noise = noise_variance
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

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        prediction = model(test_ids)
        mean = prediction.mean
        std = prediction.variance.sqrt()
    rmse = (mean - truth[test_ids]).square().mean().sqrt()
    print(f"Held-out latent RMSE: {rmse.item():.8f}")

    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(ids, truth, color="0.6", label="Latent draw")
    axis.scatter(train_ids, train_y, s=18, color="black", label="Training observations")
    axis.errorbar(test_ids, mean, yerr=2 * std, fmt="o", label="Dense latent mean ± 2 sd")
    axis.set(xlabel="Grid vertex ID (row order)", ylabel="Function value")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "prediction.png", dpi=150)
    plt.close(figure)
    np.savez(output / "prediction_inputs.npz", truth=truth.numpy(), observations=observations.numpy(),
             train_ids=train_ids.numpy(), test_ids=test_ids.numpy(), dense_mean=mean.numpy(),
             dense_variance=std.square().numpy())
    print(f"Figure and reproducible inputs: {output}")


if __name__ == "__main__":
    main()
