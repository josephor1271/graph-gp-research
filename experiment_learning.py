"""Frozen learning-fidelity experiment: exact audits of dense and CG/SLQ fits."""

from pathlib import Path
from time import perf_counter

import gpytorch
import numpy as np
import torch

from demo import grid_laplacian, plt
from dense_gp import DenseHeatKernel, GraphGP
from graph_gp import SparseHeatKernel


def main():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    laplacian = grid_laplacian()
    dense_laplacian = laplacian.to_dense()
    ids = torch.arange(36)
    train, test = ids[ids % 3 != 1], ids[ids % 3 == 1]
    noise, steps = 0.04, 60
    truth = torch.linalg.cholesky(torch.matrix_exp(-1.4 * dense_laplacian)) @ torch.randn(36, dtype=torch.float64)
    observations = truth + noise**0.5 * torch.randn_like(truth)
    y = observations[train]
    identity = torch.eye(train.numel(), dtype=y.dtype)

    def exact_loss(tau: torch.Tensor) -> torch.Tensor:
        """Independent matrix-exponential/Cholesky negative MLL per observation."""
        covariance = torch.matrix_exp(-tau * dense_laplacian)[train][:, train] + noise * identity
        factor = torch.linalg.cholesky(covariance)
        quadratic = y @ torch.cholesky_solve(y[:, None], factor)[:, 0]
        return (quadratic + 2 * factor.diagonal().log().sum() + y.numel() * np.log(2 * np.pi)) / (2 * y.numel())

    profile_tau = torch.linspace(0.0501, 3.9999, 51, dtype=y.dtype)
    profile_loss = torch.stack([exact_loss(tau) for tau in profile_tau])
    best_grid = profile_loss.argmin().item()
    print("6x6 grid; demo seed 12, true tau=1.4, noise variance=0.04, 24 train / 12 test; CPU float64, one thread")
    print("Learn only tau: starts 0.8/3.0, 60 Adam updates, lr=0.08; sparse degree 20, CG 1e-4, 64 probes, 24 Lanczos, no preconditioner")
    print(f"Best of 51 exact profile points: tau={profile_tau[best_grid]:.6f}, negative MLL={profile_loss[best_grid]:.9f}")
    trajectories, records = [], []
    for start in [0.8, 3.0]:
        for seed in [0, 71, 72, 73]:  # Zero labels deterministic dense inference, not a probe stream.
            kernel = DenseHeatKernel(laplacian, start) if seed == 0 else SparseHeatKernel(laplacian, start)
            likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
            likelihood.noise = y.new_tensor(noise)
            likelihood.raw_noise.requires_grad_(False)
            model = GraphGP(train, y, likelihood, kernel)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
            if seed in [0, 71]:
                tau = kernel.heat_time.detach().requires_grad_()
                reference_value = exact_loss(tau)
                physical_gradient, = torch.autograd.grad(reference_value, tau)
                chain, = torch.autograd.grad(kernel.heat_time, kernel.raw_heat_time)
                reference_gradient = physical_gradient * chain
                with gpytorch.settings.max_cholesky_size(1000):
                    value = -mll(model(train), y)
                    gradient, = torch.autograd.grad(value, kernel.raw_heat_time)
                torch.testing.assert_close(value.detach(), reference_value.detach(), atol=2e-12, rtol=2e-11)
                torch.testing.assert_close(gradient, reference_gradient, atol=2e-12, rtol=2e-10)
                print(f"Check start={start}, {'dense' if seed == 0 else 'Chebyshev'} Cholesky: "
                      f"NLL={value.item():.9f}, raw-tau gradient={gradient.item():+.9f}, "
                      f"reference errors={(value - reference_value).item():+.3e}/{(gradient - reference_gradient).item():+.3e}")
            torch.manual_seed(seed)  # Once per run: fresh independent probes at subsequent steps.
            trajectory = []
            training_seconds = 0.0
            with (
                gpytorch.settings.max_cholesky_size(1000 if seed == 0 else 0),
                gpytorch.settings.cg_tolerance(1e-4),
                gpytorch.settings.num_trace_samples(64),
                gpytorch.settings.max_lanczos_quadrature_iterations(24),
                gpytorch.settings.max_preconditioner_size(0),
                gpytorch.settings.fast_computations(log_prob=True),
            ):
                for step in range(steps + 1):
                    tau = kernel.heat_time.detach()
                    with torch.no_grad():
                        audited_loss = exact_loss(tau).item()
                    clock_start = perf_counter()
                    optimizer.zero_grad()
                    loss = -mll(model(train), y)
                    if step < steps:
                        loss.backward()
                        optimizer.step()
                    training_seconds += perf_counter() - clock_start
                    trajectory.append([tau.item(), audited_loss, loss.item()])
                    if step == 0 and seed == 71:
                        print(f"Check start={start}, Chebyshev iterative seed=71: stochastic NLL={loss.item():.9f}, "
                              f"raw-tau gradient={kernel.raw_heat_time.grad.item():+.9f}; "
                              f"reference errors={(loss - reference_value).item():+.3e}/{(kernel.raw_heat_time.grad - reference_gradient).item():+.3e}")
            # All predictions use exact heat at the learned tau, isolating learning.
            tau = kernel.heat_time.detach().requires_grad_()
            final_loss = exact_loss(tau)
            final_gradient, = torch.autograd.grad(final_loss, tau)
            with torch.no_grad():
                covariance = torch.matrix_exp(-tau * dense_laplacian)
                factor = torch.linalg.cholesky(covariance[train][:, train] + noise * identity)
                cross = covariance[test][:, train]
                mean = cross @ torch.cholesky_solve(y[:, None], factor)[:, 0]
                observation_variance = covariance.diagonal()[test] - (cross * torch.cholesky_solve(cross.T, factor).T).sum(dim=1) + noise
                rmse = (mean - truth[test]).square().mean().sqrt()
                predictive_nll = (0.5 * ((observations[test] - mean).square() / observation_variance + observation_variance.log() + np.log(2 * np.pi))).mean()
            trajectories.append(trajectory)
            records.append([start, seed, tau.item(), final_loss.item(), rmse.item(), predictive_nll.item(), training_seconds, final_gradient.item()])

    records = np.array(records)
    trajectories = np.array(trajectories)
    best_reference = min(profile_loss.min().item(), records[records[:, 1] == 0, 3].min())
    print(f"Gap reference: best profile/dense endpoint NLL={best_reference:.9f}; not a certified optimum")
    print("arm     start seed final_tau exact_NLL gap latent_RMSE observation_marginal_NLL train_seconds exact_dNLL/dtau")
    for start, seed, tau, value, rmse, predictive_nll, elapsed, gradient in records:
        print(f"{'dense' if seed == 0 else 'CG/SLQ':7s} {start:.1f} {int(seed):2d} {tau:.7f} {value:.9f} {value - best_reference:+.3e} "
              f"{rmse:.7f} {predictive_nll:.7f} {elapsed:.4f} {gradient:+.3e}")
    print("Times exclude construction, Cholesky diagnostics, exact audits, and prediction; include 60 updates + final training-loss evaluation, no warm-up.")
    print("Trajectory fields: tau, exact negative MLL, training estimate (stochastic for CG/SLQ); steps 0 through 60.")
    print(f"Maximum final training-estimate/exact-NLL difference: {np.max(np.abs(trajectories[:, -1, 2] - records[:, 3])):.3e}")

    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(profile_tau, profile_loss, color="0.4", label="Exact profile")
    axes[0].plot(profile_tau[best_grid], profile_loss[best_grid], "k*", label="Best profile point")
    colors = {0: "black", 71: "tab:blue", 72: "tab:orange", 73: "tab:green"}
    for row, trajectory in zip(records, trajectories):
        start, seed, tau, value = row[:4]
        label = "Dense" if seed == 0 else f"CG seed {int(seed)}"
        axes[0].plot(tau, value, "o" if start == 0.8 else "^", color=colors[seed], markersize=5)
        axes[1].plot(np.arange(steps + 1), trajectory[:, 1], color=colors[seed],
                     linestyle="-" if start == 0.8 else "--", label=label if start == 0.8 else None)
    axes[0].set(xlabel="Heat time tau", ylabel="Exact negative MLL / observation")
    axes[1].set(xlabel="Adam updates", ylabel="Exact negative MLL / observation", title="Solid: start 0.8; dashed: start 3.0")
    for axis in axes:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "learning_fidelity.png", dpi=150)
    plt.close(figure)
    np.savez(output / "learning_inputs.npz", profile_tau=profile_tau.numpy(), profile_loss=profile_loss.numpy(),
             trajectories=trajectories, records=records, best_reference=best_reference,
             record_fields=["start", "seed_0_is_dense", "tau", "exact_nll", "latent_rmse", "observation_marginal_nll", "seconds", "exact_gradient"],
             observations=observations.numpy(), truth=truth.numpy(), train_ids=train.numpy(), test_ids=test.numpy())


if __name__ == "__main__":
    main()
