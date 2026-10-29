"""Pathwise heat-GP posterior draws, validated with a small dense reference."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from linear_operator.utils.linear_cg import linear_cg

from chebyshev import apply_chebyshev, heat_coefficients
from demo import grid_laplacian, plt


def main():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    laplacian = grid_laplacian(8)
    ids = torch.arange(64)
    train = ids[::3]
    test = ids[ids % 3 != 0]
    tau = torch.tensor(2.85602691, dtype=torch.float64)
    noise = 0.04
    count = 1024
    # Exact heat covariance is used only for data generation and tiny references.
    dense_laplacian = laplacian.to_dense()
    exact_heat = torch.matrix_exp(-tau * dense_laplacian)
    truth = torch.linalg.cholesky(exact_heat) @ torch.randn(64, dtype=tau.dtype)
    y = (truth + noise**0.5 * torch.randn_like(truth))[train]

    start = perf_counter()
    coefficients = heat_coefficients(tau / 2)  # Degree 20 square root q(L).
    basis = torch.zeros(64, train.numel(), dtype=tau.dtype)
    basis[train, torch.arange(train.numel())] = 1
    # Only graph/training columns of K_q = q(L)^2; never a full graph basis.
    columns = apply_chebyshev(laplacian, coefficients, basis)
    columns = apply_chebyshev(laplacian, coefficients, columns)
    observed = columns[train] + noise * torch.eye(train.numel(), dtype=tau.dtype)
    cross = columns[test]
    factor = torch.linalg.cholesky(observed)  # One factorization for all RHS.
    setup_seconds = perf_counter() - start

    start = perf_counter()
    generator = torch.Generator().manual_seed(42)
    z = torch.randn(64, count, dtype=tau.dtype, generator=generator)
    prior = apply_chebyshev(laplacian, coefficients, z)
    observation_noise = noise**0.5 * torch.randn(train.numel(), count, dtype=tau.dtype, generator=generator)
    rhs = y[:, None] - prior[train] - observation_noise
    weights = torch.cholesky_solve(rhs, factor)
    samples = prior[test] + cross @ weights
    sampling_seconds = perf_counter() - start

    # Independent reference: NumPy polynomial evaluation in a dense eigenbasis.
    # This full matrix belongs only to validation, not the proposed sample path.
    eigenvalues, eigenvectors = torch.linalg.eigh(dense_laplacian)
    q_values = torch.from_numpy(np.polynomial.chebyshev.chebval(eigenvalues.numpy() - 1, coefficients.numpy()))
    q_reference = (eigenvectors * q_values) @ eigenvectors.T
    covariance_reference = q_reference @ q_reference
    observed_reference = covariance_reference[train][:, train] + noise * torch.eye(train.numel(), dtype=tau.dtype)
    cross_reference = covariance_reference[test][:, train]
    gain = torch.linalg.solve(observed_reference, cross_reference.T).T
    mean = gain @ y
    posterior = covariance_reference[test][:, test] - gain @ cross_reference.T
    variance = posterior.diagonal()
    torch.testing.assert_close(columns, covariance_reference[:, train], atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(prior, q_reference @ z, atol=2e-12, rtol=2e-11)
    # Check the covariance of the actual linear sampling transformation, without
    # asking a finite Monte Carlo estimate to equal the answer exactly.
    actual_gain = torch.cholesky_solve(cross.T, factor).T
    sample_map = q_reference[test] - actual_gain @ q_reference[train]
    path_covariance = sample_map @ sample_map.T + noise * actual_gain @ actual_gain.T
    torch.testing.assert_close(actual_gain @ y, mean, atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(path_covariance, posterior, atol=2e-12, rtol=2e-11)
    exact_observed = exact_heat[train][:, train] + noise * torch.eye(train.numel(), dtype=tau.dtype)
    exact_gain = torch.linalg.solve(exact_observed, exact_heat[train][:, test]).T
    exact_mean = exact_gain @ y
    exact_variance = (exact_heat[test][:, test] - exact_gain @ exact_heat[train][:, test]).diagonal()
    print(f"8x8 grid; {train.numel()} train / {test.numel()} test; tau={tau}, noise variance={noise}, degree=20; CPU float64")
    print(f"K_q versus exact heat max error: {(covariance_reference - exact_heat).abs().max():.3e}; "
          f"posterior mean/variance max error: {(mean - exact_mean).abs().max():.3e}/{(variance - exact_variance).abs().max():.3e}")
    print(f"Block construction + factorization: {1e3 * setup_seconds:.3f} ms; "
          f"{count} draws + prior action + conditioning: {1e3 * sampling_seconds:.3f} ms (one run, no scaling claim)")

    means, variances = [], []
    for size in [512, count]:
        sample_mean = samples[:, :size].mean(dim=1)
        sample_variance = samples[:, :size].var(dim=1, unbiased=True)
        means.append(sample_mean.numpy())
        variances.append(sample_variance.numpy())
        mean_error = sample_mean - mean
        variance_error = sample_variance - variance
        mean_z = mean_error / (variance / size).sqrt()
        variance_z = variance_error / (variance * (2 / (size - 1))**0.5)
        print(f"{size} samples: mean/variance RMSE={mean_error.square().mean().sqrt():.6f}/{variance_error.square().mean().sqrt():.6f}; "
              f"standard-error units RMS={mean_z.square().mean().sqrt():.3f}/{variance_z.square().mean().sqrt():.3f}, "
              f"max absolute={mean_z.abs().max():.3f}/{variance_z.abs().max():.3f}")
    print("512 uses the first columns of the same 1024 draws, not an independent repeat; seeds: data=41, samples=42.")

    # Optional solver-only comparison on the same small observed block and RHS.
    # No SLQ settings belong in a pathwise linear solve.
    start = perf_counter()
    approximate_weights = linear_cg(observed.matmul, rhs, tolerance=1e-2, max_iter=1000)
    cg_samples = prior[test] + cross @ approximate_weights
    cg_seconds = perf_counter() - start
    cg_mean = cg_samples.mean(dim=1)
    cg_variance = cg_samples.var(dim=1, unbiased=True)
    residual = (observed @ approximate_weights - rhs).norm(dim=0) / rhs.norm(dim=0)
    print(f"Optional GRF-code CG tolerance 1e-2: mean/variance RMSE="
          f"{(cg_mean - mean).square().mean().sqrt():.6f}/{(cg_variance - variance).square().mean().sqrt():.6f}; "
          f"change from Cholesky max={(cg_mean - samples.mean(dim=1)).abs().max():.3e}/"
          f"{(cg_variance - samples.var(dim=1, unbiased=True)).abs().max():.3e}; "
          f"residual mean/max={residual.mean():.3e}/{residual.max():.3e}; solve + update={1e3 * cg_seconds:.3f} ms (unwarmed)")

    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, reference, estimates, title in zip(axes, [mean, variance], [means, variances], ["Latent posterior mean", "Latent marginal variance"]):
        axis.plot(test, reference, "k-", label="Analytic K_q")
        for size, estimate in zip([512, count], estimates):
            axis.plot(test, estimate, ".", label=f"{size} draws")
        axis.set(xlabel="Held-out vertex ID", title=title)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "pathwise_posterior.png", dpi=150)
    plt.close(figure)
    np.savez(output / "pathwise_inputs.npz", test_ids=test.numpy(), train_ids=train.numpy(), observations=y.numpy(),
             mean=mean.numpy(), variance=variance.numpy(), sample_means=means, sample_variances=variances,
             exact_mean=exact_mean.numpy(), exact_variance=exact_variance.numpy(), tau=tau.numpy(), noise=noise,
             sample_counts=[512, count], seeds=[41, 42], setup_seconds=setup_seconds, sampling_seconds=sampling_seconds)


if __name__ == "__main__":
    main()
