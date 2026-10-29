"""Small CG/SLQ diagnostic; CPU float64, no optimization or solver patches."""

import inspect
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import gpytorch
import numpy as np
import torch
from linear_operator.utils.linear_cg import linear_cg

from demo import grid_laplacian, plt
from dense_gp import DenseHeatKernel, GraphGP
from graph_gp import SparseHeatKernel
from chebyshev import apply_chebyshev


def main():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    laplacian = grid_laplacian()
    ids = torch.arange(36)
    train = ids[ids % 3 != 1]
    tau, noise = 2.85602691, 0.04  # Rounded fitted time from demo.py; no refit.
    truth = torch.linalg.cholesky(torch.matrix_exp(-1.4 * laplacian.to_dense())) @ torch.randn(36, dtype=torch.float64)
    y = (truth + noise**0.5 * torch.randn_like(truth))[train]
    kernel = SparseHeatKernel(laplacian, tau)
    with torch.no_grad():
        covariance = kernel(train).evaluate_kernel().add_diagonal(y.new_full((24,), noise))
        polynomial = covariance.to_dense()  # Observed system only, built once.
    exact = torch.matrix_exp(-tau * laplacian.to_dense())[train][:, train] + noise * torch.eye(24, dtype=y.dtype)
    torch.testing.assert_close(polynomial, exact, atol=2e-12, rtol=2e-11)
    condition = torch.linalg.cond(exact).item()
    print(f"6x6 grid, 24 distinct observations, tau={tau}, noise variance={noise}, degree=20")
    print(f"torch={torch.__version__}, gpytorch={gpytorch.__version__}; CPU float64, one thread")
    print(f"CG signature: {inspect.signature(linear_cg)}")
    print(f"Condition exact/polynomial: {condition:.6f} / {torch.linalg.cond(polynomial):.6f}; "
          f"bound={(1 + noise) / (np.exp(-2 * tau) + noise):.6f}; covariance max error={(polynomial - exact).abs().max():.3e}")

    torch.manual_seed(71)
    probes = torch.randn(64, 24, dtype=y.dtype).T
    probes = probes / probes.norm(dim=0)
    solve_rows = []
    for label, rhs in [("single", y[:, None]), ("64 probes + y", torch.cat([probes, y[:, None]], dim=1))]:
        calls = 0
        last_denominator = None

        def counted_matmul(value):
            nonlocal calls, last_denominator
            calls += 1
            product = covariance @ value
            last_denominator = (value * product).sum(dim=0)
            return product

        reference = torch.linalg.solve(exact, rhs)
        # Warm the installed TorchScript updates for each RHS shape; not timed.
        with torch.no_grad():
            linear_cg(counted_matmul, rhs, tolerance=1e-2, max_iter=1000)
        for tolerance, eps in [(1e-2, 1e-10), (1e-4, 1e-10), (1e-6, 1e-10), (1e-10, 1e-10), (1e-10, 1e-20)]:
            calls = 0
            start = perf_counter()
            with torch.no_grad(), gpytorch.settings.terminate_cg_by_size(False):
                solution = linear_cg(counted_matmul, rhs, tolerance=tolerance, eps=eps, max_iter=1000)
            elapsed = perf_counter() - start
            residual = (exact @ solution - rhs).norm(dim=0) / rhs.norm(dim=0)
            error = (solution - reference).norm(dim=0) / reference.norm(dim=0)
            solve_rows.append([rhs.shape[1], tolerance, eps, calls, elapsed, residual.mean().item(), residual.max().item(), error.max().item()])
            print(f"{label:13s} tol={tolerance:.0e} eps={eps:.0e}: residual mean/max={residual.mean():.3e}/{residual.max():.3e}, "
                  f"solution error max={error.max():.3e}, matvecs={calls}, ms={1e3 * elapsed:.3f}, "
                  f"last p^T C p min/max={last_denominator.min():.3e}/{last_denominator.max():.3e}")
    print("Matvec count includes initial residual evaluation; batch stopping uses mean normalized residual.")
    print("Default eps masks small denominators; eps=1e-20 is one explicit sensitivity check, not a default change.")

    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    likelihood.noise = y.new_tensor(noise)
    likelihood.raw_noise.requires_grad_(False)
    dense_model = GraphGP(train, y, likelihood, DenseHeatKernel(laplacian, tau))
    sparse_model = GraphGP(train, y, likelihood, kernel)
    mll_rows = []
    for label in ["dense", "strict", "loose-CG", "GRF-code"]:
        model = dense_model if label == "dense" else sparse_model
        tolerance = 1e-10 if label == "strict" else 1e-2
        samples = 32 if label == "strict" else 64
        lanczos = 1 if label == "GRF-code" else 24
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        parameter = model.covar_module.raw_heat_time
        chain, = torch.autograd.grad(model.covar_module.heat_time, parameter)
        for seed in ([71] if label == "dense" else [71, 72, 73]):
            # One untimed warm-up for this exact case, followed by one measurement.
            for repeat in range(2):
                torch.manual_seed(seed)
                calls = 0
                columns = 0

                def counted_action(graph, coefficients, rhs):
                    nonlocal calls, columns
                    calls += 1
                    columns += rhs.shape[1]
                    return apply_chebyshev(graph, coefficients, rhs)

                with (
                    gpytorch.settings.max_cholesky_size(1000 if label == "dense" else 0),
                    gpytorch.settings.cg_tolerance(tolerance),
                    gpytorch.settings.num_trace_samples(samples),
                    gpytorch.settings.max_lanczos_quadrature_iterations(lanczos),
                    gpytorch.settings.min_preconditioning_size(10**10),
                    gpytorch.settings.max_preconditioner_size(0),
                    gpytorch.settings.fast_computations(log_prob=True),
                    gpytorch.settings.terminate_cg_by_size(False),
                    patch("graph_gp.apply_chebyshev", side_effect=counted_action),
                ):
                    start = perf_counter()
                    loss = -mll(model(train), y)
                    raw_gradient, = torch.autograd.grad(loss, parameter)
                    elapsed = perf_counter() - start
            gradient = (raw_gradient / chain).item()  # d(negative MLL)/d tau, not raw parameter.
            if label == "dense":
                reference_loss, reference_gradient = loss.item(), gradient
                alpha = torch.linalg.solve(exact, y)
                derivative = (-laplacian.to_dense() @ torch.matrix_exp(-tau * laplacian.to_dense()))[train][:, train]
                independent_loss = (y @ alpha + torch.linalg.slogdet(exact)[1] + 24 * np.log(2 * np.pi)) / 48
                independent_gradient = (torch.trace(torch.linalg.solve(exact, derivative)) - alpha @ derivative @ alpha) / 48
                torch.testing.assert_close(loss.detach(), independent_loss, atol=1e-12, rtol=1e-11)
                torch.testing.assert_close(y.new_tensor(gradient), independent_gradient, atol=1e-12, rtol=1e-10)
            mll_rows.append([label, seed, loss.item(), gradient, calls, columns, elapsed])
            print(f"{label:9s} seed={seed}: negative MLL={loss.item():.9f}, gradient={gradient:+.9f}, "
                  f"errors={loss.item() - reference_loss:+.3e}/{gradient - reference_gradient:+.3e}, "
                  f"sign match={np.sign(gradient) == np.sign(reference_gradient)}, recurrence calls/columns={calls}/{columns}, ms={1e3 * elapsed:.3f}")
    print("MLL timings include forward + heat gradient, exclude construction, and include counting overhead.")
    print("Strict: tol=1e-10/32 probes/24 Lanczos; loose-CG: 1e-2/64/24; GRF-code: 1e-2/64/1.")
    print("Only settings were compared: no GRF kernel, private/global mutations, or optimizer was used.")

    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    rows = np.array(solve_rows)
    figure, axis = plt.subplots(figsize=(6, 4))
    for width, label in [(1, "Single RHS"), (65, "64 probes + y (mean)")]:
        chosen = rows[(rows[:, 0] == width) & (rows[:, 2] == 1e-10)]
        axis.loglog(chosen[:, 1], chosen[:, 5], "o-", label=label)
    axis.loglog([1e-10, 1e-2], [1e-10, 1e-2], "k:", label="Achieved = requested")
    axis.axvline(1e-2, color="0.6", linestyle="--", label="GRF-code tolerance")
    axis.set(xlabel="Requested CG tolerance", ylabel="True relative residual")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "cg_residual.png", dpi=150)
    plt.close(figure)
    np.savez(output / "cg_inputs.npz", solves=rows, mll=np.array(mll_rows, dtype=str),
             exact_covariance=exact.numpy(), observations=y.numpy(), condition=condition)


if __name__ == "__main__":
    main()
