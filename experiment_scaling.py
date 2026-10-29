"""Fixed CPU crossover study; --pilot runs the first two cases; CSV resumes arms."""

import csv
import os
import platform
import sys
import warnings
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import gpytorch
import numpy as np
import torch

from chebyshev import apply_chebyshev, heat_coefficients
from demo import grid_laplacian, plt
from dense_gp import DenseHeatKernel, GraphGP
from graph_gp import SparseHeatKernel


def observed_block(kernel, ids: torch.Tensor) -> torch.Tensor:
    """Differentiate existing operator products, with at most 32 basis RHS."""
    operator = kernel(ids).evaluate_kernel()
    blocks = []
    for first in range(0, ids.numel(), 32):
        width = min(32, ids.numel() - first)
        basis = kernel.raw_heat_time.new_zeros(ids.numel(), width)
        basis[torch.arange(first, first + width), torch.arange(width)] = 1
        blocks.append(operator @ basis)
    return torch.cat(blocks, dim=1)


def gaussian_nll(covariance: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    factor = torch.linalg.cholesky(covariance + 0.04 * torch.eye(y.numel(), dtype=y.dtype))
    quadratic = y @ torch.cholesky_solve(y[:, None], factor)[:, 0]
    return (quadratic + 2 * factor.diagonal().log().sum() + y.numel() * np.log(2 * np.pi)) / (2 * y.numel())


def main():
    torch.set_num_threads(min(4, os.cpu_count()))
    print(f"{platform.platform()}, {platform.machine()}, torch {torch.__version__}, gpytorch {gpytorch.__version__}; "
          f"CPU float64, threads={torch.get_num_threads()}", flush=True)
    print("AdamW defaults except lr=0.05, 100 updates, tau start=0.8; degree 20, fixed noise variance=0.04, true tau=1.4")
    print("Setup includes common graph/data cost and one forward/backward warmup without an update; training excludes audits/prediction.")
    print("GRF-code: CG=1e-2/64 probes/1 Lanczos; control: 1e-4/64/24; both force iterative, no preconditioner.")
    print("Prediction: mean only; dense Cholesky or Chebyshev CG tolerance 1e-4 for every sparse arm; no variances checked.")
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    path = output / "scaling_rows.csv"
    rows = []
    if path.exists():
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
    fields = "n m arm seed steps data_s warm_s setup_s train_s total_s tau reported_nll audit_nll rmse prediction_s warnings nll_gap rmse_ratio quality status".split()
    cases = [(20, 128), (40, 128), (20, 256), (40, 512), (60, 128), (60, 512), (80, 128), (80, 512), (100, 128), (100, 512)]
    if "--pilot" in sys.argv:
        cases = cases[:2]
    for side, m in cases:
        n = side**2
        clock = perf_counter()
        graph = grid_laplacian(side)
        generator = torch.Generator().manual_seed(12 + n)
        truth = apply_chebyshev(graph, heat_coefficients(torch.tensor(0.7, dtype=torch.float64)),
                                torch.randn(n, dtype=torch.float64, generator=generator))
        observations = truth + 0.2 * torch.randn(n, dtype=torch.float64, generator=generator)
        order = torch.randperm(n, generator=torch.Generator().manual_seed(23 + n))
        test, train = order[:64], order[64:64 + m]  # Same held-out set and nested uniform train sets for each n.
        y = observations[train]
        data_seconds = perf_counter() - clock
        arms = ["dense", "block", "grf"] if n == 400 else ["block", "grf"]
        if (n, m) in [(400, 128), (1600, 512)]:
            arms.append("control")
        if n == 10000:
            arms = ["grf"]  # Optional sparse-only feasibility; no crossover claim.
        for arm in arms:
            if any(int(row["n"]) == n and int(row["m"]) == m and row["arm"] == arm for row in rows):
                print(f"Already recorded n={n}, m={m}, {arm}; not retried", flush=True)
                continue
            row = dict.fromkeys(fields, float("nan"))
            row.update(n=n, m=m, arm=arm, seed=71, steps=0, data_s=data_seconds, warnings=0, quality="uncertified", status="pending")
            print(f"START n={n}, m={m}, arm={arm}; data/split seeds={12 + n}/{23 + n}, probe seed=71", flush=True)
            messages = []
            standard_warning = warnings.showwarning

            def visible_warning(message, category, filename, lineno, file=None, line=None):
                messages.append(str(message))
                standard_warning(message, category, filename, lineno, file, line)

            model = kernel = optimizer = mll = None
            try:
                clock = perf_counter()
                kernel = DenseHeatKernel(graph, 0.8) if arm == "dense" else SparseHeatKernel(graph, 0.8)
                likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
                likelihood.noise = y.new_tensor(0.04)
                likelihood.raw_noise.requires_grad_(False)
                model = GraphGP(train, y, likelihood, kernel)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

                def objective():
                    if arm == "block":
                        return gaussian_nll(observed_block(kernel, train), y)
                    return -mll(model(train), y)

                with (
                    gpytorch.settings.max_cholesky_size(1000 if arm == "dense" else 0),
                    gpytorch.settings.cg_tolerance(1e-4 if arm == "control" else 1e-2),
                    gpytorch.settings.num_trace_samples(64),
                    gpytorch.settings.max_lanczos_quadrature_iterations(24 if arm == "control" else 1),
                    gpytorch.settings.max_preconditioner_size(0),
                    gpytorch.settings.min_preconditioning_size(10**10),
                    gpytorch.settings.fast_computations(log_prob=True),
                    patch("warnings.showwarning", side_effect=visible_warning),
                ):
                    # A guarded warmup verifies actual sparse products outside training timing.
                    widths = []

                    def checked_action(laplacian, coefficients, rhs):
                        assert laplacian.layout == torch.sparse_coo
                        assert rhs.shape[1] <= 65 and rhs.shape[0] == n
                        widths.append(rhs.shape[1])
                        return apply_chebyshev(laplacian, coefficients, rhs)

                    warm_clock = perf_counter()
                    with patch("graph_gp.apply_chebyshev", side_effect=checked_action):
                        objective().backward()
                    optimizer.zero_grad()
                    row["warm_s"] = perf_counter() - warm_clock
                    row["setup_s"] = data_seconds + perf_counter() - clock
                    print(f"Warmup recurrence calls={len(widths)}, maximum RHS width={max(widths, default=0)}", flush=True)
                    if (n, m, arm) == (400, 128, "dense"):
                        reference = objective()
                        reference_gradient, = torch.autograd.grad(reference, kernel.raw_heat_time)
                        check_kernel = SparseHeatKernel(graph, 0.8)
                        candidate = gaussian_nll(observed_block(check_kernel, train), y)
                        candidate_gradient, = torch.autograd.grad(candidate, check_kernel.raw_heat_time)
                        torch.testing.assert_close(candidate, reference, atol=2e-12, rtol=2e-11)
                        torch.testing.assert_close(candidate_gradient, reference_gradient, atol=2e-12, rtol=2e-10)
                        print(f"Fixed-tau dense/block NLL/raw-gradient errors: {(candidate - reference).item():+.3e}/{(candidate_gradient - reference_gradient).item():+.3e}")
                    torch.manual_seed(71)  # Fresh probes across updates; same stream for matched arms.
                    train_clock = perf_counter()
                    for step in range(100):
                        optimizer.zero_grad()
                        loss = objective()
                        loss.backward()
                        optimizer.step()
                        row["steps"] = step + 1
                        if perf_counter() - train_clock > 900:
                            raise TimeoutError("15-minute arm budget exceeded; incomplete, excluded from crossover")
                    row["train_s"] = perf_counter() - train_clock
                    row["total_s"] = row["setup_s"] + row["train_s"]
                    with torch.no_grad():
                        row["tau"] = kernel.heat_time.item()
                        row["reported_nll"] = objective().item()  # Endpoint evaluation excluded from 100-update timing.
                        audit_kernel = SparseHeatKernel(graph, row["tau"])
                        row["audit_nll"] = gaussian_nll(observed_block(audit_kernel, train), y).item()
                    model.eval()
                    with (
                        torch.no_grad(), gpytorch.settings.skip_posterior_variances(True),
                        gpytorch.settings.max_eager_kernel_size(0), gpytorch.settings.eval_cg_tolerance(1e-4),
                    ):
                        prediction_clock = perf_counter()
                        mean = model(test).mean
                        row["prediction_s"] = perf_counter() - prediction_clock
                        row["rmse"] = (mean - truth[test]).square().mean().sqrt().item()
                    assert np.isfinite([row["tau"], row["audit_nll"], row["reported_nll"], row["rmse"]]).all()
                    row["status"] = "complete"
            except (RuntimeError, MemoryError, TimeoutError, AssertionError) as error:
                row["status"] = f"failed or incomplete: {type(error).__name__}: {error}"
                if row["steps"] > 0 and not np.isfinite(row["train_s"]):
                    row["train_s"] = perf_counter() - train_clock
                    row["total_s"] = row["setup_s"] + row["train_s"]
                print(row["status"], flush=True)
            row["warnings"] = len(messages)
            baseline = next((r for r in rows if int(r["n"]) == n and int(r["m"]) == m and r["arm"] == "block" and r["status"] == "complete"), None)
            if baseline is not None and row["status"] == "complete":
                row["nll_gap"] = row["audit_nll"] - float(baseline["audit_nll"])
                row["rmse_ratio"] = row["rmse"] / float(baseline["rmse"])
                row["quality"] = "pass" if row["nll_gap"] <= 0.02 and row["rmse_ratio"] <= 1.05 else "fail"
            rows.append(row)
            with path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            print(row, flush=True)
            del model, kernel, optimizer, mll  # Release per-arm graph references before the next case.

    # Plot only measured matched cases. Optional n=10000 rows remain feasibility evidence in CSV.
    complete = [r for r in rows if r["status"] == "complete"]
    matched = {(int(r["n"]), int(r["m"])) for r in complete if r["arm"] == "block"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, color in [("dense", "black"), ("block", "tab:blue"), ("grf", "tab:orange"), ("control", "tab:green")]:
        for m, marker in [(128, "o"), (256, "s"), (512, "^")]:
            chosen = sorted([r for r in complete if r["arm"] == arm and int(r["m"]) == m and (int(r["n"]), m) in matched], key=lambda r: int(r["n"]))
            if not chosen:
                continue
            axes[0].loglog([int(r["n"]) for r in chosen], [float(r["train_s"]) for r in chosen], color=color, marker=marker, label=f"{arm}, m={m}")
            if arm in ["grf", "control"]:
                axes[1].plot([int(r["n"]) for r in chosen], [float(r["nll_gap"]) for r in chosen], color=color, marker=marker, label=f"{arm}, m={m}")
    axes[0].set(xlabel="Graph vertices n", ylabel="100-update training seconds")
    axes[1].axhline(0.02, color="black", linestyle="--", label="NLL gate 0.02 (also require RMSE ≤ 1.05×)")
    axes[1].set(xlabel="Graph vertices n", ylabel="Polynomial-Cholesky NLL gap / observation")
    for axis in axes:
        axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(output / "scaling_crossover.png", dpi=150)
    plt.close(figure)
    print(f"Saved {path}; recorded failed/incomplete rows are not retried. Missing scheduled rows remain pending.")


if __name__ == "__main__":
    main()
