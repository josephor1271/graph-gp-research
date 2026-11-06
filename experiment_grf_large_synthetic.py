"""Sparse weighted-road scaling; --smoke first, then the fixed CPU sweep."""

import csv
import hashlib
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import gpytorch
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import expm_multiply
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from experiment_grf_frontier import BoundedDiffusionGRFKernel
from experiment_grf_traffic import (
    CORE, CORE_COMMIT, EXPERIMENT_COMMIT, UPSTREAM, GRFGP, GRFSampler,
    HeatGP, SparseLinearOperator, chebyshev_kernel, plt,
)
from graph_gp import HeatOperator

OUT = Path(__file__).resolve().parent / "results/large_synthetic"
SHAPES = {36: (6, 6), 512: (16, 32), 1000: (25, 40), 2500: (50, 50),
          5000: (50, 100), 10000: (100, 100), 25000: (125, 200), 50000: (200, 250)}


def append_csv(name, row):
    path = OUT / name
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    rows.append(row)
    with path.with_suffix(".tmp").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".tmp").replace(path)


def graph_data(n, seeds=None, output=None):
    """Optional explicit seeds/output support independent replications; defaults are unchanged."""
    directory = OUT if output is None else output
    graph_seed, data_seed, split_seed = (31000+n, 41000+n, 51000+n) if seeds is None else seeds
    clock = perf_counter()
    height, width = SHAPES[n]
    rng = np.random.default_rng(graph_seed)
    coordinates = np.indices((height, width)).reshape(2, -1).T + rng.uniform(-.15, .15, (n, 2))
    nodes = np.arange(n).reshape(height, width)
    first = np.concatenate([nodes[:, :-1].ravel(), nodes[:-1, :].ravel()])
    second = np.concatenate([nodes[:, 1:].ravel(), nodes[1:, :].ravel()])
    diagonal = rng.random((height-1)*(width-1)) < .1
    first = np.concatenate([first, nodes[:-1, :-1].ravel()[diagonal]])
    second = np.concatenate([second, nodes[1:, 1:].ravel()[diagonal]])
    weights = 1/np.linalg.norm(coordinates[first]-coordinates[second], axis=1)
    w = sp.csr_matrix((np.tile(weights, 2), (np.r_[first, second], np.r_[second, first])), shape=(n, n))
    degree = np.diff(w.indptr)
    d = sp.diags(1/np.sqrt(np.asarray(w.sum(1)).ravel()))
    laplacian = (sp.eye(n, format="csr")-d@w@d).tocsr()
    components = connected_components(w, directed=False, return_labels=False)
    assert components == 1 and (w-w.T).nnz == 0 and weights.min() > 0
    metadata = dict(n=n, edges=len(first), mean_degree=float(degree.mean()), max_degree=int(degree.max()),
                    components=components, nnz_l=laplacian.nnz, graph_generation_s=perf_counter()-clock)
    clock = perf_counter()
    rng = np.random.default_rng(data_seed)
    z = rng.standard_normal(n)
    truth = expm_multiply(-.75*laplacian, z)  # Independent SciPy heat half-kernel draw.
    observations = truth + np.sqrt(.1)*rng.standard_normal(n)
    order = np.random.default_rng(split_seed).permutation(n)
    train, test = order[:n//4], order[n//4:n//2]
    assert not np.intersect1d(train, test).size
    mean, scale = observations[train].mean(), observations[train].std()
    normalized = (observations-mean)/scale
    metadata.update(m_train=len(train), m_test=len(test), data_generation_s=perf_counter()-clock,
                    train_mean=float(mean), train_std=float(scale))
    fingerprint = hashlib.sha256(laplacian.data.tobytes()+laplacian.indices.tobytes()+
                                 normalized.tobytes()+order.tobytes()).hexdigest()
    path = directory / f"data_{n}.npz"
    if path.exists():
        with np.load(path) as saved:
            assert str(saved["fingerprint"]) == fingerprint
    else:
        sp.save_npz(directory / f"laplacian_{n}.npz", laplacian)
        np.savez(path, fingerprint=fingerprint, coordinates=coordinates, z=z, f=truth,
                 y=observations, normalized=normalized, train=train, test=test, mean=mean, scale=scale)
    clock = perf_counter()
    coo = laplacian.tocoo()
    graph = torch.sparse_coo_tensor(np.stack([coo.row, coo.col]), coo.data, laplacian.shape).coalesce()
    metadata["graph_generation_s"] += perf_counter()-clock
    return graph, torch.from_numpy(normalized), torch.from_numpy(train), torch.from_numpy(test), metadata


class SparseGuard(TorchDispatchMode):
    """Smoke-only instrumentation: reject dense covariance/features and Cholesky."""

    def __init__(self, n, m):
        super().__init__()
        self.forbidden = {(n, n), (m, m), (n, m), (m, n)}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if "cholesky" in str(func):
            raise RuntimeError(f"Forbidden Cholesky in iterative path: {func}")
        output = func(*args, **(kwargs or {}))
        tensors = output if isinstance(output, (tuple, list)) else [output]
        for tensor in tensors:
            if (isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided and
                    tensor.is_floating_point() and tensor.ndim >= 2 and tuple(tensor.shape[-2:]) in self.forbidden):
                raise RuntimeError(f"Forbidden dense covariance/feature block: {func}, {tuple(tensor.shape)}")
        return output


def small_checks():
    graph, y, train, test, _ = graph_data(36)
    dense = graph.to_dense()
    torch.testing.assert_close(dense, dense.T)
    spectrum = torch.linalg.eigvalsh(dense)
    assert spectrum.min() > -1e-12 and spectrum.max() < 2+1e-12
    with np.load(OUT / "data_36.npz") as data:
        reference = torch.matrix_exp(-.75*dense)@torch.from_numpy(data["z"])
        np.testing.assert_allclose(data["f"], reference.numpy(), atol=2e-13, rtol=2e-12)
    repeat = graph_data(36)
    for actual, expected in zip(repeat[1:4], [y, train, test]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    print("PASS: tiny weighted L, deterministic graph/data/split, SciPy draw vs same-z dense exponential", flush=True)


def run(n, arm, repetition, smoke, messages):
    row = dict(n=n, arm=arm, repetition=repetition, status="incomplete", steps=0, nlpd="unavailable",
               prediction_variance_s="unavailable", heat_calls=0, grf_sparse_calls=0)
    graph, y, train, test, metadata = graph_data(n)
    row.update(metadata)
    tag = f"{n}_{arm}_{repetition}"
    clock = perf_counter()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if arm == "Cheb":
        kernel = chebyshev_kernel(graph)
        model = HeatGP(train, y[train], likelihood, kernel)
    else:
        sampler = GRFSampler(graph, 4096, .5, 3, seed=101, use_tqdm=False, n_processes=4)
        matrices = sampler.sample_random_walk_matrices()
        row["walk_sampling_s"] = perf_counter()-clock
        kernel = BoundedDiffusionGRFKernel(matrices, 3)
        model = GRFGP(train, y[train], likelihood, kernel)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    row["model_setup_s"] = perf_counter()-clock
    raw = kernel.base_kernel.raw_heat_time if arm == "Cheb" else kernel.raw_beta
    amplitude = kernel.raw_outputscale if arm == "Cheb" else kernel.raw_sigma_f
    assert raw.item() == amplitude.item() == 1.0 and likelihood.raw_noise.item() == 0.0
    heat_action, sparse_action = HeatOperator._matmul, SparseLinearOperator._matmul

    def heat_count(operator, rhs):
        row["heat_calls"] += 1
        return heat_action(operator, rhs)

    def sparse_count(operator, rhs):
        row["grf_sparse_calls"] += 1
        return sparse_action(operator, rhs)

    with (
        gpytorch.settings.max_cholesky_size(0), gpytorch.settings.max_eager_kernel_size(0),
        gpytorch.settings.cg_tolerance(1e-4), gpytorch.settings.eval_cg_tolerance(1e-4),
        gpytorch.settings.num_trace_samples(64), gpytorch.settings.max_lanczos_quadrature_iterations(24),
        gpytorch.settings.max_preconditioner_size(0), gpytorch.settings.min_preconditioning_size(10**10),
        gpytorch.settings.fast_computations(log_prob=True),
        patch.object(HeatOperator, "_matmul", heat_count),
        patch.object(SparseLinearOperator, "_matmul", sparse_count),
        SparseGuard(n, len(train)) if smoke else nullcontext(),
    ):
        torch.manual_seed(71)  # Same numerical stream for methods and timing repetitions.
        clock = perf_counter()
        for step in range(2 if smoke else 100):
            optimizer.zero_grad()
            loss = -mll(model(train), y[train])
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            row["steps"] = step+1
            if (step+1) % 10 == 0:
                (OUT / f"progress_{tag}.json").write_text(json.dumps(dict(steps=step+1, training_s=perf_counter()-clock)))
                print(f"UPDATE {tag} {step+1}: NLL={loss.item():.8f}", flush=True)
        row["training_s"] = perf_counter()-clock
        row["training_heat_calls"], row["training_grf_sparse_calls"] = row["heat_calls"], row["grf_sparse_calls"]
        with torch.no_grad():
            row["training_nll"] = -mll(model(train), y[train]).item()  # Endpoint audit, excluded from timing.
            model.eval()
            counts = row["heat_calls"], row["grf_sparse_calls"]
            clock = perf_counter()
            with gpytorch.settings.skip_posterior_variances(True):
                mean = model(test).mean
            row["prediction_mean_s"] = perf_counter()-clock
            row["prediction_heat_calls"] = row["heat_calls"]-counts[0]
            row["prediction_grf_sparse_calls"] = row["grf_sparse_calls"]-counts[1]
            assert torch.isfinite(mean).all()
            row["rmse"] = row["train_std"]*(mean-y[test]).square().mean().sqrt().item()
            row["beta"] = (kernel.base_kernel.heat_time if arm == "Cheb" else kernel.beta).item()
            row["sigma_f"] = (kernel.outputscale.sqrt() if arm == "Cheb" else kernel.sigma_f).item()
            row["noise_variance"] = likelihood.noise.item()
    row["model_total_s"] = row["model_setup_s"]+row["training_s"]+row["prediction_mean_s"]
    if smoke or n == 1000:
        # Independent observed-system reference is outside all benchmark timers/guards.
        with torch.no_grad():
            ids = torch.cat([train, test])
            block = kernel(ids).to_dense()
            observed = block[:len(train), :len(train)] + likelihood.noise*torch.eye(len(train))
            alpha = torch.linalg.solve(observed, y[train])
            reference_mean = block[len(train):, :len(train)]@alpha
            reference_nll = (y[train]@alpha+torch.linalg.slogdet(observed)[1]+len(train)*np.log(2*np.pi))/(2*len(train))
            row.update(mean_max_error=(mean-reference_mean).abs().max().item(), nll_error=row["training_nll"]-reference_nll.item())
            if arm == "Cheb":
                exact = kernel.outputscale*torch.matrix_exp(-kernel.base_kernel.heat_time*graph.to_dense())
                row["cheb_block_max_error"] = (block-exact[ids][:, ids]).abs().max().item()
                assert row["cheb_block_max_error"] < 3e-12
                exact_alpha = torch.linalg.solve(exact[train][:, train]+likelihood.noise*torch.eye(len(train)), y[train])
                row["mean_vs_exact_heat_max_error"] = (mean-exact[test][:, train]@exact_alpha).abs().max().item()
    row["peak_parent_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform == "darwin" else 1024)
    row.update(emitted_warnings=len(messages), numerical_warnings=sum(c == "NumericalWarning" for c, _ in messages),
               warning_messages=json.dumps(messages))
    row["status"] = "complete"
    torch.save(dict(model=model.state_dict(), mean=mean, row=row), OUT / f"model_{tag}{'_smoke' if smoke else ''}.pt")
    append_csv("smoke.csv" if smoke else "results.csv", row)
    print(row, flush=True)


def launch(n, arm, repetition, smoke=False):
    tag = f"{n}_{arm}_{repetition}"
    command = [sys.executable, str(Path(__file__).resolve()), "--case", str(n), arm, str(repetition)]
    if smoke:
        command.append("--smoke")
    clock = perf_counter()
    peak_group = 0.0
    reason = ""
    attempt = len(list(OUT.glob(f"{tag}*.log")))
    log_path = OUT / f"{tag}_{attempt}{'_smoke' if smoke else ''}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                groups = subprocess.check_output(["ps", "-axo", "pgid=,rss="], text=True)
                rss = sum(int(line.split()[1]) for line in groups.splitlines() if int(line.split()[0]) == process.pid)/1024
                peak_group = max(peak_group, rss)
                if perf_counter()-clock > 900 or rss > 12*1024:
                    reason = "900-second wall limit" if perf_counter()-clock > 900 else "sampled sum-RSS exceeded 12 GiB"
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
    status = "complete" if process.returncode == 0 else "incomplete" if reason else "failed"
    if process.returncode != 0 and not reason:
        reason = log_path.read_text().strip().splitlines()[-1]
    progress = OUT / f"progress_{tag}.json"
    last_step = json.loads(progress.read_text())["steps"] if progress.exists() else 0
    row = dict(n=n, arm=arm, repetition=repetition, smoke=smoke, status=status,
               reason=reason or f"exit {process.returncode}", elapsed_s=perf_counter()-clock,
               sampled_peak_group_rss_mib=peak_group, log=log_path.name,
               last_recorded_step=(2 if smoke else 100) if status == "complete" else last_step)
    append_csv("status.csv", row)
    print(row, flush=True)
    return status == "complete"


def figures():
    rows = list(csv.DictReader((OUT / "results.csv").open()))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    accuracy, axis = plt.subplots(figsize=(6, 4))
    for arm, color in [("Cheb", "C0"), ("GRF_C", "C1")]:
        chosen = [r for r in rows if r["arm"] == arm]
        sizes = sorted({int(r["n"]) for r in chosen})
        for plot, metric in zip(axes, ["model_setup_s", "training_s", "model_total_s"]):
            means = [np.mean([float(r[metric]) for r in chosen if int(r["n"]) == n]) for n in sizes]
            plot.loglog(sizes, means, "o-", color=color, label=arm)
            plot.set(xlabel="Vertices n; train/test each n/4", ylabel=f"{metric} (s)")
            plot.legend()
        means = [np.mean([float(r["rmse"]) for r in chosen if int(r["n"]) == n]) for n in sizes]
        axis.semilogx(sizes, means, "o-", color=color, label=arm)
    axis.set(xlabel="Vertices n", ylabel="Noisy-test RMSE, original synthetic units")
    axis.legend()
    fig.tight_layout()
    accuracy.tight_layout()
    fig.savefig(OUT / "runtime.png", dpi=150)
    accuracy.savefig(OUT / "rmse.png", dpi=150)
    plt.close("all")


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    for path, expected in [(CORE, CORE_COMMIT), (UPSTREAM, EXPERIMENT_COMMIT)]:
        assert subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() == expected
    if "--case" in sys.argv:
        messages = []
        original_warning = warnings.showwarning

        def visible_warning(message, category, filename, lineno, file=None, line=None):
            messages.append((category.__name__, str(message)))
            original_warning(message, category, filename, lineno, file, line)

        with patch("warnings.showwarning", visible_warning):
            run(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), "--smoke" in sys.argv, messages)
        return
    if "--figure-only" in sys.argv:
        figures()
        return
    if "--smoke" in sys.argv:
        small_checks()
        for arm in ["Cheb", "GRF_C"]:
            if not launch(512, arm, 0, smoke=True):
                raise RuntimeError("Feasibility gate failed; do not launch the large sweep")
        (OUT / "gate.json").write_text(json.dumps(dict(passed=True, script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))
        return
    gate = json.loads((OUT / "gate.json").read_text())
    assert gate["passed"] and gate["script_sha"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = dict(machine=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                    gpytorch=gpytorch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                    dtype="float64", threads=4, sampler_workers=4, cg=1e-4, eval_cg=1e-4,
                    probes=64, lanczos=24, cholesky=0, preconditioner=0, beta_true=1.5, noise_true=.1,
                    graph_seed="31000+n", data_seed="41000+n", split_seed="51000+n", walk_seed=101,
                    numerical_seed=71, steps=100, optimizer="AdamW lr=.05 defaults", degree=20,
                    initial_raw_parameters=[1, 1, 0], max_cg_iterations=gpytorch.settings.max_cg_iterations.value(),
                    script_sha=gate["script_sha"], limit_seconds=900, limit_sum_rss_gib=12,
                    prediction="conditional mean only; NLPD/variance unavailable",
                    grf=dict(J=3, halt=.5, W=4096), core=CORE_COMMIT, experiments=EXPERIMENT_COMMIT,
                    commands=[f"{sys.executable} {Path(__file__).name} --smoke", f"{sys.executable} {Path(__file__).name}"])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    previous = list(csv.DictReader((OUT / "status.csv").open())) if (OUT / "status.csv").exists() else []
    for n in [1000, 2500, 5000, 10000, 25000, 50000]:
        for repetition in range(3 if n <= 5000 else 1):
            for arm in ["Cheb", "GRF_C"]:
                if any(r["smoke"] == "False" and int(r["n"]) == n and r["arm"] == arm and int(r["repetition"]) == repetition for r in previous):
                    print(f"SKIP recorded attempt {n}/{arm}/{repetition}", flush=True)
                    continue
                launch(n, arm, repetition)
    figures()


if __name__ == "__main__":
    main()
