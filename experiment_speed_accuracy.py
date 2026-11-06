"""Staged, independently tuned mean-only GP frontiers; artifacts stay in results/."""

import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from functools import partial
from time import perf_counter
from unittest.mock import patch
import warnings

import gpytorch
import numpy as np
from scipy.special import ive
import torch

from chebyshev import apply_chebyshev, heat_coefficients
from experiment_grf_large_synthetic import graph_data, SparseGuard
from experiment_grf_frontier import BoundedDiffusionGRFKernel
from experiment_grf_traffic import (
    CORE, CORE_COMMIT, EXPERIMENT_COMMIT, UPSTREAM, GRFGP, GRFSampler,
    HeatGP, SparseLinearOperator, chebyshev_kernel, plt, posterior,
)
from graph_gp import HeatOperator

OUT = Path(__file__).resolve().parent / "results/speed_accuracy"
MENU = [(8, 8), (16, 16), (32, 16), (32, 24), (64, 24)]
DEGREES = [6, 8, 10, 12, 16, 20]
GRF_KERNELS = [(3, .5, 256), (3, .5, 1024), (3, .5, 4096),
               (5, .5, 1024), (5, .5, 4096), (10, .1, 1024)]


def protocol():
    return dict(question="Independently tuned numerical/predictive speed-accuracy frontiers",
                baseline_main="2dbf18d850ebb07618e43f3ff2cf7870740ea0cf",
                degrees=DEGREES, inference_menu=MENU, grf_kernels=GRF_KERNELS,
                development_seeds=["131000+n", "141000+n", "151000+n"],
                final_seeds=["231000+n+100000*r", "241000+n+100000*r", "251000+n+100000*r"],
                development_walk_seeds=[101, 202], final_walk_seed="301+r", trace_seed=71,
                steps=100, optional_matched_steps=200, optimizer="AdamW .05 defaults",
                cg=1e-4, cholesky=0, preconditioner=0, dtype="float64", torch_threads=4, sampler_workers=4,
                numerical_screen=dict(max_nll_per_obs=.02, max_beta_gradient_absolute=.01, max_true_residual=.0002,
                    common_states=[(1.4621171572600098, 1.3132616875182228, .6932471805599453), (1.8, 1.5, .2)],
                    trace_seeds=[71, 72, 73]),
                predictive_screen="RMSE <= 1.01 * same-method original high-numerical-accuracy development reference",
                qualified_large_comparisons="Passing numerical screen; rejected settings stay visible on development frontier",
                final_replicates={5000: 3, 10000: 3, 25000: 2, 50000: 1},
                watchdog=dict(seconds=1800, sampled_sum_rss_gib=12), uncertainty="unavailable; mean-only workload")


def append(name, row):
    path = OUT / name
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    rows.append(row)
    with path.with_suffix(".tmp").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".tmp").replace(path)


def configuration(method, degree=20, probes=64, lanczos=24, walks=4096, cutoff=3, halt=.5):
    return dict(method=method, degree=degree, probes=probes, lanczos=lanczos,
                walks=walks, cutoff=cutoff, halt=halt, cg=1e-4)


def label(c):
    kernel = f"d{c['degree']}" if c["method"] == "Cheb" else f"j{c['cutoff']}-w{c['walks']}-h{c['halt']}"
    return f"{c['method']}-{kernel}-p{c['probes']}-l{c['lanczos']}"


@contextmanager
def controls(c):
    # The coefficient function exposes degree; the production kernel fixes its default.
    # This per-worker scoped override restores that function and never edits core code.
    with (
        patch("graph_gp.heat_coefficients", partial(heat_coefficients, degree=c["degree"])),
        gpytorch.settings.max_cholesky_size(0), gpytorch.settings.max_eager_kernel_size(0),
        gpytorch.settings.cg_tolerance(c["cg"]), gpytorch.settings.eval_cg_tolerance(c["cg"]),
        gpytorch.settings.num_trace_samples(c["probes"]),
        gpytorch.settings.max_lanczos_quadrature_iterations(c["lanczos"]),
        gpytorch.settings.max_preconditioner_size(0), gpytorch.settings.min_preconditioning_size(10**10),
        gpytorch.settings.fast_computations(log_prob=True),
    ):
        yield


def dataset(n, phase="dev", replicate=0):
    directory = OUT / f"data_{phase}_{replicate}"
    directory.mkdir(exist_ok=True)
    offset = 100000 if phase == "dev" else 200000+100000*replicate
    seeds = tuple(base+n+offset for base in [31000, 41000, 51000])
    return graph_data(n, seeds=seeds, output=directory)


def model_for(graph, train, y, c, walk_seed):
    clock = perf_counter()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    walk_s = 0.0
    if c["method"] == "Cheb":
        kernel = chebyshev_kernel(graph)
        model = HeatGP(train, y, likelihood, kernel)
    else:
        sampler = GRFSampler(graph, c["walks"], c["halt"], c["cutoff"], seed=walk_seed, use_tqdm=False, n_processes=4)
        matrices = sampler.sample_random_walk_matrices()
        walk_s = perf_counter()-clock
        model = GRFGP(train, y, likelihood, BoundedDiffusionGRFKernel(matrices, c["cutoff"]))
    return model, perf_counter()-clock, walk_s


def physical(model):
    kernel = model.covar_module
    cheb = isinstance(model, HeatGP)
    raw = kernel.base_kernel.raw_heat_time if cheb else kernel.raw_beta
    beta = kernel.base_kernel.heat_time if cheb else kernel.beta
    sigma = kernel.outputscale.sqrt() if cheb else kernel.sigma_f
    return raw, beta, sigma, model.likelihood.noise


def degree_checks():
    directory = OUT / "default_generator_check"
    directory.mkdir(exist_ok=True)
    graph, _, _, _, _ = graph_data(36, output=directory)
    # Recorded from the unmodified main generator in the pinned traffic environment.
    with np.load(directory / "data_36.npz") as new:
        assert str(new["fingerprint"]) == "08f6975bc2a7106295f73a746c31fceea1fb0b156b96baf757b43ed518c6688f"
    dense, identity = graph.to_dense(), torch.eye(36)
    lam = np.linspace(0, 2, 10001)
    safe = []
    for degree in DEGREES:
        minima = []
        for value in [.1, .5, 1.0, 1.5, 1.9, 1.999, 2.0]:
            beta = torch.tensor(value, requires_grad=True)
            coefficients = heat_coefficients(beta, degree)
            derivative = torch.stack([torch.autograd.grad(v, beta, retain_graph=True)[0] for v in coefficients])
            polynomial = np.polynomial.chebyshev.chebval(lam-1, coefficients.detach().numpy())
            slope = np.polynomial.chebyshev.chebval(lam-1, derivative.detach().numpy())
            exact = torch.matrix_exp(-beta.detach()*dense)
            candidate = apply_chebyshev(graph, coefficients, identity)
            candidate_derivative = apply_chebyshev(graph, derivative, identity)
            ideal = 2*(-1.)**np.arange(degree+1)*ive(np.arange(degree+1), value)
            ideal[0] /= 2
            row = dict(degree=degree, beta=value, spectral_max=float(np.max(abs(polynomial-np.exp(-value*lam)))),
                       derivative_max=float(np.max(abs(slope+lam*np.exp(-value*lam)))), minimum=float(polynomial.min()),
                       matrix_max=float((candidate.detach()-exact).abs().max()),
                       matrix_derivative_max=float((candidate_derivative+dense@exact).abs().max()),
                       ideal_tail=2*float(ive(np.arange(degree+1, 200), value).sum()),
                       coefficient_difference=float(np.max(abs(coefficients.detach().numpy()-ideal))))
            append("degrees.csv", row)
            minima.append(row["minimum"])
        if min(minima) > 0:
            safe.append(degree)
    (OUT / "safe_degrees.json").write_text(json.dumps(safe))
    assert safe == DEGREES, "Empirical polynomial positivity failed; stop and review"
    print(f"Numerically positive degrees on audited beta/spectral grids: {safe}; not a formal PSD certificate", flush=True)


def numerical_checks(candidates=None):
    """Compare stochastic inference with Cholesky for the identical finite kernel."""
    graph, y, train, test, _ = dataset(1000)
    rows = []
    name = "numerical" if candidates is None else "finalist_numerical"
    candidates = candidates or [configuration(method, probes=p, lanczos=l)
                                for method in ["Cheb", "GRF"] for p, l in MENU]
    for c in candidates:
        method = c["method"]
        model, _, _ = model_for(graph, train, y[train], c, 101)
        raw, _, _, _ = physical(model)
        kernel = model.covar_module
        for state, (beta, sigma, noise) in enumerate([(1.4621171572600098, 1.3132616875182228, .6932471805599453), (1.8, 1.5, .2)]):
            signal_raw = kernel.raw_outputscale if method == "Cheb" else kernel.raw_sigma_f
            with torch.no_grad():
                raw.copy_(torch.logit(torch.tensor(beta/2)))
                signal_raw.copy_(torch.log(torch.expm1(torch.tensor(sigma))))
            model.likelihood.noise = noise
            with controls(c):
                reference, mean, _ = posterior(kernel(torch.cat([train, test])).to_dense(), y[train], model.likelihood.noise)
                chain = beta*(1-beta/2)
                gradient = torch.autograd.grad(reference, raw)[0].item()/chain
            for seed in [71, 72, 73]:
                model.train()
                torch.manual_seed(seed)
                with controls(c):
                    loss = -gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)(model(train), y[train])
                    grad = torch.autograd.grad(loss, raw)[0].item()/chain
                    model.eval()
                    with torch.no_grad(), gpytorch.settings.skip_posterior_variances(True):
                        prediction = model(test).mean
                        alpha = model.prediction_strategy.mean_cache
                        residual = kernel(train)@alpha+model.likelihood.noise*alpha-y[train]
                row = dict(configuration=label(c), method=method, probes=c["probes"], lanczos=c["lanczos"], state=state, trace_seed=seed,
                           nll_error=loss.item()-reference.item(), beta_gradient_error=grad-gradient,
                           exact_beta_gradient=gradient, estimated_beta_gradient=grad,
                           mean_max_error=float((prediction-mean.detach()).abs().max()),
                           true_residual=float(residual.norm()/y[train].norm()))
                rows.append(row)
                append(f"{name}.csv", row)
    # Prespecified exploratory numerical fidelity screen, separate from the 1% predictive screen.
    approved = []
    for c in candidates:
        a = [r for r in rows if r["configuration"] == label(c)]
        passed = (max(abs(r["nll_error"]) for r in a) <= .02 and
                  max(abs(r["beta_gradient_error"]) for r in a) <= .01 and
                  max(r["true_residual"] for r in a) <= 2e-4 and
                  all(r["estimated_beta_gradient"]*r["exact_beta_gradient"] >= 0 or abs(r["exact_beta_gradient"]) <= .01 for r in a))
        approved.append(dict(configuration=label(c), method=c["method"], probes=c["probes"], lanczos=c["lanczos"], passed=passed))
    (OUT / f"{name}_gate.json").write_text(json.dumps(approved, indent=2))
    print(approved, flush=True)
    if name == "numerical":
        assert all(a["passed"] for a in approved if a["probes"] == 64), "Reference settings failed; stop"


def numerical_pass(c):
    specific = OUT / "finalist_numerical_gate.json"
    if specific.exists():
        for g in json.loads(specific.read_text()):
            if g["configuration"] == label(c):
                return g["passed"]
    gates = json.loads((OUT / "numerical_gate.json").read_text())
    return next(g["passed"] for g in gates if (g["method"], g["probes"], g["lanczos"]) ==
                (c["method"], c["probes"], c["lanczos"]))


def job(c, n, stage, replicate=0, walk_seed=101, steps=100):
    phase_key = stage if stage in ["final", "confirm", "smoke", "extended"] else "dev"
    key = f"{phase_key}-{n}-r{replicate}-w{walk_seed}-{label(c)}-s{steps}"
    return dict(key=key, config=c, n=n, stage=stage, replicate=replicate,
                walk_seed=walk_seed, steps=steps, data="final" if stage == "final" else "dev")


def fit(j, messages):
    started_utc, started_clock = datetime.now(timezone.utc), perf_counter()
    c, n = j["config"], j["n"]
    directory = OUT / "jobs" / j["key"]
    graph, y, train, test, meta = dataset(n, j["data"], j["replicate"])
    row = {k: v for k, v in j.items() if k != "config"}
    row.update(c, **meta, settings=json.dumps(c), status="incomplete", completed_steps=0,
               numerical_screen=numerical_pass(c), nlpd="unavailable", heat_calls=0, sparse_calls=0)
    checkpoint = directory / "checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=True)
        if saved["numerical_warnings"]:
            saved = None  # A failed numerical attempt is retryable, not silently accepted.
    else:
        saved = None
    if saved is not None and c["method"] == "GRF":
        clock = perf_counter()
        matrices = [SparseLinearOperator(t) for t in torch.load(directory / "walks.pt", weights_only=True)]
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GRFGP(train, y[train], likelihood, BoundedDiffusionGRFKernel(matrices, c["cutoff"]))
        setup, walk_s = saved["setup_s"], saved["walk_s"]
    else:
        model, setup, walk_s = model_for(graph, train, y[train], c, j["walk_seed"])
        if c["method"] == "GRF":
            torch.save([m.sparse_csr_tensor for m in model.covar_module.rw_mats], directory / "walks.pt")
    clock = perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    setup += perf_counter()-clock
    raw, beta, sigma, noise = physical(model)
    assert raw.item() == 1.0 and abs(beta.item()-1.4621171572600098) < 1e-14
    assert abs(sigma.item()-1.3132616875182228) < 1e-14 and abs(noise.item()-.6932471805599453) < 1e-14
    step, training_s, history = 0, 0.0, []
    torch.manual_seed(71)
    if saved is not None:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.random.set_rng_state(saved["rng"])
        step, training_s, history = saved["step"], saved["training_s"], saved["history"]
        setup, walk_s = saved["setup_s"], saved["walk_s"]
        row.update(heat_calls=saved["heat_calls"], sparse_calls=saved["sparse_calls"])
    heat_action, sparse_action = HeatOperator._matmul, SparseLinearOperator._matmul

    def heat_count(operator, rhs):
        row["heat_calls"] += 1
        return heat_action(operator, rhs)

    def sparse_count(operator, rhs):
        row["sparse_calls"] += 1
        return sparse_action(operator, rhs)

    with controls(c), patch.object(HeatOperator, "_matmul", heat_count), patch.object(SparseLinearOperator, "_matmul", sparse_count):
        with SparseGuard(n, len(train)) if j["stage"] == "smoke" else nullcontext():
            model.train()
            for endpoint in ([2] if j["steps"] == 2 else [v for v in [25, 50, 75, 100, 150, 200] if v <= j["steps"]]):
                if endpoint <= step:
                    continue
                clock = perf_counter()
                for _ in range(step, endpoint):
                    optimizer.zero_grad()
                    loss = -mll(model(train), y[train])
                    assert torch.isfinite(loss)
                    loss.backward()
                    optimizer.step()
                    step += 1
                    history.append([step, loss.item()])
                training_s += perf_counter()-clock
                saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=torch.random.get_rng_state(),
                             step=step, training_s=training_s, setup_s=setup, walk_s=walk_s, history=history,
                             heat_calls=row["heat_calls"], sparse_calls=row["sparse_calls"],
                             numerical_warnings=sum(category == "NumericalWarning" for category, _ in messages))
                torch.save(saved, checkpoint.with_suffix(".tmp"))
                checkpoint.with_suffix(".tmp").replace(checkpoint)
                if endpoint in [25, 50, 100, 200]:
                    torch.save(saved, directory / f"step{endpoint}.pt")
                print(f"UPDATE {j['key']}: {step} updates, cumulative train {training_s:.4f}s", flush=True)
            row.update(completed_steps=step, setup_s=setup, walk_s=walk_s, training_s=training_s,
                       training_heat_calls=row["heat_calls"], training_sparse_calls=row["sparse_calls"])
            with torch.no_grad():
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(71)
                    row["estimated_nll"] = -mll(model(train), y[train]).item()
                model.eval()
                clock = perf_counter()
                with gpytorch.settings.skip_posterior_variances(True):
                    mean = model(test).mean
                row["prediction_s"] = perf_counter()-clock
                alpha = model.prediction_strategy.mean_cache
                residual = model.covar_module(train)@alpha+model.likelihood.noise*alpha-y[train]
                row["true_residual"] = (residual.norm()/y[train].norm()).item()
                row["rmse"] = meta["train_std"]*(mean-y[test]).square().mean().sqrt().item()
                raw, beta, sigma, noise = physical(model)
                row.update(beta=beta.item(), sigma_f=sigma.item(), noise_variance=noise.item())
        if n <= 2500:
            # Fitted own-kernel reference is off-timer and outside the sparse-path smoke.
            ids = torch.cat([train, test])
            exact_nll, exact_mean, _ = posterior(model.covar_module(ids).to_dense(), y[train], model.likelihood.noise)
            derivative = torch.autograd.grad(exact_nll, raw)[0].item()/(beta.item()*(1-beta.item()/2))
            model.train()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(71)
                estimated = -mll(model(train), y[train])
                estimated_gradient = torch.autograd.grad(estimated, raw)[0].item()/(beta.item()*(1-beta.item()/2))
            model.eval()
            row.update(beta_gradient_error=estimated_gradient-derivative, exact_nll=exact_nll.item(), nll_error=row["estimated_nll"]-exact_nll.item(),
                       exact_beta_gradient=derivative, mean_error=(mean-exact_mean.detach()).abs().max().item())
        row["total_s"] = setup+training_s+row["prediction_s"]
        row.update(started_utc=started_utc.isoformat(), finished_utc=datetime.now(timezone.utc).isoformat(),
                   clock_gap_s=(datetime.now(timezone.utc)-started_utc).total_seconds()-(perf_counter()-started_clock))
        row["peak_parent_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform == "darwin" else 1024)
        row.update(warnings=json.dumps(messages), numerical_warnings=sum(category == "NumericalWarning" for category, _ in messages))
        row["status"] = "complete" if step == j["steps"] and row["numerical_warnings"] == 0 and row["true_residual"] <= 2*c["cg"] else "numerical_failure"
        assert np.isfinite([row[k] for k in ["rmse", "estimated_nll", "true_residual", "total_s"]]).all()
        torch.save(dict(model=model.state_dict(), mean=mean, row=row), directory / "model.pt")
        (directory / "result.json").write_text(json.dumps(row, indent=2))
        append("runs.csv", row)
        print(row, flush=True)


def launch(j):
    directory = OUT / "jobs" / j["key"]
    directory.mkdir(parents=True, exist_ok=True)
    result = directory / "result.json"
    if result.exists() and json.loads(result.read_text())["status"] == "complete":
        print(f"SKIP completed {j['key']}", flush=True)
        return json.loads(result.read_text())
    (directory / "job.json").write_text(json.dumps(j, indent=2))
    attempt = len(list(directory.glob("attempt*.log")))
    log_path = directory / f"attempt{attempt}.log"
    started_utc = datetime.now(timezone.utc)
    start, peak_group, reason = perf_counter(), 0.0, ""
    with log_path.open("w") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--case", str(directory / "job.json")],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                listing = subprocess.check_output(["ps", "-axo", "pgid=,rss="], text=True)
                rss = sum(int(line.split()[1]) for line in listing.splitlines() if int(line.split()[0]) == process.pid)/1024
                peak_group = max(peak_group, rss)
                if perf_counter()-start > 1800 or rss > 12*1024:
                    reason = "1800s perf_counter budget" if perf_counter()-start > 1800 else "12GiB sampled sum-RSS budget"
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
    status = json.loads(result.read_text()) if process.returncode == 0 and result.exists() else dict(status="incomplete" if reason else "failed")
    append("attempts.csv", dict(key=j["key"], stage=j["stage"], n=j["n"], configuration=label(j["config"]),
                                attempt=attempt, status=status["status"], reason=reason or f"exit {process.returncode}",
                                wall_s=perf_counter()-start, wall_utc_s=(datetime.now(timezone.utc)-started_utc).total_seconds(),
                                sampled_peak_group_rss_mib=peak_group, log=str(log_path)))
    print(f"RESULT {j['key']}: {status.get('status')}, total={status.get('total_s')}, RMSE={status.get('rmse')}", flush=True)
    return status


def development_a():
    assert (OUT / "numerical_gate.json").exists()
    for degree in json.loads((OUT / "safe_degrees.json").read_text()):
        launch(job(configuration("Cheb", degree=degree), 2500, "degree"))


def development_b():
    degree_rows = [r for r in completed() if r["stage"] == "degree"]
    baseline = next(r for r in degree_rows if r["degree"] == 20)
    promising = sorted([r for r in degree_rows if r["degree"] != 20 and r["rmse"] <= 1.01*baseline["rmse"]], key=lambda r:r["total_s"])[:2]
    degrees = [20]+[r["degree"] for r in promising]
    (OUT / "degree_interactions.json").write_text(json.dumps(degrees))
    for degree in degrees:
        for probes, lanczos in MENU:
            launch(job(configuration("Cheb", degree=degree, probes=probes, lanczos=lanczos), 2500, "solver"))
    for probes, lanczos in MENU:
        launch(job(configuration("GRF", probes=probes, lanczos=lanczos), 2500, "solver"))


def completed():
    return [json.loads(p.read_text()) for p in (OUT / "jobs").glob("*/result.json") if json.loads(p.read_text())["status"] == "complete"]


def development_d():
    rows = [r for r in completed() if r["stage"] == "solver" and r["method"] == "GRF"]
    reference = next(r for r in rows if r["probes"] == 64)
    eligible = [r for r in rows if numerical_pass(r) and r["rmse"] <= 1.01*reference["rmse"]]
    best = min(eligible, key=lambda r: r["total_s"])
    budgets = [(best["probes"], best["lanczos"])]
    if budgets[0] != (64, 24):
        budgets.append((64, 24))
    (OUT / "grf_inference.json").write_text(json.dumps(budgets))
    for cutoff, halt, walks in GRF_KERNELS:
        for probes, lanczos in budgets:
            c = configuration("GRF", cutoff=cutoff, halt=halt, walks=walks, probes=probes, lanczos=lanczos)
            launch(job(c, 2500, "features"))


def kernel_checks():
    graph, _, train, test, _ = dataset(1000)
    dense, ids = graph.to_dense(), torch.cat([train, test])
    for cutoff, halt, walks in GRF_KERNELS:
        c = configuration("GRF", cutoff=cutoff, halt=halt, walks=walks)
        model, setup, _ = model_for(graph, train, torch.zeros(len(train)), c, 101)
        kernel = model.covar_module
        with torch.no_grad():
            for beta in [1.0, 1.9]:
                kernel.initialize(raw_beta=torch.logit(torch.tensor(beta/2)),
                                  raw_sigma_f=torch.log(torch.expm1(torch.tensor(1.0))))
                heat = torch.matrix_exp(-beta*dense)[ids][:, ids]
                term, q = torch.eye(len(graph)), torch.eye(len(graph))
                for power in range(1, cutoff):
                    term = (-beta/(2*power))*(term@dense)
                    q = q+term
                truncated = q[ids]@q[ids].T
                sampled = kernel(ids).to_dense()
                append("kernel_errors.csv", dict(cutoff=cutoff, halt=halt, walks=walks, beta=beta,
                       truncation=float((truncated-heat).norm()/heat.norm()),
                       random_walk=float((sampled-truncated).norm()/truncated.norm()),
                       overall=float((sampled-heat).norm()/heat.norm()),
                       diagonal_max_abs=float((sampled-heat).diagonal().abs().max()), setup_s=setup))


def select_development():
    rows = [r for r in completed() if r["data"] == "dev" and r["n"] == 2500 and r["steps"] == 100]
    if not (OUT / "finalist_numerical_gate.json").exists():
        candidates = [{k: r[k] for k in configuration("Cheb")} for r in rows if numerical_pass(r)]
        numerical_checks(candidates)
    selected = []
    for method in ["Cheb", "GRF"]:
        baseline = configuration(method)
        reference = next(r for r in rows if label(r) == label(baseline))
        qualified = [r for r in rows if r["method"] == method and numerical_pass(r) and
                     r["rmse"] <= 1.01*reference["rmse"] and label(r) != label(baseline)]
        frontier = [r for r in qualified if not any(q["total_s"] <= r["total_s"] and q["rmse"] <= r["rmse"] and
                    (q["total_s"] < r["total_s"] or q["rmse"] < r["rmse"]) for q in qualified)]
        fastest = min(frontier, key=lambda r: r["total_s"])
        choices = [baseline, {k: fastest[k] for k in baseline}]
        remaining = [r for r in frontier if label(r) != label(fastest)]
        if remaining:
            accurate = min(remaining, key=lambda r: r["rmse"])
            choices.append({k: accurate[k] for k in baseline})
        if method == "GRF":
            # Retain an inference-only control even when a different covariance dominates on one draw.
            original_kernel = [r for r in qualified if (r["cutoff"], r["halt"], r["walks"]) == (3, .5, 4096)]
            control = min(original_kernel, key=lambda r: r["total_s"])
            if label(control) not in [label(c) for c in choices]:
                choices.append({k: control[k] for k in baseline})
        selected.extend(choices)
    (OUT / "development_candidates.json").write_text(json.dumps(selected, indent=2))
    for c in selected:
        for seed in ([101, 202] if c["method"] == "GRF" else [101]):
            launch(job(c, 5000, "confirm", walk_seed=seed))


def freeze():
    path = OUT / "finalists.json"
    assert not path.exists(), "Finalists already frozen; never select from final outcomes"
    assert not any(r["stage"] == "final" for r in completed())
    candidates = json.loads((OUT / "development_candidates.json").read_text())
    rows = [r for r in completed() if r["stage"] == "confirm"]
    qualified = []
    leaders = []
    for method in ["Cheb", "GRF"]:
        baseline = configuration(method)
        group = []
        for c in [c for c in candidates if c["method"] == method]:
            measurements = [r for r in rows if label(r) == label(c)]
            assert len(measurements) == (2 if method == "GRF" else 1)
            passes = numerical_pass(c) and all(r["rmse"] <= 1.01*next(
                b["rmse"] for b in rows if label(b) == label(baseline) and b["walk_seed"] == r["walk_seed"])
                for r in measurements)
            append("selection.csv", dict(configuration=label(c), confirm_pass=passes,
                   mean_rmse=np.mean([r["rmse"] for r in measurements]), mean_total_s=np.mean([r["total_s"] for r in measurements])))
            if passes or label(c) == label(baseline):
                group.append(c)
        leader = min(group, key=lambda c: np.mean([r["total_s"] for r in rows if label(r) == label(c)]))
        leaders.append(leader)
        qualified.extend(group)
    manifest = dict(candidates=qualified, leaders=leaders, baselines=[configuration(m) for m in ["Cheb", "GRF"]],
                    frozen_utc=datetime.now(timezone.utc).isoformat(),
                    protocol=json.loads((OUT / "protocol.json").read_text()),
                    upstream_pins=dict(core=CORE_COMMIT, experiments=EXPERIMENT_COMMIT),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    selection="Development numerical gate plus <=1% RMSE at n2500 and confirmation n5000, both GRF walk seeds; fastest qualified leads; retain original-C inference-only control")
    path.write_text(json.dumps(manifest, indent=2))


def extended_runs():
    """Matched optional step-budget check; never pooled with 100-step totals."""
    manifest = json.loads((OUT / "finalists.json").read_text())
    for c in manifest["leaders"]:
        launch(job(c, 5000, "extended", steps=200))


def final_runs():
    manifest = json.loads((OUT / "finalists.json").read_text())
    for n, count in [(5000, 3), (10000, 3), (25000, 2), (50000, 1)]:
        candidates = manifest["candidates"] if n <= 10000 else manifest["leaders"]+manifest["baselines"]
        seen = set()
        for c in candidates:
            if label(c) in seen:
                continue
            seen.add(label(c))
            for replicate in range(count):
                launch(job(c, n, "final", replicate=replicate, walk_seed=301+replicate))


def figures():
    """Regenerate four descriptive plots from completed measurements only."""
    degrees = list(csv.DictReader((OUT / "degrees.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for field, axis in zip(["spectral_max", "derivative_max"], axes):
        axis.semilogy(DEGREES, [max(float(r[field]) for r in degrees if int(r["degree"]) == d) for d in DEGREES], "o-", label="Implemented, max over beta grid")
        axis.set(xlabel="Degree", ylabel=field.replace("_", " "))
    axes[0].semilogy(DEGREES, [max(float(r["ideal_tail"]) for r in degrees if int(r["degree"]) == d) for d in DEGREES], "--", label="Ideal Bessel tail")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "degree_errors.png", dpi=160)
    plt.close(fig)
    rows = completed()
    dev = [r for r in rows if r["data"] == "dev" and r["n"] == 2500 and r["steps"] == 100]
    manifest = json.loads((OUT / "finalists.json").read_text())
    originals = [label(c) for c in manifest["baselines"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, field in zip(axes, ["training_s", "total_s"]):
        for method, color in [("Cheb", "tab:blue"), ("GRF", "tab:orange")]:
            for r in [r for r in dev if r["method"] == method]:
                marker = "*" if label(r) in originals else ("o" if numerical_pass(r) else "x")
                axis.scatter(r[field], r["rmse"], marker=marker, color=color, s=95 if marker == "*" else 25)
            reference = next(r for r in dev if label(r) == label(configuration(method)))
            axis.axhline(1.01*reference["rmse"], color=color, linestyle=":", linewidth=.7)
            axis.scatter([], [], color=color, label=method)
        axis.set(xlabel=field.replace("_", " "), ylabel="Noisy-observation RMSE, raw units")
        axis.legend()
    fig.suptitle("Development n=2500: stars original; crosses fail numerical screen; dotted +1%")
    fig.tight_layout()
    fig.savefig(OUT / "development_frontier.png", dpi=160)
    plt.close(fig)
    final = [r for r in rows if r["stage"] == "final"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c in manifest["candidates"]:
        group = [r for r in final if label(r) == label(c)]
        sizes = sorted({r["n"] for r in group})
        for field, axis in zip(["total_s", "rmse"], axes):
            values = [[r[field] for r in group if r["n"] == n] for n in sizes]
            axis.errorbar(sizes, [np.mean(v) for v in values], yerr=[np.std(v, ddof=1) if len(v) > 1 else 0 for v in values],
                          marker="o", linestyle="--" if label(c) in originals else "-", label=label(c), capsize=2)
            axis.set(xscale="log", xlabel="Vertices", ylabel=field.replace("_", " "))
        if len(sizes) >= 3:
            exponent = np.polyfit(np.log(sizes), np.log([np.mean([r["total_s"] for r in group if r["n"] == n]) for n in sizes]), 1)[0]
            append("finite_range_exponents.csv", dict(configuration=label(c), sizes=json.dumps(sizes), exponent=float(exponent)))
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=6)
    fig.suptitle("Frozen fresh seeds: mean ± sample SD; 3/3/2/1 datasets at 5K/10K/25K/50K")
    fig.tight_layout()
    fig.savefig(OUT / "frozen_scaling.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    for axis, measurements in zip(axes, [dev, final]):
        groups = sorted({(r["n"], label(r)) for r in measurements if label(r) in [label(c) for c in manifest["candidates"]]})
        left = np.zeros(len(groups))
        for field in ["setup_s", "training_s", "prediction_s"]:
            values = [np.mean([r[field] for r in measurements if (r["n"], label(r)) == g]) for g in groups]
            axis.barh(range(len(groups)), values, left=left, label=field)
            left += values
        axis.set(yticks=range(len(groups)), yticklabels=[f"{n}: {name}" for n, name in groups], xlabel="Seconds")
        axis.tick_params(axis="y", labelsize=6)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "cost_breakdown.png", dpi=160)
    plt.close(fig)


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    path = OUT / "protocol.json"
    if not path.exists():
        path.write_text(json.dumps(protocol(), indent=2))
    assert json.loads(path.read_text()) == json.loads(json.dumps(protocol())), "Preserve the declared protocol"
    for path, expected in [(CORE, CORE_COMMIT), (UPSTREAM, EXPERIMENT_COMMIT)]:
        assert subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() == expected
    if "--case" in sys.argv:
        j = json.loads(Path(sys.argv[2]).read_text())
        messages = []
        original_warning = warnings.showwarning

        def visible_warning(message, category, filename, lineno, file=None, line=None):
            messages.append((category.__name__, str(message)))
            original_warning(message, category, filename, lineno, file, line)

        with patch("warnings.showwarning", visible_warning):
            fit(j, messages)
        return
    if "--smoke" in sys.argv:
        candidates = [configuration("Cheb",degree=6,probes=8,lanczos=8), configuration("Cheb"),
                      configuration("GRF",walks=256,probes=8,lanczos=8),
                      configuration("GRF",cutoff=5,walks=1024,probes=16,lanczos=16),
                      configuration("GRF",cutoff=10,halt=.1,walks=1024)]
        for c in candidates:
            assert launch(job(c,512,"smoke",steps=2))["status"] == "complete"
        (OUT / "smoke_passed.json").write_text(json.dumps(dict(passed=True)))
    if "--degree-screen" in sys.argv:
        assert (OUT / "smoke_passed.json").exists()
        development_a()
    if "--solver-screen" in sys.argv:
        development_b()
    if "--numerics" in sys.argv:
        degree_checks()
        numerical_checks()
    if "--features" in sys.argv:
        development_d()
        kernel_checks()
    if "--select-confirm" in sys.argv:
        select_development()
    if "--freeze" in sys.argv:
        freeze()
    if "--final" in sys.argv:
        final_runs()
    if "--extended" in sys.argv:
        extended_runs()
    if "--figures" in sys.argv:
        figures()


if __name__ == "__main__":
    main()
