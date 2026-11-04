"""Frozen PEMS-BAY approximation frontier; run --pilot before the full command."""

import csv
import hashlib
import platform
import subprocess
import sys
from time import perf_counter

import gpytorch
import numpy as np
import torch

from experiment_grf_traffic import (
    CORE, CORE_COMMIT, EXPERIMENT_COMMIT, OUTPUT, UPSTREAM,
    DiffusionGRFKernel, GRFGP, GRFSampler, HeatGP, SparseLinearOperator,
    TrafficExactKernel, chebyshev_kernel, load_data, numerical_checks, plt,
    posterior, score,
)

# J counts recorded powers, including the identity. This grid is not selected by scores.
GRID = {
    "A": (3, .5, 256), "B": (3, .5, 1024), "C": (3, .5, 4096),
    "D": (10, .5, 256), "E": (10, .5, 1024), "F": (10, .1, 256),
    "G": (10, .1, 1024), "H": (10, .1, 4096),
}
TABLE = OUTPUT / "frontier.csv"
SPLIT_HASHES = {
    0: "0863757154dee83b6262fcd35dacb089fe44a4a7416bd1018a911c505869bee5",
    2: "10589a2650367dfb8b65d2762c2661d75e87a1a0e10e5b488f4e6ec0633091cd",
}


class BoundedDiffusionGRFKernel(DiffusionGRFKernel):
    """Only change the upstream beta transform to match the traffic heat model."""

    @property
    def beta(self):
        return 2 * torch.sigmoid(self.raw_beta)


def record(rows, row):
    """Replace one checkpoint atomically; completed cells survive interruption."""
    key = (row["kind"], row["seed"], row["arm"], row["budget"])
    rows[:] = [r for r in rows if (r["kind"], int(r["seed"]), r["arm"], float(r["budget"])) != key]
    rows.append(row)
    fields = list(dict.fromkeys(field for r in rows for field in r))
    temporary = TABLE.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(TABLE)


def sampled_walks(seed, arm, laplacian):
    j, halt, walks = GRID["G" if arm == "native" else arm]
    path = OUTPUT / f"frontier_walks_{seed}_{'G' if arm == 'native' else arm}.pt"
    if path.exists():
        saved = torch.load(path, weights_only=True)
    else:
        clock = perf_counter()
        sampler = GRFSampler(laplacian, walks, halt, j, seed=seed, use_tqdm=False, n_processes=4)
        matrices = sampler.sample_random_walk_matrices()
        saved = dict(seed=seed, J=j, p_halt=halt, W=walks, effective_seed=sampler.seed,
                     workers=4, walk_s=perf_counter()-clock,
                     matrices=[m.sparse_csr_tensor for m in matrices])
        torch.save(saved, path)
    assert (saved["seed"], saved["J"], saved["p_halt"], saved["W"], saved["workers"]) == (seed, j, halt, walks, 4)
    assert saved["effective_seed"] == (seed or 42)
    matrices = [SparseLinearOperator(m) for m in saved["matrices"]]
    assert len(matrices) == j and all(m.shape == laplacian.shape for m in matrices)
    torch.testing.assert_close(matrices[0].to_dense(), torch.eye(len(laplacian)))
    metadata = {k: v for k, v in saved.items() if k not in ["matrices", "seed"]}
    print(f"WALKS {arm}: {metadata}", flush=True)
    return matrices, metadata


def references(laplacian, ids):
    """Dense off-timer reference only; never used by either training path."""
    result = {}
    with torch.no_grad():
        for beta in [1.0, 1.9]:
            heat = torch.matrix_exp(-beta * laplacian)[ids][:, ids]
            term = torch.eye(len(laplacian))
            q = term.clone()
            for power in range(1, 10):
                term = (-beta / (2 * power)) * (term @ laplacian)
                q = q + term
                if power in [2, 9]:
                    if power == 2:
                        closed = torch.eye(len(laplacian)) - beta/2*laplacian + beta**2/8*(laplacian@laplacian)
                        torch.testing.assert_close(q, closed, atol=2e-15, rtol=2e-14)
                    selected = q[ids]
                    result[beta, power+1] = (heat, selected @ selected.T)
            short = (result[beta, 3][1]-heat).norm()
            long = (result[beta, 10][1]-heat).norm()
            assert long < short
            cheb = chebyshev_kernel(laplacian)
            cheb.base_kernel.initialize(raw_heat_time=torch.logit(torch.tensor(beta/2)))
            cheb.outputscale = 1.0
            candidate = cheb(ids).to_dense()
            torch.testing.assert_close(candidate, heat, atol=3e-12, rtol=3e-11)
            print(f"REFERENCE beta={beta}: Cheb relative error={(candidate-heat).norm()/heat.norm():.3e}; "
                  f"truncation J3/J10={short/heat.norm():.6g}/{long/heat.norm():.6g}", flush=True)
    return result


def approximation(rows, seed, arm, matrices, metadata, ids, reference):
    kernel = BoundedDiffusionGRFKernel(matrices, metadata["J"])
    with torch.no_grad():
        for beta in [1.0, 1.9]:
            kernel.initialize(raw_beta=torch.logit(torch.tensor(beta/2)), raw_sigma_f=torch.log(torch.expm1(torch.tensor(1.0))))
            heat, truncated = reference[beta, metadata["J"]]
            sampled = kernel(ids).to_dense()
            row = dict(kind="approx", seed=seed, arm=arm, budget=beta, status="complete", **metadata)
            row.update(truncation=float((truncated-heat).norm()/heat.norm()),
                       sampling=float((sampled-truncated).norm()/truncated.norm()),
                       overall=float((sampled-heat).norm()/heat.norm()),
                       diagonal_max_abs=float((sampled-heat).diagonal().abs().max()),
                       sampling_diagonal_max_abs=float((sampled-truncated).diagonal().abs().max()),
                       kernel_max_abs=float(sampled.abs().max()))
            assert np.isfinite([row[k] for k in ["truncation", "sampling", "overall", "diagonal_max_abs"]]).all()
            record(rows, row)
            print(row, flush=True)


def evaluate(model, row, ids, y, yt, scale):
    model.eval()
    kernel = model.covar_module
    cheb = isinstance(model, HeatGP)
    with torch.no_grad():
        clock = perf_counter()
        covariance = kernel(ids).to_dense()
        nll, mean, variance = posterior(covariance, y, model.likelihood.noise)
        oracle_seconds = perf_counter()-clock
        noise = model.likelihood.noise
        row.update(beta=float(kernel.base_kernel.heat_time if cheb else kernel.beta),
                   sigma_f=float(kernel.outputscale.sqrt() if cheb else kernel.sigma_f),
                   noise=float(noise), kernel_nll=float(nll), oracle_s=oracle_seconds,
                   latent_var_min=float(variance.min()), latent_var_mean=float(variance.mean()),
                   latent_var_max=float(variance.max()), observation_var_mean=float((variance+noise).mean()))
        row["oracle_rmse"], row["oracle_nlpd"] = score(mean, variance, yt, noise, scale)
        row["oracle_nlpd_mph"] = row["oracle_nlpd"] + np.log(scale)
        if cheb:
            row.update(pred_s=oracle_seconds, deployed_rmse=row["oracle_rmse"], deployed_nlpd=row["oracle_nlpd"])
        else:
            for sample_seed in [100, 101]:
                torch.manual_seed(sample_seed)
                clock = perf_counter()
                with gpytorch.settings.cg_tolerance(1e-4):
                    samples = model.predict_sample(ids[y.numel():], n_samples=512)
                elapsed = perf_counter()-clock
                sample_mean, sample_variance = samples.mean(0), samples.var(0, unbiased=True)
                rmse, nlpd = score(sample_mean, sample_variance, yt, noise, scale)
                row.update({f"mc{sample_seed}_s": elapsed, f"mc{sample_seed}_rmse": rmse,
                            f"mc{sample_seed}_nlpd": nlpd,
                            f"mc{sample_seed}_mean_error": float((sample_mean-mean).square().mean().sqrt()),
                            f"mc{sample_seed}_variance_error": float((sample_variance-variance).square().mean().sqrt())})
            # Seed 100 is the declared deployed run; seed 101 is a separate replication.
            row.update(pred_s=row["mc100_s"], deployed_rmse=row["mc100_rmse"], deployed_nlpd=row["mc100_nlpd"])
        row["deployed_nlpd_mph"] = row["deployed_nlpd"] + np.log(scale)
        row["total_s"] = row["walk_s"] + row["setup_s"] + row["train_s"] + row["pred_s"]
        assert np.isfinite([row[k] for k in ["kernel_nll", "oracle_rmse", "oracle_nlpd", "deployed_rmse", "deployed_nlpd"]]).all()
    row["status"] = "complete"
    print(row, flush=True)


def fit(rows, seed, arm, matrices, metadata, data, limit):
    path = OUTPUT / f"frontier_fit_{seed}_{arm}.pt"
    completed = {int(r["budget"]) for r in rows if r["kind"] == "fit" and int(r["seed"]) == seed
                 and r["arm"] == arm and r["status"] == "complete"}
    if limit in completed:
        print(f"SKIP complete {seed}/{arm}/{limit}", flush=True)
        return
    train, test = torch.from_numpy(data["train_ids"]), torch.from_numpy(data["test_ids"])
    y, yt = torch.from_numpy(data["y_train"]), torch.from_numpy(data["y_test"])
    ids = torch.cat([train, test])
    clock = perf_counter()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if arm == "Cheb":
        kernel = chebyshev_kernel(torch.from_numpy(data["laplacian"]))
        model = HeatGP(train, y, likelihood, kernel)
    else:
        cls = DiffusionGRFKernel if arm == "native" else BoundedDiffusionGRFKernel
        kernel = cls(matrices, metadata["J"])
        model = GRFGP(train, y, likelihood, kernel)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    setup_seconds = perf_counter()-clock
    expected = TrafficExactKernel(torch.from_numpy(data["laplacian"]))
    if arm != "native":
        beta = kernel.base_kernel.heat_time if arm == "Cheb" else kernel.beta
        amplitude = kernel.outputscale.sqrt() if arm == "Cheb" else kernel.sigma_f
        torch.testing.assert_close(beta, expected.beta)
        torch.testing.assert_close(amplitude, expected.sigma_f)
        raw = kernel.base_kernel.raw_heat_time if arm == "Cheb" else kernel.raw_beta
        raw_amplitude = kernel.raw_outputscale if arm == "Cheb" else kernel.raw_sigma_f
        assert float(raw.detach()) == float(raw_amplitude.detach()) == 1.0
        torch.testing.assert_close(likelihood.raw_noise, torch.zeros_like(likelihood.raw_noise))
        print(f"START parameters {seed}/{arm}: raw beta/sigma/noise=1/1/0; "
              f"physical beta/sigma/noise={float(beta.detach()):.9f}/"
              f"{float(amplitude.detach()):.9f}/{float(likelihood.noise.detach()):.9f}", flush=True)
    step, train_seconds, losses = 0, 0.0, []
    if path.exists():
        saved = torch.load(path, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step, train_seconds, setup_seconds, losses = saved["step"], saved["train_s"], saved["setup_s"], saved["losses"]
    print(f"FIT {seed}/{arm}: resume update {step}, limit {limit}; AdamW {optimizer.defaults}", flush=True)
    pending = dict(kind="fit", seed=seed, arm=arm, budget=1000, status="incomplete", actual_steps=step,
                   setup_s=setup_seconds, train_s=train_seconds, **metadata)
    record(rows, pending)
    try:
        for endpoint in [100, 1000]:
            if endpoint > limit or endpoint in completed:
                continue
            assert step <= endpoint, "A later state cannot stand in for a missing earlier checkpoint"
            model.train()
            clock = perf_counter()
            # Tiny checkpointable equivalent of official experiments.utils.train_model.
            while step < endpoint:
                optimizer.zero_grad()
                loss = -mll(model(train), y)
                assert torch.isfinite(loss)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                step += 1
            train_seconds += perf_counter()-clock
            saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                         train_s=train_seconds, setup_s=setup_seconds, losses=losses)
            temporary = path.with_suffix(".tmp")
            torch.save(saved, temporary)
            temporary.replace(path)
            # Preserve both model states even after the resumable fit advances.
            torch.save(saved, OUTPUT / f"frontier_fit_{seed}_{arm}_{endpoint}.pt")
            row = dict(kind="fit", seed=seed, arm=arm, budget=endpoint, actual_steps=step,
                       setup_s=setup_seconds, train_s=train_seconds, **metadata)
            evaluate(model, row, ids, y, yt, float(data["train_std"]))
            record(rows, row)
            if endpoint == 100:
                pending.update(actual_steps=step, train_s=train_seconds)
                record(rows, pending)
    except (RuntimeError, MemoryError, AssertionError) as error:
        pending.update(status="failed", actual_steps=step, error=f"{type(error).__name__}: {error}")
        record(rows, pending)
        raise


def figure(rows):
    complete = [r for r in rows if r["kind"] == "fit" and r["status"] == "complete"]
    for row in complete:
        baseline = next((r for r in complete if r["arm"] == "Cheb" and
                         int(r["seed"]) == int(row["seed"]) and int(r["budget"]) == int(row["budget"])), None)
        if baseline is not None:
            for path in ["oracle", "deployed"]:
                row[f"{path}_matched"] = (float(row[f"{path}_rmse"]) <= 1.02*float(baseline[f"{path}_rmse"]) and
                                          float(row[f"{path}_nlpd"]) <= float(baseline[f"{path}_nlpd"])+.02)
            row["total_ratio_to_cheb"] = float(row["total_s"])/float(baseline["total_s"])
            record(rows, row)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = set()
    for row in complete:
        arm = row["arm"]
        group = arm if arm in ["Cheb", "native"] else f"GRF J={int(row['J'])}"
        color = {"Cheb": "black", "native": "C3", "GRF J=3": "C1", "GRF J=10": "C0"}[group]
        marker = "o" if int(row["seed"]) == 0 else "^"
        face = color if int(row["budget"]) == 1000 else "none"
        for axis, metric in zip(axes, ["oracle_nlpd", "oracle_rmse"]):
            axis.scatter(float(row["total_s"]), float(row[metric]), color=color, facecolors=face, marker=marker,
                         label=group if group not in labels else None)
        labels.add(group)
    for axis, ylabel in zip(axes, ["Marginal observation NLPD (normalized), oracle", "RMSE (mph), oracle"]):
        axis.set(xscale="log", xlabel="Method total seconds (one deployed prediction)", ylabel=ylabel)
        axis.legend(fontsize=8)
    fig.suptitle("Seed 0: circles; seed 2: triangles. Hollow: 100 updates; filled: 1000.", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT / "grf_frontier.png", dpi=160)
    plt.close(fig)


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    print(f"{platform.platform()}, {platform.machine()}, Torch {torch.__version__}, GPyTorch {gpytorch.__version__}; CPU float64, four threads", flush=True)
    print("Prediction CG=1e-4; native Cholesky training threshold=800; MC seeds100/101, 512 draws each. Practical tolerance: RMSE<=1.02*Cheb AND NLPD<=Cheb+.02.")
    for path, commit in [(CORE, CORE_COMMIT), (UPSTREAM, EXPERIMENT_COMMIT)]:
        assert subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() == commit
    rows = list(csv.DictReader(TABLE.open())) if TABLE.exists() else []
    if "--figure-only" in sys.argv:
        figure(rows)
        return
    pilot = "--pilot" in sys.argv
    for seed in ([0] if pilot else [0, 2]):
        assert (OUTPUT / f"seed{seed}.npz").exists(), "PR4 frozen snapshot required; never regenerate silently"
        data = load_data(seed)
        digest = hashlib.sha256()
        for key in ["train_ids", "test_ids", "y_train", "y_test", "train_mean", "train_std"]:
            digest.update(data[key].tobytes())
        assert digest.hexdigest() == SPLIT_HASHES[seed]
        assert str(data["adjacency_hash"]) == "96026f51a092dac8461388374081628aa1e85349a33214dbcd51cb88ae156a64"
        assert str(data["laplacian_hash"]) == "64661f4a768224ebcf799bfee713dce1402b36febec7d6236b94625c3548671e"
        laplacian = torch.from_numpy(data["laplacian"])
        ids = torch.from_numpy(np.concatenate([data["train_ids"], data["test_ids"]]))
        reference = references(laplacian, ids)
        arms = ["A", "G"] if pilot else ["Cheb", "A", "G", "B", "C", "D", "E", "F", "H"]
        if not pilot and seed == 0:
            arms.append("native")
        # Both pilot samplers and mathematical checks precede any pilot fitting.
        sampled = {}
        if pilot:
            for arm in arms:
                sampled[arm] = sampled_walks(seed, arm, laplacian)
                approximation(rows, seed, arm, *sampled[arm], ids, reference)
            numerical_checks(laplacian, ids[:250], ids[250:], torch.from_numpy(data["y_train"]), sampled["A"][0])
        for arm in arms:
            if arm == "Cheb":
                matrices, metadata = None, dict(walk_s=0.0, J=20, p_halt=0.0, W=0, effective_seed=seed, workers=0)
            else:
                matrices, metadata = sampled[arm] if arm in sampled else sampled_walks(seed, arm, laplacian)
                if not pilot and arm != "native":
                    approximation(rows, seed, arm, matrices, metadata, ids, reference)
            fit(rows, seed, arm, matrices, metadata, data, 100 if pilot else 1000)
    figure(rows)


if __name__ == "__main__":
    main()
