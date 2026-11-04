"""Frozen new-Monday replication: --pilot first, then resume the full 15 fits."""

import csv
import hashlib
import json
import pickle
import subprocess
import sys
from time import perf_counter

import gpytorch
import networkx as nx
import numpy as np
import pandas as pd
import torch

from experiment_grf_frontier import GRID, BoundedDiffusionGRFKernel, evaluate
from experiment_grf_traffic import (
    CORE, CORE_COMMIT, EXPERIMENT_COMMIT, OUTPUT, UPSTREAM, GRFGP, GRFSampler,
    HeatGP, SparseLinearOperator, chebyshev_kernel, get_normalized_laplacian,
    load_data, load_PEMS, plt, score, train_model,
)

OUT = OUTPUT / "heldout"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare():
    """Recover the sensor map once; upstream load_PEMS itself keeps only week zero."""
    assert (OUTPUT / "seed0.npz").exists()
    base = load_data(0)
    assert str(base["adjacency_hash"]) == "96026f51a092dac8461388374081628aa1e85349a33214dbcd51cb88ae156a64"
    assert str(base["laplacian_hash"]) == "64661f4a768224ebcf799bfee713dce1402b36febec7d6236b94625c3548671e"
    path, manifest_path = OUT / "data.npz", OUT / "manifest.json"
    if not path.exists():
        folder = UPSTREAM / "data/traffic/PEMS"
        assert (folder / "bay_graph.pkl").is_file(), "Cached roads required; no OSM download"
        graph, _, _, (vertices, first_y) = load_PEMS(path=str(folder.parent))
        assert len(graph) == 1016 and graph.number_of_edges() == 1173
        assert list(graph.nodes) == list(range(1016))
        adjacency = nx.to_numpy_array(graph)
        assert sha(adjacency.tobytes()) == str(base["adjacency_hash"])
        assert sha(get_normalized_laplacian(adjacency).tobytes()) == str(base["laplacian_hash"])
        with (folder / "adj_mx_bay.pkl").open("rb") as handle:
            sensor_ids = np.asarray(pickle.load(handle, encoding="latin1")[0], dtype=np.int64)
        signals = pd.read_hdf(folder / "pems-bay.h5")
        signals = signals[(signals.index.weekday == 0) & (signals.index.hour == 17) & (signals.index.minute == 30)]
        assert len(signals) > 9 and signals.index.is_monotonic_increasing
        # Explicit label selection: HDF column order is not graph-vertex order.
        ordered = signals.loc[:, sensor_ids].to_numpy()
        np.testing.assert_array_equal(ordered[0], first_y.ravel())
        vertices = vertices.astype(np.int64).ravel()
        assert len(set(sensor_ids)) == len(set(vertices)) == 325
        order = np.random.default_rng(20261104).permutation(325)
        train, test = vertices[order[:250]], vertices[order[250:]]
        assert not np.intersect1d(train, test).size and set(train) != set(base["train_ids"])
        raw = ordered[[1, 5, 9]]
        assert np.isfinite(raw).all(), "Invalid frozen date; stop rather than replace it"
        mean, std = raw[:, order[:250]].mean(1), raw[:, order[:250]].std(1)
        assert (std > 0).all()
        np.savez(path, laplacian=base["laplacian"], sensor_ids=sensor_ids, vertices=vertices,
                 train=train, test=test, indices=[1, 5, 9], timestamps=signals.index[[1, 5, 9]].astype(str).to_numpy(dtype="U"),
                 raw=raw, mean=mean, std=std, y=(raw[:, order[:250]]-mean[:, None])/std[:, None],
                 yt=(raw[:, order[250:]]-mean[:, None])/std[:, None])
        manifest = dict(indices=[1, 5, 9], timestamps=signals.index[[1, 5, 9]].astype(str).tolist(),
                        train_ids=train.tolist(), test_ids=test.tolist(), split_seed=20261104,
                        mean=mean.tolist(), std=std.tolist(), index0_alignment_max_error=0.0,
                        adjacency_sha=str(base["adjacency_hash"]), laplacian_sha=str(base["laplacian_hash"]),
                        sensor_map_sha=sha(np.stack([sensor_ids, vertices]).tobytes()), targets_sha=sha(raw.tobytes()),
                        hdf_sha=sha((folder / "pems-bay.h5").read_bytes()), data_sha=sha(path.read_bytes()),
                        experiment_commit=EXPERIMENT_COMMIT, core_commit=CORE_COMMIT)
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    manifest = json.loads(manifest_path.read_text())
    assert sha(path.read_bytes()) == manifest["data_sha"]
    with np.load(path) as archive:
        data = dict(archive)
    assert sha(data["laplacian"].tobytes()) == str(base["laplacian_hash"])
    print(f"DATA validated: {manifest}", flush=True)
    return data


def walks(arm, seed, laplacian):
    path = OUT / f"walks_{arm}_{seed}.pt"
    j, halt, count = GRID[arm]
    if not path.exists():
        clock = perf_counter()
        sampler = GRFSampler(laplacian, count, halt, j, seed=seed, use_tqdm=False, n_processes=4)
        matrices = sampler.sample_random_walk_matrices()
        torch.save(dict(matrices=[m.sparse_csr_tensor for m in matrices], walk_s=perf_counter()-clock,
                        J=j, halt=halt, W=count, seed=sampler.seed, workers=4), path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
    saved = torch.load(path, weights_only=True)
    assert (saved["J"], saved["halt"], saved["W"], saved["seed"], saved["workers"]) == (j, halt, count, seed, 4)
    print(f"WALKS {arm}/{seed}: {saved['walk_s']:.6f} seconds, J={j}, halt={halt}, W={count}, workers=4", flush=True)
    return [SparseLinearOperator(m) for m in saved["matrices"]], saved["walk_s"]


def save_rows(rows):
    path = OUT / "rows.csv"
    with path.with_suffix(".tmp").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".tmp").replace(path)


def figure(rows):
    for row in rows:
        if row["status"] != "complete":
            continue
        baseline = next(r for r in rows if r["arm"] == "Cheb" and r["timestamp"] == row["timestamp"])
        row["rmse_delta"] = float(row["oracle_rmse"])-float(baseline["oracle_rmse"])
        row["nlpd_delta"] = float(row["oracle_nlpd"])-float(baseline["oracle_nlpd"])
        row["rmse_gate_margin"] = .02*float(baseline["oracle_rmse"])-row["rmse_delta"]
        row["nlpd_gate_margin"] = .02-row["nlpd_delta"]
        row["qualified"] = row["rmse_gate_margin"] >= 0 and row["nlpd_gate_margin"] >= 0
    save_rows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm, color in [("C", "C1"), ("H", "C0")]:
        for seed, marker in [(101, "o"), (202, "^")]:
            chosen = [r for r in rows if r["arm"] == arm and int(r["feature_seed"]) == seed and r["status"] == "complete"]
            for axis, metric in zip(axes, ["rmse_delta", "nlpd_delta"]):
                axis.scatter([r["timestamp"][:10] for r in chosen], [float(r[metric]) for r in chosen],
                             marker=marker, color=color, label=f"{arm}, walks seed {seed}")
    for axis, ylabel in zip(axes, ["Oracle RMSE difference from Cheb (mph)", "Oracle observation NLPD difference from Cheb"]):
        axis.axhline(0, color="black", linewidth=.8)
        axis.set(ylabel=ylabel)
        axis.tick_params(axis="x", rotation=20)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "heldout.png", dpi=160)
    plt.close(fig)


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    assert gpytorch.settings.max_cholesky_size.value() == 800
    for path, commit in [(CORE, CORE_COMMIT), (UPSTREAM, EXPERIMENT_COMMIT)]:
        assert subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() == commit
    OUT.mkdir(exist_ok=True)
    table = OUT / "rows.csv"
    rows = list(csv.DictReader(table.open())) if table.exists() else []
    if "--figure-only" in sys.argv:
        figure(rows)
        return
    data = prepare()  # All three dates and the graph/map are validated before fitting.
    laplacian = torch.from_numpy(data["laplacian"])
    train, test = torch.from_numpy(data["train"]), torch.from_numpy(data["test"])
    arms = [("Cheb", 0), ("C", 101), ("C", 202), ("H", 101), ("H", 202)]
    cases = [(date, arm, seed) for date in range(3) for arm, seed in arms]
    if "--pilot" in sys.argv:
        cases = cases[:2]
    sampled = {}
    for date, arm, seed in cases:
        existing = next((r for r in rows if int(r["index"]) == int(data["indices"][date]) and r["arm"] == arm and int(r["feature_seed"]) == seed), None)
        if existing is not None and existing["status"] == "complete":
            print(f"SKIP complete {data['indices'][date]}/{arm}/{seed}", flush=True)
            continue
        if arm != "Cheb" and (arm, seed) not in sampled:
            sampled[arm, seed] = walks(arm, seed, laplacian)
        y, yt = torch.from_numpy(data["y"][date]), torch.from_numpy(data["yt"][date])
        row = dict(index=int(data["indices"][date]), timestamp=str(data["timestamps"][date]), arm=arm,
                   feature_seed=seed, status="incomplete", walk_s=0.0 if arm == "Cheb" else sampled[arm, seed][1])
        if existing is not None:
            rows.remove(existing)
        rows.append(row)
        save_rows(rows)
        try:
            clock = perf_counter()
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            kernel = chebyshev_kernel(laplacian) if arm == "Cheb" else BoundedDiffusionGRFKernel(sampled[arm, seed][0], GRID[arm][0])
            model = HeatGP(train, y, likelihood, kernel) if arm == "Cheb" else GRFGP(train, y, likelihood, kernel)
            row["setup_s"] = perf_counter()-clock
            raw_beta = kernel.base_kernel.raw_heat_time if arm == "Cheb" else kernel.raw_beta
            raw_signal = kernel.raw_outputscale if arm == "Cheb" else kernel.raw_sigma_f
            assert float(raw_beta.detach()) == float(raw_signal.detach()) == 1.0 and float(likelihood.raw_noise.detach()) == 0.0
            checkpoint = OUT / f"fit_{row['index']}_{arm}_{seed}.pt"
            if checkpoint.exists():
                state = torch.load(checkpoint, weights_only=True)
                model.load_state_dict(state["model"])
                row.update(setup_s=state["setup_s"], train_s=state["train_s"])
            else:
                clock = perf_counter()
                losses = train_model(model, likelihood, train, y, lr=.05, max_iter=1000, print_every=250, progress_desc=f"{row['index']}/{arm}/{seed}")
                row["train_s"] = perf_counter()-clock
                torch.save(dict(model=model.state_dict(), losses=losses, train_s=row["train_s"], setup_s=row["setup_s"]), checkpoint.with_suffix(".tmp"))
                checkpoint.with_suffix(".tmp").replace(checkpoint)
            evaluate(model, row, torch.cat([train, test]), y, yt, float(data["std"][date]))
            if arm == "Cheb":
                # Cold ordinary GP prediction: observed Cholesky, no dense joint audit charged.
                clock = perf_counter()
                with torch.no_grad(), gpytorch.settings.max_eager_kernel_size(0):
                    prediction = model(test)
                    mean, variance = prediction.mean, prediction.variance
                row["pred_s"] = perf_counter()-clock
                rmse, nlpd = score(mean, variance, yt, likelihood.noise.detach(), float(data["std"][date]))
                np.testing.assert_allclose([rmse, nlpd], [row["oracle_rmse"], row["oracle_nlpd"]], atol=2e-11, rtol=2e-11)
                row["total_s"] = row["setup_s"]+row["train_s"]+row["pred_s"]
            row["amortized_s"] = row["total_s"] - 2*row["walk_s"]/3
            save_rows(rows)
            print(f"COMPLETE {row}", flush=True)
        except (RuntimeError, MemoryError, AssertionError) as error:
            row.update(status="failed", error=str(error))
            save_rows(rows)
            raise
    figure(rows)


if __name__ == "__main__":
    main()
