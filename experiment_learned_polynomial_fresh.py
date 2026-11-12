"""Frozen new-date/spatial-split confirmation; check, blind pilot, then full."""

import csv
import json
from pathlib import Path
import pickle
import resource
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import gpytorch
from linear_operator.operators import LinearOperator
import numpy as np
import pandas as pd
import torch
import experiment_grf_heldout as held
from experiment_learned_polynomial_traffic import model_for, polynomial_kernel, check_kernel
from experiment_grf_traffic import CORE, CORE_COMMIT, UPSTREAM, EXPERIMENT_COMMIT, posterior, score, train_model

OUT = held.OUT.parent / "learned_polynomial_fresh"
INDICES = [2, 6, 10, 14, 18]
DATES = [f"2017-{date} 17:30:00" for date in ("01-16", "02-13", "03-13", "04-10", "05-08")]
ARMS = [("Heat", 0), ("Poly", 0), ("C", 303), ("C", 404), ("General", 303), ("General", 404)]


def prepare(stage):
    old = held.prepare()
    original = json.loads((held.OUT/"manifest.json").read_text())
    assert original["data_sha"] == "802e5e523b5db6ff9d13da3388c566de7d0212b09b23497a3d0e50a0385ad71e"
    folder = UPSTREAM/"data/traffic/PEMS"
    assert held.sha((folder/"pems-bay.h5").read_bytes()) == original["hdf_sha"]
    with (folder/"adj_mx_bay.pkl").open("rb") as handle:
        sensor_ids = np.asarray(pickle.load(handle, encoding="latin1")[0], dtype=np.int64)
    np.testing.assert_array_equal(sensor_ids, old["sensor_ids"])
    vertices = old["vertices"]
    assert held.sha(np.stack([sensor_ids, vertices]).tobytes()) == original["sensor_map_sha"]
    assert len(set(sensor_ids)) == len(set(vertices)) == 325
    signals = pd.read_hdf(folder/"pems-bay.h5")
    signals = signals[(signals.index.weekday == 0) & (signals.index.hour == 17) & (signals.index.minute == 30)]
    assert signals.index.is_monotonic_increasing and signals.index.is_unique and len(signals) > max(INDICES)
    assert signals.index[INDICES].astype(str).tolist() == DATES, "Missing frozen date; no replacement"
    assert not set(INDICES).intersection([0, *old["indices"].tolist()])
    raw = signals.iloc[INDICES].loc[:, sensor_ids].to_numpy()
    assert raw.shape == (5, 325) and np.isfinite(raw).all()
    order = np.random.default_rng(20261112).permutation(325)
    train, test = vertices[order[:250]], vertices[order[250:]]
    assert not np.intersect1d(train, test).size and set(train) != set(old["train"])
    mean, std = raw[:, order[:250]].mean(1), raw[:, order[:250]].std(1)
    assert (std > 0).all()
    data = dict(laplacian=old["laplacian"], sensor_ids=sensor_ids, vertices=vertices, train=train, test=test,
        indices=np.array(INDICES), timestamps=np.array(DATES), raw=raw, mean=mean, std=std,
        y=(raw[:, order[:250]]-mean[:, None])/std[:, None], yt=(raw[:, order[250:]]-mean[:, None])/std[:, None])
    np.testing.assert_allclose(data["y"].mean(1), 0, atol=1e-13)
    np.testing.assert_allclose(data["y"].std(1), 1, atol=1e-13)
    path, manifest_path = OUT/"data.npz", OUT/"manifest.json"
    assert path.exists() == manifest_path.exists(), "Incomplete frozen data; manual review required"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    assert previous is not None or stage == "check", "Freeze/check data before fitting"
    if previous is not None:
        assert held.sha(path.read_bytes()) == previous["data_sha"]
        with np.load(path) as saved:
            for key, value in data.items():
                np.testing.assert_array_equal(value, saved[key])
    else:
        np.savez(path, **data)
    matrices, walk_seconds = {}, {}
    for seed in (303, 404):
        walk_path = OUT/f"walks_C_{seed}.pt"
        if previous is not None:
            assert held.sha(walk_path.read_bytes()) == previous["walk_sha"][str(seed)]
        with patch.object(held, "OUT", OUT):
            matrices[seed], walk_seconds[seed] = held.walks("C", seed, torch.from_numpy(data["laplacian"]))
    sources = [Path(__file__).name, "learned_polynomial.py", "chebyshev.py", "graph_gp.py", "dense_gp.py",
        "experiment_learned_polynomial_traffic.py", "experiment_grf_heldout.py", "experiment_grf_traffic.py",
        "experiment_grf_frontier.py", "tests/test_learned_polynomial.py"]
    manifest = dict(original=original, indices=INDICES, timestamps=DATES, split_seed=20261112,
        train_ids=train.tolist(), test_ids=test.tolist(), train_sensor_ids=sensor_ids[order[:250]].tolist(),
        test_sensor_ids=sensor_ids[order[250:]].tolist(), mean=mean.tolist(), std=std.tolist(),
        data_sha=held.sha(path.read_bytes()), targets_sha=held.sha(raw.tobytes()),
        walk_sha={str(s):held.sha((OUT/f"walks_C_{s}.pt").read_bytes()) for s in matrices}, walk_s=walk_seconds,
        source_sha={s:held.sha(Path(s).read_bytes()) for s in sources}, torch=torch.__version__,
        gpytorch=gpytorch.__version__, numpy=np.__version__, base="e6e4a795aefc06c88fb8ca395fa73079fa095b61",
        protocol=dict(arms=ARMS, steps=1000, pilot_steps=20, optimizer="AdamW", lr=.05, weight_decay=.01,
            dtype="float64", device="cpu", threads=4, workers=4, J=3, halt=.5, W=4096, seeds=[303,404],
            rng=0, initialization="Unchanged PR12 model_for", cholesky=800, heat_degree=20, filter_degree=2,
            scoring="After all fits: exact own-kernel posterior, RMSE mph, marginal observation NLPD; noise once"))
    # JSON turns arm tuples into lists; compare the persisted representation.
    manifest = json.loads(json.dumps(manifest))
    if previous is not None:
        assert previous == manifest, "Frozen inputs/protocol/source changed"
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print("FROZEN", DATES, "data", manifest["data_sha"], flush=True)
    return data, matrices, held.sha(manifest_path.read_bytes())


def fit(data, matrices, manifest_sha, date, arm, seed, steps):
    tag = f"{INDICES[date]}_{arm}_{seed}_{steps}"
    path, checkpoint = OUT/f"{tag}.json", OUT/f"{tag}.pt"
    if path.exists():
        row = json.loads(path.read_text())
        assert row["manifest_sha"] == manifest_sha and row["steps"] == steps
        assert row["status"] in ("fitted", "complete", "failed", "score_failed"), "Interrupted fit; no automatic refit"
        if row["status"] in ("fitted", "complete"):
            assert held.sha(checkpoint.read_bytes()) == row["checkpoint_sha"]
            saved = torch.load(checkpoint, weights_only=True)
            assert len(saved["losses"]) == steps and np.isfinite(saved["losses"]).all()
        print("SKIP verified", tag, row["status"], flush=True)
        return row
    assert not checkpoint.exists(), "Orphan checkpoint; no automatic refit"
    row = dict(tag=tag, timestamp=DATES[date], arm=arm, seed=seed, steps=steps, manifest_sha=manifest_sha, status="running")
    path.write_text(json.dumps(row, indent=2))
    try:
        graph, train, y = torch.from_numpy(data["laplacian"]), torch.from_numpy(data["train"]), torch.from_numpy(data["y"][date])
        torch.manual_seed(0)
        clock = perf_counter()
        model = model_for("Cheb" if arm == "Heat" else arm, graph, matrices.get(seed), train, y)
        row["setup_s"] = perf_counter()-clock
        row["initial"] = {name:p.detach().tolist() for name,p in model.named_parameters()}
        clock = perf_counter()
        losses = train_model(model, model.likelihood, train, y, lr=.05, max_iter=steps, print_every=250, progress_desc=tag)
        row["train_s"] = perf_counter()-clock
        torch.save(dict(model=model.state_dict(), losses=losses), checkpoint.with_suffix(".tmp"))
        checkpoint.with_suffix(".tmp").replace(checkpoint)
        row["checkpoint_sha"] = held.sha(checkpoint.read_bytes())
        assert len(losses) == steps and np.isfinite(losses).all()
        model.zero_grad()
        nll = -gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)(model(train), y)
        assert torch.isfinite(nll)
        nll.backward()
        assert all(torch.isfinite(p).all() and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        row.update(nll=float(nll.detach()), gradient_norm=float(torch.cat([p.grad.flatten() for p in model.parameters()]).norm()),
            parameters={name:p.detach().tolist() for name,p in model.named_parameters()}, noise=float(model.likelihood.noise.detach()),
            peak_parent_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        kernel = model.covar_module
        if arm == "Poly":
            row.update(a1=float(kernel.base_kernel.a1.detach()), a2=float(kernel.base_kernel.a2.detach()))
        if arm in ("Heat", "C"):
            row["beta"] = float((kernel.base_kernel.heat_time if arm == "Heat" else kernel.beta).detach())
        row["status"] = "fitted"
    except Exception as error:
        row.update(status="failed", error=repr(error))
    finally:
        path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    print(row["status"].upper(), tag, row.get("error", ""), flush=True)
    return row


def evaluate(row, data, matrices, date):
    if row["status"] != "fitted":
        return
    try:
        train, test = torch.from_numpy(data["train"]), torch.from_numpy(data["test"])
        y = torch.from_numpy(data["y"][date])
        torch.manual_seed(0)
        model = model_for("Cheb" if row["arm"] == "Heat" else row["arm"], torch.from_numpy(data["laplacian"]), matrices.get(row["seed"]), train, y)
        checkpoint = OUT/f"{row['tag']}.pt"
        assert held.sha(checkpoint.read_bytes()) == row["checkpoint_sha"]
        model.load_state_dict(torch.load(checkpoint, weights_only=True)["model"])
        model.eval()
        clock = perf_counter()
        with torch.no_grad():
            noise = model.likelihood.noise
            covariance = model.covar_module(torch.cat([train, test])).to_dense()
            nll, mean, variance = posterior(covariance, y, noise)
            np.testing.assert_allclose(float(nll), row["nll"], atol=1e-10, rtol=1e-10)
            rmse, nlpd = score(mean, variance, torch.from_numpy(data["yt"][date]), noise, float(data["std"][date]))
        row.update(status="complete", rmse=rmse, nlpd=nlpd, oracle_s=perf_counter()-clock)
        torch.save(dict(mean=mean, latent_variance=variance), OUT/f"{row['tag']}_posterior.pt")
    except Exception as error:
        row.update(status="score_failed", error=repr(error))
    (OUT/f"{row['tag']}.json").write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    print("SCORED", row["tag"], row["status"], flush=True)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv)>1 else "check"
    assert stage in ("check", "pilot", "full")
    for source, revision in ((CORE, CORE_COMMIT), (UPSTREAM, EXPERIMENT_COMMIT)):
        assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == revision
        subprocess.run(["git", "-C", str(source), "diff", "--exit-code"], check=True)
    data, matrices, manifest_sha = prepare(stage)
    original_dense = LinearOperator.to_dense

    def selected_dense(operator, *args, **kwargs):
        assert operator.shape != (1016, 1016), "Full graph kernel/identity forbidden"
        return original_dense(operator, *args, **kwargs)

    with gpytorch.settings.max_cholesky_size(800), gpytorch.settings.max_eager_kernel_size(0), patch.object(LinearOperator, "to_dense", selected_dense):
        if stage == "check":
            ids = [0, 1, 2, 5, 3, 4]
            diagnostics = check_kernel(polynomial_kernel(torch.from_numpy(data["laplacian"])[ids][:, ids]))
            (OUT/"checks.json").write_text(json.dumps(dict(manifest_sha=manifest_sha, diagnostics=diagnostics), indent=2))
            print("PASS numerical gates", diagnostics, flush=True)
        else:
            assert json.loads((OUT/"checks.json").read_text())["manifest_sha"] == manifest_sha
            if stage == "full":
                pilot = json.loads((OUT/"2_Poly_0_20.json").read_text())
                assert pilot["status"] == "fitted" and pilot["manifest_sha"] == manifest_sha
            cases = [(0, "Poly", 0)] if stage == "pilot" else [(d,a,s) for d in range(5) for a,s in ARMS]
            rows = [fit(data, matrices, manifest_sha, d, a, s, 20 if stage == "pilot" else 1000) for d,a,s in cases]
            if stage == "pilot":
                assert rows[0]["status"] == "fitted", "Pilot failed; stop before full"
            else:
                assert all(r["status"] in ("fitted", "complete", "failed", "score_failed") for r in rows)
                print("ALL 30 FIT ATTEMPTS FINISHED; beginning test scoring", flush=True)
                for (date, _, _), row in zip(cases, rows):
                    evaluate(row, data, matrices, date)
                with (OUT/"rows.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["timestamp","arm","seed","status","rmse","nlpd","nll","noise","setup_s","train_s","oracle_s","error"], extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
