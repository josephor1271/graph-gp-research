"""Post-hoc inner-200/50 AdamW study; outer test labels are never loaded."""

import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import gpytorch
from linear_operator.operators import LinearOperator
import numpy as np
from scipy.sparse import csr_matrix
import torch
import experiment_grf_heldout as held
from experiment_learned_polynomial_traffic import model_for, polynomial_kernel, check_kernel
from experiment_grf_traffic import CORE, CORE_COMMIT, UPSTREAM, EXPERIMENT_COMMIT, posterior, score

FRESH = held.OUT.parent / "learned_polynomial_fresh"
OUT = held.OUT.parent / "polynomial_optimizer"
POLICIES = {"A": (.05, 1000), "B": (.01, 1000), "C": (.05, 2000), "D": (.05, 2000)}
ARMS = [("Poly", 0), ("General", 303), ("General", 404)]


def prepare():
    prior = json.loads((FRESH/"manifest.json").read_text())
    assert held.sha((FRESH/"data.npz").read_bytes()) == prior["data_sha"] == "d14c1e5a45a094987f0c9409ca949bd90de27b6beac69d3a6a4eba187b457e91"
    with np.load(FRESH/"data.npz") as saved:
        # Deliberately avoid both 'yt' and 'raw': they contain outer test labels.
        graph, ids, outer = saved["laplacian"], saved["train"], saved["test"]
        normalized, old_mean, old_std = saved["y"][:3], saved["mean"][:3], saved["std"][:3]
        dates = saved["timestamps"][:3].tolist()
        assert saved["indices"][:3].tolist() == [2, 6, 10]
        assert held.sha(np.stack([saved["sensor_ids"], saved["vertices"]]).tobytes()) == prior["original"]["sensor_map_sha"]
    assert dates == [f"2017-{day} 17:30:00" for day in ("01-16", "02-13", "03-13")]
    assert ids.tolist() == prior["train_ids"] and len(set(ids)) == 250 and len(set(outer)) == 75
    assert not np.intersect1d(ids, outer).size and held.sha(graph.tobytes()) == prior["original"]["laplacian_sha"]
    raw = normalized*old_std[:, None]+old_mean[:, None]
    assert normalized.shape == (3, 250)
    np.testing.assert_allclose((raw-old_mean[:, None])/old_std[:, None], normalized, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(raw.mean(1), old_mean, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(raw.std(1), old_std, atol=1e-12, rtol=1e-12)
    mean, scale = raw[:, :200].mean(1), raw[:, :200].std(1)
    assert np.isfinite(raw).all() and (scale > 0).all() and not np.intersect1d(ids[:200], ids[200:]).size
    targets = (raw-mean[:, None])/scale[:, None]
    np.testing.assert_allclose(targets[:, :200].mean(1), 0, atol=1e-13)
    np.testing.assert_allclose(targets[:, :200].std(1), 1, atol=1e-13)
    matrices = {}
    for seed in (303, 404):
        assert held.sha((FRESH/f"walks_C_{seed}.pt").read_bytes()) == prior["walk_sha"][str(seed)]
        with patch.object(held, "OUT", FRESH):
            matrices[seed], _ = held.walks("C", seed, torch.from_numpy(graph))
    sources = [Path(__file__).name, *prior["source_sha"]]
    assert all(held.sha(Path(s).read_bytes()) == expected for s,expected in prior["source_sha"].items())
    manifest = dict(parent_manifest_sha=held.sha((FRESH/"manifest.json").read_bytes()), data_sha=prior["data_sha"],
        walk_sha=prior["walk_sha"], dates=dates, train_ids=ids[:200].tolist(), validation_ids=ids[200:].tolist(),
        mean=mean.tolist(), scale=scale.tolist(), targets_sha=held.sha(targets.tobytes()), policies=POLICIES,
        source_sha={s:held.sha(Path(s).read_bytes()) for s in sources}, torch=torch.__version__, gpytorch=gpytorch.__version__,
        numpy=np.__version__, packages=subprocess.check_output([sys.executable,"-m","pip","freeze"], text=True).splitlines(),
        protocol="CPU float64/4 threads; seed0; AdamW defaults, decay.01; D changes LR to .005 at step1000 without resetting state; Cholesky800; no outer labels",
        selection="Complete candidates only; seed mean then date mean NLPD; within <.01 of minimum choose lowest median individual fit train_s")
    manifest = json.loads(json.dumps(manifest))
    path = OUT/"manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Frozen script/config/data/walks changed"
    else:
        path.write_text(json.dumps(manifest, indent=2)+"\n")
    print("FROZEN inner200/validation50", dates, flush=True)
    return torch.from_numpy(graph), torch.from_numpy(ids), torch.from_numpy(targets), scale, matrices, held.sha(path.read_bytes())


def checks(graph, matrices):
    ids = torch.tensor([0, 1, 2, 5, 3, 4])
    diagnostics = dict(Poly=check_kernel(polynomial_kernel(graph[ids][:, ids])), General=[])
    y, target = torch.tensor([.2, -.7, .4]), torch.tensor([-.3, .5, .1])
    for seed in (303, 404):
        rows = []
        for operator in matrices[seed]:
            m = operator.sparse_csr_tensor
            rows.append(torch.from_numpy(csr_matrix((m.values().numpy(), m.col_indices().numpy(), m.crow_indices().numpy()), shape=m.shape)[ids.numpy()].toarray()))
        model = model_for("General", graph, matrices[seed], ids[:3], y)
        for modulation in (None, torch.tensor([1., -.7, .2])):
            model.train()
            if modulation is not None:
                model.covar_module.initialize(raw_modulation_function=modulation)
            phi = sum(f*r for f,r in zip(model.covar_module.raw_modulation_function, rows))
            K = phi @ phi.T
            actual = model.covar_module(ids).to_dense()
            torch.testing.assert_close(actual, K, atol=1e-10, rtol=1e-9)
            assert torch.linalg.eigvalsh(K.detach()).min() > -1e-10
            H = K[:3, :3]+model.likelihood.noise*torch.eye(3)
            expected = -torch.distributions.MultivariateNormal(torch.zeros(3), H).log_prob(y)/3
            loss = -gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)(model(ids[:3]), y)
            torch.testing.assert_close(loss, expected, atol=1e-10, rtol=1e-9)
            a = torch.autograd.grad(loss, tuple(model.parameters()))
            b = torch.autograd.grad(expected, tuple(model.parameters()), retain_graph=True)
            for candidate, reference in zip(a, b):
                torch.testing.assert_close(candidate, reference, atol=1e-9, rtol=1e-8)
            model.eval()
            with torch.no_grad():
                _, candidate_mean, candidate_variance = posterior(actual.detach(), y, model.likelihood.noise)
                mean = K[3:, :3] @ torch.linalg.solve(H, y)
                variance = K.diagonal()[3:]-(K[3:, :3]*torch.linalg.solve(H, K[:3, 3:]).T).sum(1)
                assert (variance > 0).all()
                torch.testing.assert_close(candidate_mean, mean, atol=1e-10, rtol=1e-9)
                torch.testing.assert_close(candidate_variance, variance, atol=1e-10, rtol=1e-9)
                nlpd = -torch.distributions.Normal(mean, (variance+model.likelihood.noise).sqrt()).log_prob(target).mean()
                torch.testing.assert_close(torch.tensor(score(candidate_mean, candidate_variance, target, model.likelihood.noise, 1.)[1]), nlpd, atol=1e-10, rtol=1e-9)
            diagnostics["General"].append(dict(seed=seed, covariance=float((actual-K).detach().abs().max()),
                nll=float((loss-expected).detach().abs()), gradient=max(float((x-z).abs().max()) for x,z in zip(a,b)),
                mean=float((candidate_mean-mean).abs().max()), variance=float((candidate_variance-variance).abs().max()),
                nlpd=float(abs(score(candidate_mean,candidate_variance,target,model.likelihood.noise,1.)[1]-nlpd))))
    return diagnostics


def run(graph, ids, targets, scale, matrices, fingerprint, date, arm, seed, policy, steps, tag):
    path, checkpoint = OUT/f"{tag}.json", OUT/f"{tag}.pt"
    if path.exists():
        row = json.loads(path.read_text())
        assert row["manifest_sha"] == fingerprint and row["steps"] == steps
        assert row["status"] in ("complete", "failed"), "Interrupted case; no automatic refit"
        assert held.sha(checkpoint.read_bytes()) == row["checkpoint_sha"]
        print("SKIP verified", tag, flush=True)
        return row
    assert not checkpoint.exists(), "Orphan checkpoint; no automatic refit"
    row = dict(tag=tag, date=date, arm=arm, seed=seed, policy=policy, steps=steps, manifest_sha=fingerprint, status="running")
    path.write_text(json.dumps(row))
    model, optimizer, losses, noises, rates = None, None, [], [], []
    try:
        torch.manual_seed(0)
        clock = perf_counter()
        model = model_for(arm, graph, matrices.get(seed), ids[:200], targets[date, :200])
        row.update(setup_s=perf_counter()-clock, initial={n:p.detach().tolist() for n,p in model.named_parameters()})
        optimizer = torch.optim.AdamW(model.parameters(), lr=POLICIES[policy][0], weight_decay=.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        clock = perf_counter()
        for step in range(steps):
            if policy == "D" and step == 1000:
                for group in optimizer.param_groups:
                    group["lr"] = .005
            optimizer.zero_grad()
            loss = -mll(model(ids[:200]), targets[date, :200])
            loss.backward()
            assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            losses.append(loss.item())
            noises.append(float(model.likelihood.noise.detach()))
            rates.append(optimizer.param_groups[0]["lr"])
            if (step+1) % 250 == 0:
                print(tag, step+1, loss.item(), flush=True)
        row["train_s"] = perf_counter()-clock
        optimizer.zero_grad()
        final = -mll(model(ids[:200]), targets[date, :200])
        final.backward()
        assert torch.isfinite(final) and all(torch.isfinite(p).all() and torch.isfinite(p.grad).all() for p in model.parameters())
        row.update(nll=float(final.detach()), gradient_norm=float(torch.cat([p.grad.flatten() for p in model.parameters()]).norm()),
            last100_change=losses[-1]-losses[-min(100,len(losses))], noise=noises[-1], noise_min=min(noises), noise_max=max(noises),
            parameters={n:p.detach().tolist() for n,p in model.named_parameters()})
        model.eval()
        clock = perf_counter()
        with torch.no_grad():
            nll, mean, variance = posterior(model.covar_module(ids).to_dense(), targets[date, :200], model.likelihood.noise)
            np.testing.assert_allclose(float(nll), row["nll"], atol=1e-10, rtol=1e-10)
            rmse, nlpd = score(mean, variance, targets[date, 200:], model.likelihood.noise, float(scale[date]))
        row.update(status="complete", rmse=rmse, nlpd=nlpd, validation_s=perf_counter()-clock)
        torch.save(dict(mean=mean, latent_variance=variance), OUT/f"{tag}_posterior.pt")
    except Exception as error:
        row.update(status="failed", error=repr(error))
    finally:
        if model is not None:
            torch.save(dict(model=model.state_dict(), optimizer=None if optimizer is None else optimizer.state_dict(), losses=losses, noises=noises, rates=rates), checkpoint)
            row["checkpoint_sha"] = held.sha(checkpoint.read_bytes())
        path.write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
    print(row["status"].upper(), tag, flush=True)
    return row


def select(rows):
    summaries, selected = [], {}
    for arm in ("Poly", "General"):
        for policy in POLICIES:
            chosen = [r for r in rows if r["arm"] == arm and r["policy"] == policy]
            row = dict(arm=arm, policy=policy, complete=sum(r["status"] == "complete" for r in chosen), eligible=False)
            if row["complete"] == (3 if arm == "Poly" else 6):
                row.update(eligible=True, median_train_s=float(np.median([r["train_s"] for r in chosen])))
                for key in ("nlpd", "rmse", "nll", "setup_s", "validation_s"):
                    row[key] = float(np.mean([np.mean([r[key] for r in chosen if r["date"] == d]) for d in range(3)]))
            summaries.append(row)
        eligible = [r for r in summaries if r["arm"] == arm and r["eligible"]]
        if eligible:
            best = min(r["nlpd"] for r in eligible)
            selected[arm] = min([r for r in eligible if r["nlpd"]-best < .01], key=lambda r:r["median_train_s"])["policy"]
    (OUT/"selection.json").write_text(json.dumps(dict(summary=summaries, selected=selected), indent=2)+"\n")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv)>1 else "check"
    assert stage in ("check", "pilot", "full", "replay")
    for source, revision in ((CORE, CORE_COMMIT), (UPSTREAM, EXPERIMENT_COMMIT)):
        assert subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"], text=True).strip() == revision
        subprocess.run(["git","-C",str(source),"diff","--exit-code"], check=True)
    graph, ids, targets, scale, matrices, fingerprint = prepare()
    original_dense = LinearOperator.to_dense

    def selected_dense(operator, *args, **kwargs):
        assert operator.shape != (1016, 1016), "Full graph covariance/identity forbidden"
        return original_dense(operator, *args, **kwargs)

    with gpytorch.settings.max_cholesky_size(800), gpytorch.settings.max_eager_kernel_size(0), patch.object(LinearOperator,"to_dense",selected_dense):
        if stage == "check":
            (OUT/"checks.json").write_text(json.dumps(dict(manifest_sha=fingerprint, diagnostics=checks(graph,matrices)), indent=2))
        else:
            assert json.loads((OUT/"checks.json").read_text())["manifest_sha"] == fingerprint
            if stage == "replay":
                row = run(graph,ids,targets,scale,matrices,fingerprint,0,"Poly",0,"A",1000,"replay_2_Poly_0_A")
                original = json.loads((OUT/"2_Poly_0_A.json").read_text())
                assert row["status"] == original["status"] == "complete" and row["initial"] == original["initial"]
                a, b = [torch.load(OUT/f"{tag}.pt", weights_only=True) for tag in (row["tag"],original["tag"])]
                assert a["losses"] == b["losses"]
                for name in row["parameters"]:
                    assert torch.equal(a["model"][name],b["model"][name])
                print("PASS A replay: identical initial parameters, 1000 losses and final parameters", flush=True)
            else:
                cases = [(0,a,s,"A") for a,s in ARMS[:2]] if stage == "pilot" else [(d,a,s,p) for d in range(3) for p in POLICIES for a,s in ARMS]
                if stage == "full":
                    pilots = [json.loads((OUT/f"pilot_2_{a}_{s}_A.json").read_text()) for a,s in ARMS[:2]]
                    assert all(r["status"] == "complete" and r["manifest_sha"] == fingerprint for r in pilots)
                rows = []
                for d,a,s,p in cases:
                    tag = f"{'pilot_' if stage == 'pilot' else ''}{[2,6,10][d]}_{a}_{s}_{p}"
                    rows.append(run(graph,ids,targets,scale,matrices,fingerprint,d,a,s,p,20 if stage == "pilot" else POLICIES[p][1],tag))
                with patch.object(held,"OUT",OUT):
                    held.save_rows(rows)
                if stage == "pilot":
                    assert all(r["status"] == "complete" for r in rows), "Pilot failed; stop"
                else:
                    select(rows)
