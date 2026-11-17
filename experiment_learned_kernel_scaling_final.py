"""Continue the frozen development, sparse smoke and 72-fit mean-only study."""

from contextlib import nullcontext
import json
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch
import warnings

import gpytorch
from linear_operator.utils.warnings import NumericalWarning
import numpy as np
import torch
import experiment_learned_kernel_scaling as prior
import experiment_learned_kernel_numerical_learning as learning
from experiment_grf_large_synthetic import SparseGuard

OUT = Path(__file__).resolve().parent / "results/learned_kernel_scaling_final"
SELECTION_SHA = "b781f585c498a2721d9c06165657c9bf4d742f53d8e56459fcfed0c7dde974fc"


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")


def interrupted(signum, frame):
    raise TimeoutError("External watchdog terminated this attempt")


def run(n, r, regime, arm, walks, steps, audit, provenance):
    tag = f"n{n}_r{r}_{regime}_{arm}_W{walks}_{steps}"
    path, checkpoint = OUT/f"{tag}.json", OUT/f"{tag}.pt"
    saved = json.loads(path.read_text()) if path.exists() else None
    assert not saved or saved["provenance"] == provenance and all(prior.sha((OUT/f"{'development' if r == 9 else 'final'}/n{n}_r{r}"/name).read_bytes()) == value for name,value in saved.get("input_hashes",{}).items())
    if saved and not audit:
        if saved["status"] == "complete":
            assert saved["checkpoint_sha"] == prior.sha(checkpoint.read_bytes())
        print("PRESERVED", tag, saved["status"], flush=True)
        return
    assert audit or not checkpoint.exists(), "Orphan checkpoint: preserve, do not retry"
    row = dict(n=n, r=r, regime=regime, arm=arm, W=walks, requested_steps=steps, steps=0,
        status="attempted", losses=[], trace_seed=781000+n+100000*r, provenance=provenance, replay=steps == 200)
    if audit:
        path = OUT/f"{tag}_audit.json"
        assert not path.exists(), "Audit already recorded"
    write(path, row)
    model, mean, clock = None, None, perf_counter()
    with warnings.catch_warnings(record=True) as caught, patch.object(prior, "OUT", OUT):
        warnings.simplefilter("always", NumericalWarning)
        try:
            if r != 9:
                manifest = json.loads((OUT/"manifest.json").read_text())
                assert manifest["provenance"] == provenance and [n,r,regime,arm,walks,steps] in manifest["schedule"]
                row["manifest_sha"] = prior.sha((OUT/"manifest.json").read_bytes())
            graph, targets, train, test, scale, directory, fingerprint, _ = prior.inputs(n, r)
            y = targets[regime]
            model, walk_s = prior.construct(graph, train, y, arm, walks, directory, r)
            matrices = model.covar_module.rw_mats if arm == "General" else None
            row.update(fingerprint=fingerprint, input_hashes={p.name:prior.sha(p.read_bytes()) for p in directory.iterdir() if p.is_file()}, walk_s=walk_s)
            guard = SparseGuard(n, len(train))
            joint = len(train)+len(test)
            guard.forbidden |= {(joint,joint),(n,joint),(joint,n),(len(train),joint),(joint,len(train))}
            with learning.inference(), guard if n == 512 or audit else nullcontext():
                setup = perf_counter()
                torch.manual_seed(0)
                model = prior.model_for("Cheb" if arm == "Heat" else arm, graph, matrices, train, y[train])
                optimizer = torch.optim.AdamW(model.parameters(), lr=.05)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
                row.update(setup_s=perf_counter()-setup, initial=learning.raw(model))
                assert sum(p.numel() for p in model.parameters()) == (3 if arm == "Heat" else 4)
                if audit:
                    assert saved["status"] == "complete" and saved["checkpoint_sha"] == prior.sha(checkpoint.read_bytes())
                    assert saved["fingerprint"] == fingerprint and saved["input_hashes"] == row["input_hashes"]
                    state = torch.load(checkpoint, weights_only=True)
                    model.load_state_dict(state["model"])
                    mean, residual = learning.prediction(model, train, test, y)
                    torch.testing.assert_close(mean, state["mean"], atol=.002, rtol=1e-8)
                else:
                    torch.manual_seed(row["trace_seed"])
                    training = perf_counter()
                    for step in range(steps):
                        optimizer.zero_grad()
                        loss = -mll(model(train), y[train])
                        loss.backward()
                        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
                        assert not any(issubclass(w.category, NumericalWarning) for w in caught)
                        optimizer.step()
                        row["losses"].append(float(loss.detach()))
                        row["steps"] = step+1
                        assert all(torch.isfinite(p).all() for p in model.parameters())
                    row["training_s"] = perf_counter()-training
                    model.eval()
                    with torch.no_grad(), gpytorch.settings.skip_posterior_variances(True):
                        prediction = perf_counter()
                        mean = model(test).mean
                        row["prediction_s"] = perf_counter()-prediction
                        residual = prior.residual(model, train, y)
                    row["total_s"] = row["setup_s"]+walk_s+row["training_s"]+row["prediction_s"]
                assert torch.isfinite(mean).all() and np.isfinite(residual) and residual <= 2e-4
                row.update(rmse=float(scale[regime]*(mean-y[test]).square().mean().sqrt()), residual=residual, final=learning.raw(model))
                if not audit:
                    rng = torch.get_rng_state()
                    model.train()
                    torch.manual_seed(row["trace_seed"])
                    with torch.no_grad():
                        row["estimated_nll"] = float(-mll(model(train), y[train]))
                    if n == 2500:
                        with learning.inference(exact=True):
                            exact_nll, gradient, exact_mean = learning.oracle(model, train, test, y)
                        row.update(exact_nll=exact_nll, exact_gradient=gradient.tolist(), nll_error=row["estimated_nll"]-exact_nll,
                            mean_error=float((mean-exact_mean).abs().max()))
                        assert abs(row["nll_error"]) <= .02 and row["mean_error"] <= .002 and torch.isfinite(gradient).all()
                    torch.set_rng_state(rng)
                    model.train()
                    if steps == 200:
                        control = json.loads((OUT/f"n{n}_r{r}_{regime}_{arm}_W{walks}_100.json").read_text())
                        np.testing.assert_array_equal(row["losses"][:100], control["losses"])
                assert np.isfinite(row["rmse"]) and np.isfinite(row.get("estimated_nll", 0))
            assert not any(issubclass(w.category, NumericalWarning) for w in caught)
            row["status"] = "complete"
        except Exception as error:
            row.update(status="failed", error=repr(error))
        row.update(wall_s=perf_counter()-clock, warnings=[f"{w.category.__name__}: {w.message}" for w in caught])
        if not audit:
            torch.save(dict(model=model.state_dict() if model else None, mean=mean, rng=torch.get_rng_state(), row=row), checkpoint)
            row["checkpoint_sha"] = prior.sha(checkpoint.read_bytes())
        write(path, row)
        print(tag, row["status"], row.get("total_s"), row.get("rmse"), row.get("error", ""), flush=True)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    signal.signal(signal.SIGTERM, interrupted)
    OUT.mkdir(parents=True, exist_ok=True)
    selection = learning.OUT/"selection.json"
    assert prior.sha(selection.read_bytes()) == SELECTION_SHA
    evidence = json.loads(selection.read_text())
    assert evidence["common"] == dict(probes=32,lanczos=16) and evidence["W1024_eligible"]
    for path, revision in ((prior.CORE,prior.CORE_COMMIT),(prior.UPSTREAM,prior.EXPERIMENT_COMMIT)):
        assert subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip() == revision
        subprocess.run(["git","-C",str(path),"diff","HEAD","--exit-code"],check=True)
    provenance = dict(base="f779b471b5b1b94197a9fd5e79072eff23f821e3", selection_sha=SELECTION_SHA,
        order_sha=prior.sha((OUT/"work_order.md").read_bytes()), sources={p.name:prior.sha(p.read_bytes()) for p in Path(__file__).resolve().parent.glob("*.py")},
        packages=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True).splitlines(), upstream=[prior.CORE_COMMIT,prior.EXPERIMENT_COMMIT],
        settings=dict(probes=32,lanczos=16,cg=1e-4,maxCG=1000,degree=6,J=3,halt=.5,lr=.05,decay=.01,threads=4,workers=4),
        seeds="base+n+100000*r; graph/data/split/z2/walk/trace bases731000/741000/751000/761000/771000/781000")
    protocol = OUT/"protocol.json"
    if protocol.exists():
        assert json.loads(protocol.read_text()) == provenance, "Source/config/environment changed"
    else:
        write(protocol, provenance)
    if sys.argv[1] in ("fit", "audit"):
        mode,n,r,regime,arm,walks,steps = sys.argv[1:]
        run(int(n),int(r),int(regime),arm,int(walks),int(steps),mode == "audit",provenance)
    elif sys.argv[1] == "choose":
        rows = {(regime,arm,w):json.loads((OUT/f"n2500_r9_{regime}_{arm}_W{w}_100.json").read_text()) for regime in (0,1) for arm,w in prior.ARMS}
        assert all(rows[regime,arm,w]["status"] == "complete" for regime in (0,1) for arm,w in prior.ARMS if w != 1024)
        cheap = all(rows[regime,"General",1024]["status"] == "complete" and rows[regime,"General",1024]["rmse"] <= 1.01*rows[regime,"General",4096]["rmse"] for regime in (0,1))
        write(OUT/"walk_selection.json", dict(W=1024 if cheap else 4096, rule="safe and RMSE<=1.01*4096 in EACH H/M regime", rows=list(rows.values())))
    elif sys.argv[1] == "freeze":
        walks = json.loads((OUT/"walk_selection.json").read_text())["W"]
        required = [OUT/f"n{n}_r9_{regime}_{arm}_W{walks if arm == 'General' else 0}_{steps}.json" for n,steps in [(2500,200),(512,2)] for regime in (0,1) for arm in ("Heat","Poly","General")]
        assert all(json.loads(p.read_text())["status"] == "complete" for p in required)
        schedule = []
        for index,n in enumerate((5000,10000,25000,50000)):
            for r in (0,1,2):
                for regime in (0,1):
                    arms = ["Heat","Poly","General"]
                    offset = (index+r+regime)%3
                    schedule.extend([n,r,regime,arm,walks if arm == "General" else 0,100] for arm in arms[offset:]+arms[:offset])
        manifest = dict(provenance=provenance,W=walks,schedule=schedule,walk_selection_sha=prior.sha((OUT/"walk_selection.json").read_bytes()),development={p.name:prior.sha(p.read_bytes()) for p in OUT.glob("n*_r9_*.json")})
        assert not (OUT/"manifest.json").exists(), "Keep the original freeze"
        write(OUT/"manifest.json", manifest)
    else:
        raise ValueError("Use fit, choose, freeze or audit")
