"""Fresh development-only same-kernel numerical and 100-update learning checks."""

from contextlib import contextmanager
import json
from pathlib import Path
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
from experiment_speed_accuracy import controls
from experiment_grf_traffic import posterior

OUT = Path(__file__).resolve().parent / "results/learned_kernel_numerical_learning"
MENU = [(32, 16), (64, 24), (128, 24)]


@contextmanager
def inference(probes=32, lanczos=16, exact=False):
    with controls(dict(degree=6, cg=1e-4, probes=probes, lanczos=lanczos)), gpytorch.settings.max_cg_iterations(1000), \
            gpytorch.settings.max_cholesky_size(800 if exact else 0), gpytorch.settings.fast_computations(
                log_prob=not exact, solves=not exact, covar_root_decomposition=not exact):
        yield


def raw(model):
    return {name: p.detach().tolist() for name, p in model.named_parameters()}


def gradient(loss, model):
    return torch.cat([g.flatten() for g in torch.autograd.grad(loss, tuple(model.parameters()))]).detach()


def oracle(model, train, validation, y):
    loss, mean, _ = posterior(model.covar_module(torch.cat([train, validation])).to_dense(), y[train], model.likelihood.noise)
    return float(loss.detach()), gradient(loss, model), mean.detach()


def prediction(model, train, validation, y):
    model.eval()
    with torch.no_grad(), gpytorch.settings.skip_posterior_variances(True):
        return model(validation).mean, prior.residual(model, train, y)


def numerical(model, arm, train, validation, y):
    groups = []
    for state in range(2):
        if state:
            prior.second_state(model, arm)
        with inference(exact=True):
            reference, g, exact_mean = oracle(model, train, validation, y)
            model.train()
            rng = torch.get_rng_state()
            loss = -gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)(model(train), y[train])
            exact_gradient = gradient(loss, model)
            mll_error, gradient_error = float(loss.detach())-reference, float((exact_gradient-g).abs().max())
            torch.testing.assert_close(loss.detach(), torch.tensor(reference), atol=1e-9, rtol=1e-8)
            torch.testing.assert_close(exact_gradient, g, atol=1e-9, rtol=1e-8)
            assert torch.equal(rng, torch.get_rng_state()), "Exact MLL consumed random probes"
        for probes, lanczos in MENU:
            rows = []
            for seed in (81, 82, 83):
                with warnings.catch_warnings(record=True) as caught, inference(probes, lanczos):
                    warnings.simplefilter("always", NumericalWarning)
                    model.train()
                    torch.manual_seed(seed)
                    loss = -gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)(model(train), y[train])
                    estimate = gradient(loss, model)
                    mean, residual = prediction(model, train, validation, y)
                error = estimate-g
                row = dict(seed=seed, gradient=estimate.tolist(), errors=error.tolist(), scaled_errors=(error.abs()/(.02+.05*g.abs())).tolist(),
                    wider_scaled_errors=(error.abs()/(.05+.10*g.abs())).tolist(), old_absolute_pass=(error.abs() <= .02).tolist(),
                    individual_mean_bound_pass=(error.abs() <= .02+.05*g.abs()).tolist(), nll_error=float(loss.detach())-reference,
                    residual=residual, mean_error=float((mean-exact_mean).abs().max()), warnings=[str(w.message) for w in caught])
                row["safe"] = bool(torch.isfinite(estimate).all() and torch.isfinite(loss) and torch.isfinite(mean).all() and np.isfinite(residual) and residual <= 2e-4 and row["mean_error"] <= .002 and not any(issubclass(w.category, NumericalWarning) for w in caught))
                row["passed"] = row["safe"] and abs(row["nll_error"]) <= .02 and bool(((estimate*g >= 0) | (g.abs() <= .02)).all()) and max(row["wider_scaled_errors"]) <= 1
                rows.append(row)
            mean_scaled = (np.abs(np.mean([r["errors"] for r in rows], axis=0))/(.02+.05*g.abs().numpy())).tolist()
            groups.append(dict(state=state, probes=probes, lanczos=lanczos, reference_nll=reference, reference_gradient=g.tolist(),
                raw=raw(model), exact_mll_error=mll_error, exact_gradient_error=gradient_error,
                mean_scaled_errors=mean_scaled, trials=rows, safe=all(r["safe"] for r in rows), passed=all(r["passed"] for r in rows) and max(mean_scaled) <= 1))
    return dict(status="complete", groups=groups)


def fit(model, train, validation, y, scale, probes, lanczos, seed):
    exact = seed == 0
    row = dict(status="failed", initial=raw(model), losses=[], steps=0, seed=seed)
    with warnings.catch_warnings(record=True) as caught, inference(probes, lanczos, exact):
        warnings.simplefilter("always", NumericalWarning)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.05)
        torch.manual_seed(seed)
        rng = torch.get_rng_state()
        clock = perf_counter()
        try:
            for step in range(100):
                optimizer.zero_grad()
                loss = -gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)(model(train), y[train])
                loss.backward()
                assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
                assert not any(issubclass(w.category, NumericalWarning) for w in caught), "NumericalWarning during training"
                optimizer.step()
                row["steps"] = step+1
                row["losses"].append(float(loss.detach()))
                assert all(torch.isfinite(p).all() for p in model.parameters())
            row["status"] = "complete"
        except Exception as error:
            row["error"] = repr(error)
        row.update(training_s=perf_counter()-clock, final=raw(model), noise=float(model.likelihood.noise.detach()),
            rng_unchanged=torch.equal(rng, torch.get_rng_state()))
        try:
            if row["status"] != "complete":
                raise RuntimeError("Training incomplete; endpoint audit skipped")
            with inference(exact=True):
                nll, g, exact_mean = oracle(model, train, validation, y)
                if exact:
                    endpoint = -gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)(model(train), y[train])
                    torch.testing.assert_close(endpoint.detach(), torch.tensor(nll), atol=1e-9, rtol=1e-8)
                    assert row["rng_unchanged"], "Exact training consumed random probes"
            mean, residual = prediction(model, train, validation, y)
            row.update(nll=nll, gradient=g.tolist(), exact_rmse=float(scale*(exact_mean-y[validation]).square().mean().sqrt()),
                rmse=float(scale*(mean-y[validation]).square().mean().sqrt()), residual=residual, mean_error=float((mean-exact_mean).abs().max()))
            assert np.isfinite([nll, row["rmse"], residual, row["mean_error"]]+g.tolist()).all()
        except Exception as error:
            row.update(status="failed", audit_error=repr(error))
        row["warnings"] = [str(w.message) for w in caught]
        row["numerical_warning"] = any(issubclass(w.category, NumericalWarning) for w in caught)
    return row


def selection(cases, protocol):
    gates = []
    for arm, walks in prior.ARMS:
        for probes, lanczos in MENU:
            numerical_groups, fits = [], []
            for tag in cases:
                if cases[tag][2:] != [arm, walks]:
                    continue
                numerical_groups.extend(g for g in json.loads((OUT/f"{tag}_numerical.json").read_text())["groups"] if g["probes"] == probes)
                reference = json.loads((OUT/f"{tag}_0_0.json").read_text())
                for seed in (91, 92):
                    row = json.loads((OUT/f"{tag}_{probes}_{seed}.json").read_text())
                    passed = row["status"] == reference["status"] == "complete" and not row["numerical_warning"] and row["steps"] == 100 and row["residual"] <= 2e-4 and row["mean_error"] <= .002 and row["nll"]-reference["nll"] <= .02 and row["rmse"] <= 1.02*reference["exact_rmse"]
                    fits.append(dict(case=tag, seed=seed, passed=passed))
            gates.append(dict(arm=arm, W=walks, probes=probes, lanczos=lanczos, numerical_pass=sum(g["passed"] for g in numerical_groups),
                learning=fits, passed=len(numerical_groups) == 8 and all(g["passed"] for g in numerical_groups) and len(fits) == 8 and all(f["passed"] for f in fits)))
    eligible = [p for p, _ in MENU if all(g["passed"] for g in gates if g["probes"] == p and g["W"] != 1024)]
    common = next((dict(probes=p, lanczos=l) for p, l in MENU if p in eligible), None)
    return dict(gates=gates, common=common, W1024_eligible=bool(common and any(g["passed"] and g["W"] == 1024 and g["probes"] == common["probes"] for g in gates)),
        protocol_sha=prior.sha(json.dumps(protocol, sort_keys=True).encode()), protocol=protocol, rule="First common numerical AND learning pass in 32/16,64/24,128/24; W1024 optional")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    for path, revision in ((prior.CORE, prior.CORE_COMMIT), (prior.UPSTREAM, prior.EXPERIMENT_COMMIT)):
        assert subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() == revision
        subprocess.run(["git", "-C", str(path), "diff", "HEAD", "--exit-code"], check=True)
    with patch.object(prior, "OUT", OUT):
        data = {r: prior.inputs(1000, r) for r in (10, 11)}
        cases = {f"r{r}_{regime}_{arm}{w}": [r, regime, arm, w] for r in data for regime in (0, 1) for arm, w in prior.ARMS}
        for r, (graph, targets, train, validation, scales, directory, fingerprint, metadata) in data.items():
            assert len(train) == len(validation) == 250 and graph.dtype == torch.float64
            for arm, w in prior.ARMS:
                model, _ = prior.construct(graph, train, targets[0], arm, w, directory, r)
                assert sum(p.numel() for p in model.parameters()) == (3 if arm == "Heat" else 4)
        protocol = dict(base="986389eeaab0fad1319daf87687bba4f532c5fb8", work_order=prior.sha(Path(sys.argv[2]).read_bytes()),
            sources={p.name: prior.sha(p.read_bytes()) for p in Path(__file__).resolve().parent.glob("*.py")},
            packages=subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
            upstream=[prior.CORE_COMMIT, prior.EXPERIMENT_COMMIT], data={str(r): dict(fingerprint=d[6], metadata={k:v for k,v in d[7].items() if not k.endswith("_s")}) for r, d in data.items()},
            files={str(p.relative_to(OUT)): prior.sha(p.read_bytes()) for p in (OUT/"final").rglob("*") if p.is_file()},
            development_replicates=[10, 11], cases=cases, menu=MENU, numerical_seeds=[81,82,83], learning_seeds=[91,92],
            schedule=[f"{tag}_{p}_{s}" for tag in cases for p, s in [(0,0)]+[(p,s) for p,_ in MENU for s in (91,92)]])
        frozen = OUT/"protocol.json"
        if frozen.exists():
            assert json.loads(frozen.read_text()) == json.loads(json.dumps(protocol)), "Frozen provenance changed"
        else:
            frozen.write_text(json.dumps(protocol, indent=2))
        if sys.argv[1] == "select":
            (OUT/"selection.json").write_text(json.dumps(selection(cases, protocol), indent=2))
        else:
            assert sys.argv[1] in ("numerical", "learn")
            for tag, (r, regime, arm, walks) in cases.items():
                graph, targets, train, validation, scales, directory, fingerprint, metadata = data[r]
                jobs = [(0, 0, 0)] if sys.argv[1] == "numerical" else [(32,16,0)]+[(p,l,s) for p,l in MENU for s in (91,92)]
                for probes, lanczos, seed in jobs:
                    path = OUT/f"{tag}_{'numerical' if sys.argv[1] == 'numerical' else f'{probes if seed else 0}_{seed}'}.json"
                    if path.exists():
                        saved = json.loads(path.read_text())
                        assert saved["fingerprint"] == fingerprint and saved["checkpoint_sha"] == prior.sha(path.with_suffix(".pt").read_bytes())
                        print("PRESERVED", path.name, flush=True)
                        continue
                    model, walk_s = prior.construct(graph, train, targets[regime], arm, walks, directory, r)
                    initial = raw(model)
                    row = dict(status="failed")
                    try:
                        if sys.argv[1] == "numerical":
                            row = numerical(model, arm, train, validation, targets[regime])
                        else:
                            gate = json.loads((OUT/f"{tag}_numerical.json").read_text())
                            assert initial == gate["initial"], "Fresh initialization changed"
                            if seed and not all(g["safe"] for g in gate["groups"] if g["probes"] == probes):
                                row = dict(status="blocked", reason="Unsafe fixed-state numerical trial")
                            else:
                                row = fit(model, train, validation, targets[regime], scales[regime], probes, lanczos, seed)
                    except Exception as error:
                        row.update(status="failed", error=repr(error))
                    checkpoint = path.with_suffix(".pt")
                    torch.save(dict(model=model.state_dict(), rng=torch.get_rng_state(), row=row), checkpoint)
                    row.update(initial=initial, parameter_names=list(initial), fingerprint=fingerprint, cached_walk_s=walk_s,
                        checkpoint_sha=prior.sha(checkpoint.read_bytes()))
                    path.write_text(json.dumps(row, indent=2))
                    print(path.name, row["status"], row.get("training_s"), row.get("error", ""), flush=True)
