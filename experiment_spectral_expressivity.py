"""Frozen five-date PEMS-BAY degree screen. Run from repository root."""
import json
import signal
import warnings
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import gpytorch
import numpy as np
import pandas as pd
import torch
from linear_operator.utils.warnings import NumericalWarning
from scipy.optimize import minimize

from experiment_grf_heldout import sha
from experiment_grf_traffic import UPSTREAM, posterior
from experiment_learned_polynomial_traffic import LearnedPolynomialKernel, model_for
from graph_gp import HeatOperator

OUT = Path("results/spectral_expressivity")
FRESH = Path("results/traffic/learned_polynomial_fresh")
DATES = [f"2017-{day} 17:30:00" for day in ("01-23", "02-20", "03-20", "04-17", "05-15")]
BOUNDS = np.log([[1e-4, 64], [1e-6, 1e4], [1e-4, 1e2]])

class Poly4(LearnedPolynomialKernel):
    def __init__(self, laplacian):
        super().__init__(laplacian)
        self.a3 = torch.nn.Parameter(torch.zeros_like(self.a1))
        self.a4 = torch.nn.Parameter(torch.zeros_like(self.a1))

    @property
    def coefficients(self):
        c = torch.stack((1+self.a1-self.a2+self.a3-self.a4, self.a1, self.a2, self.a3, self.a4))
        i = torch.arange(5)
        half = torch.outer(c, c).flatten()/2
        sums, differences = i[:, None]+i, (i[:, None]-i).abs()
        return c.new_zeros(9).index_add(0, sums.flatten(), half).index_add(0, differences.flatten(), half)

def heat_block(logpars, eigenvalues, rows):
    beta, variance, noise = logpars.exp()
    return (rows * (variance * torch.exp(-beta*eigenvalues))) @ rows.T, noise

def labels(hdf, sensors):
    with pd.HDFStore(hdf, mode="r") as store:
        frame = store.get_storer("speed")
        times = frame.read_index("axis1")
        mondays = times[(times.weekday == 0) & (times.hour == 17) & (times.minute == 30)]
        assert mondays[[3, 7, 11, 15, 19]].astype(str).tolist() == DATES
        columns = frame.read_index("block0_items").get_indexer(sensors)
        assert min(columns) >= 0 and len(set(columns)) == len(sensors)
        values = np.array([[frame.group.block0_values[t, c] for c in columns] for t in times.get_indexer(DATES)])
        assert np.isfinite(values).all()
        return torch.from_numpy(values).double()

def run():
    prior = json.loads((FRESH/"manifest.json").read_text())
    assert sha((FRESH/"data.npz").read_bytes()) == prior["data_sha"] == "d14c1e5a45a094987f0c9409ca949bd90de27b6beac69d3a6a4eba187b457e91"
    hdf = UPSTREAM/"data/traffic/PEMS/pems-bay.h5"
    assert sha(hdf.read_bytes()) == prior["original"]["hdf_sha"]
    with np.load(FRESH/"data.npz") as data:
        graph, train, test = (torch.from_numpy(data[k]) for k in ("laplacian", "train", "test"))
        assert sha(np.stack([data["sensor_ids"], data["vertices"]]).tobytes()) == prior["original"]["sensor_map_sha"]
    assert train.tolist() == prior["train_ids"] and test.tolist() == prior["test_ids"]
    raw = labels(hdf, prior["train_sensor_ids"])
    center, scale = raw.mean(1), raw.std(1, unbiased=False)
    assert (scale > 0).all()
    targets = (raw-center[:, None])/scale[:, None]
    eig, vectors = torch.linalg.eigh(graph)
    assert eig.min() >= -1e-10 and eig.max() <= 2+1e-10
    arms = [["Poly2", "Poly4"] if i%2 == 0 else ["Poly4", "Poly2"] for i in range(5)]
    rows = [dict(date=date, arm=a, status="unattempted") for date, order in zip(DATES, arms) for a in [*order, "heat0", "heat8", "heat32"]]
    log = OUT/"attempts.json"
    log.touch(exist_ok=False)
    torch.save(dict(train=train, test=test, targets=targets, center=center, scale=scale), OUT/"training.pt")
    models = {}
    for row in rows:
        date, arm = row["date"], row["arm"]
        y = targets[DATES.index(date)]
        row["status"] = "running"
        log.write_text(json.dumps(rows, indent=2))
        signal.alarm(1800)
        try:
            torch.manual_seed(0)
            start = perf_counter()
            model = model_for("Poly", graph, None, train, y)
            kernel = model.covar_module
            if arm == "Poly4":
                kernel.base_kernel = Poly4(kernel.base_kernel.laplacian)
            row["setup_s"] = perf_counter()-start
            start = perf_counter()
            if arm.startswith("heat"):
                def objective(x):
                    p = torch.tensor(x, requires_grad=True)
                    cov, noise = heat_block(p, eig, train_vectors)
                    loss = posterior(cov, y, noise)[0]
                    return loss.item(), torch.autograd.grad(loss, p)[0].numpy()
                train_vectors = vectors[train]
                beta = float(arm[4:]) or 2*torch.sigmoid(torch.tensor(1.)).item()
                initial = np.log([beta, kernel.outputscale.item(), model.likelihood.noise.item()])
                fit = minimize(objective, initial, jac=True, bounds=BOUNDS, method="L-BFGS-B",
                               options=dict(maxiter=200, ftol=1e-12, gtol=1e-6, maxls=30))
                row.update(nll=fit.fun, params=np.exp(fit.x).tolist(), message=fit.message, success=bool(fit.success))
                assert np.isfinite([fit.fun, *fit.x, *fit.jac]).all()
            else:
                opt = torch.optim.AdamW(model.parameters(), lr=.05, weight_decay=.01)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
                losses = []
                for step in range(1001):
                    opt.zero_grad()
                    loss = -mll(model(train), y)
                    loss.backward()
                    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() and torch.isfinite(p).all() for p in model.parameters())
                    if step < 1000:
                        losses.append(loss.item())
                        opt.step()
                        if step == 999:
                            row["train_s"] = perf_counter()-start
                row.update(nll=loss.item(), grad_norm=torch.cat([p.grad.flatten() for p in model.parameters()]).norm().item())
                checkpoint = OUT/f"{date[:10]}-{arm}.pt"
                torch.save(dict(state=model.state_dict(), losses=losses), checkpoint)
                row["checkpoint_sha"] = sha(checkpoint.read_bytes())
                models[date, arm] = model
            row.setdefault("train_s", perf_counter()-start)
            row["status"] = "fitted" if row.get("success", True) else "optimizer_failed"
        except BaseException as exc:
            row.update(status="failed", error=repr(exc))
            raise
        finally:
            signal.alarm(0)
            log.write_text(json.dumps(rows, indent=2))
    best = [min((r for r in rows if r["date"] == d and "params" in r and r["status"] == "fitted"), key=lambda r:r["nll"], default=None) for d in DATES]
    observed = labels(hdf, prior["test_sensor_ids"])
    ids = torch.cat((train, test))
    for row in rows:
        date, arm = row["date"], row["arm"]
        i = DATES.index(date)
        if arm.startswith("heat") and row is not best[i]:
            continue
        start = perf_counter()
        if arm.startswith("heat"):
            cov, noise = heat_block(torch.tensor(row["params"]).log(), eig, vectors[ids])
        else:
            model = models[date, arm].eval()
            with torch.no_grad():
                cov, noise = model.covar_module(ids).to_dense(), model.likelihood.noise.detach()
        nll, mean, _ = posterior(cov, targets[i], noise)
        row["prediction_s"] = perf_counter()-start
        np.testing.assert_allclose(nll.item(), row["nll"], atol=1e-10, rtol=1e-10)
        row["rmse"] = (mean*scale[i]+center[i]-observed[i]).square().mean().sqrt().item()
        row.update(standardized_rmse=row["rmse"]/scale[i].item(), mean=mean.tolist())
    pd.DataFrame(rows).to_csv(OUT/"scores.csv", index=False)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    warnings.filterwarnings("error", category=NumericalWarning)
    original = HeatOperator._matmul
    def guarded(op, rhs):
        assert rhs.shape[-1] < 1016, "Full graph covariance/identity forbidden"
        return original(op, rhs)
    with gpytorch.settings.fast_computations(False, False, False), gpytorch.settings.max_eager_kernel_size(0), gpytorch.settings.max_cholesky_size(800), patch.object(HeatOperator, "_matmul", guarded):
        from spectral_checks import check
        check()
        run()
