"""Train-only exact heat profile on the five PR20 PEMS-BAY snapshots."""
import json
import signal
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from linear_operator.utils.warnings import NumericalWarning
from scipy.optimize import minimize
from scipy.sparse.csgraph import connected_components

from experiment_spectral_expressivity import BOUNDS, DATES, FRESH, OUT as OLD, UPSTREAM, heat_block, labels, model_for, posterior, sha

OUT = Path("results/heat_baseline_profile")
OPTIONS = dict(maxiter=200, ftol=1e-12, gtol=1e-6, maxls=30)


def spectrum(L):
    edges = L != 0
    edges.fill_diagonal_(False)
    isolated = ~edges.any(1)
    components = connected_components(edges.numpy(), directed=False, return_labels=False)
    diag = L.diag()[isolated]
    assert ((diag == 0) | (diag == 1)).all()
    rank = components - int((diag == 1).sum())
    eig, U = torch.linalg.eigh(L)
    zeros = U[:, :rank]
    assert torch.allclose(L, L.T, atol=1e-12, rtol=0) and eig[-1] <= 2+1e-10
    assert rank > 0 and eig[:rank].abs().max() <= 1e-10 and eig[rank] > 1e-10
    residual = (L @ zeros).norm().item()
    assert residual <= 1e-10 and torch.allclose(zeros.T @ zeros, torch.eye(rank), atol=1e-10, rtol=0)
    info = dict(components=components, isolated_diagonal=diag.tolist(), zero_eigenvalues=eig[:rank].tolist(),
                residual=residual, gap=eig[rank].item(), maximum=eig[-1].item())
    eig[:rank] = 0
    return eig, U, rank, info


def tiny_check():
    W = torch.diag(torch.tensor([1., 2., .7, 1.3, .8]), 1)
    W = W + W.T
    root = W.sum(1).sqrt()
    L = torch.eye(6)-W/root[:, None]/root
    eig, U, rank, info = spectrum(L)
    limit = U[:, :rank] @ U[:, :rank].T
    direct = torch.outer(root, root)/root.square().sum()
    close = dict(atol=1e-10, rtol=1e-9)
    for actual, ref in [(limit, direct), (limit @ limit, limit), (L @ limit, 0*L), (limit.T, limit)]:
        torch.testing.assert_close(actual, ref, **close)
    ids, y = [0, 2, 4], torch.tensor([.2, -.7, .4])
    for beta in (64., 256., np.inf):
        G = limit if np.isinf(beta) else (U*(-beta*eig).exp()) @ U.T
        ref = direct if np.isinf(beta) else torch.matrix_exp(-beta*L)
        torch.testing.assert_close(G[ids][:, ids], ref[ids][:, ids], **close)
        torch.testing.assert_close(torch.linalg.matrix_norm(G-limit, 2), L.new_tensor(np.exp(-beta*info["gap"])), **close)
        p = torch.tensor([.2, -.7], requires_grad=True)
        s, nu = p.exp()
        nll = posterior(s*G[ids][:, ids], y, nu)[0]
        A = s*ref[ids][:, ids]+nu*torch.eye(len(y))
        oracle = (y @ torch.linalg.solve(A, y)+torch.linalg.slogdet(A)[1]+len(y)*np.log(2*np.pi))/(2*len(y))
        torch.testing.assert_close(nll, oracle, **close)
        a, b = (torch.autograd.grad(z, p, retain_graph=True)[0] for z in (nll, oracle))
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-8)
        print("tiny beta, covariance/NLL/gradient max errors", beta, (G-ref).abs().max().item(), abs((nll-oracle).item()), (a-b).abs().max().item())


def run():
    tiny_check()
    prior = json.loads((FRESH/"manifest.json").read_text())
    hdf = UPSTREAM/"data/traffic/PEMS/pems-bay.h5"
    assert sha(hdf.read_bytes()) == prior["original"]["hdf_sha"] == "eba3ef97ab496a2fca35e69f2901dca961a16c146106ae8bd70b3375ed63df9e"
    assert sha((FRESH/"data.npz").read_bytes()) == prior["data_sha"] == "d14c1e5a45a094987f0c9409ca949bd90de27b6beac69d3a6a4eba187b457e91"
    saved = torch.load(OLD/"training.pt", weights_only=True)
    with np.load(FRESH/"data.npz") as data:
        graph, train, test = (torch.from_numpy(data[k]) for k in ("laplacian", "train", "test"))
        mapping = dict(zip(data["vertices"], data["sensor_ids"]))
        for key, ids in (("train", train), ("test", test)):
            assert torch.equal(ids, saved[key]) and [mapping[v] for v in ids.tolist()] == prior[key+"_sensor_ids"]
    raw = labels(hdf, prior["train_sensor_ids"])
    center, scale = raw.mean(1), raw.std(1, unbiased=False)
    targets = (raw-center[:, None])/scale[:, None]
    assert all(torch.equal(saved[k], v) for k, v in dict(center=center, scale=scale, targets=targets).items())
    old = pd.DataFrame(json.loads((OLD/"attempts.json").read_text()))
    scores = pd.read_csv(OLD/"scores.csv", float_precision="round_trip").set_index(["date", "arm"])
    assert len(old) == 25 and old.status.eq("fitted").all() and old.date.unique().tolist() == DATES
    winners = old[old.arm.str.startswith("heat")].loc[lambda t: t.groupby("date").nll.idxmin()]
    anchors = winners.set_index("date").params
    eig, U, rank, info = spectrum(graph)
    grid = sorted(set([1e-4, 1.4621171572600098, 2, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096]
                      + [k/info["gap"] for k in (1, 4, 16, 32)]))
    bounds = BOUNDS.copy()
    bounds[0, 1] = np.log(grid[-1])
    model = model_for("Poly", graph, None, train, targets[0])
    shared = [model.covar_module.outputscale.item(), model.likelihood.noise.item()]
    V = U[train]
    blocks = {b: (V*(-b*eig).exp()) @ V.T for b in grid}
    blocks[np.inf] = V[:, :rank] @ V[:, :rank].T
    for i, row in enumerate(winners.itertuples()):
        rec = scores.loc[row.date, row.arm]
        assert row.success and np.isfinite(rec.rmse) and json.loads(rec.params) == row.params
        beta, s, nu = row.params
        assert abs(beta-64) < 1e-10
        anchor_error = posterior(s*blocks[64], targets[i], nu)[0].item()-row.nll
        np.testing.assert_allclose(anchor_error+row.nll, row.nll, atol=1e-10, rtol=1e-9)
        print("anchor NLL difference", row.date, anchor_error)
    manifest = dict(graph=info, grid=grid, dates=DATES, anchors=anchors.to_dict(), shared=shared,
                    log_bounds=bounds.tolist(), options=OPTIONS, epsilon=.001, hashes=sha((OUT/"freeze.sha").read_bytes()),
                    selection="first successful minimum per point, finite grid then refinements; separate infinity")
    with (OUT/"manifest.json").open("x") as f:
        json.dump(manifest, f, indent=2)

    def attempt(date, beta, start, initial):
        y = targets[DATES.index(date)]
        varying = beta is None
        box = bounds if varying else bounds[1:]
        def objective(x):
            p = torch.tensor(x, requires_grad=True)
            cov, noise = heat_block(p, eig, V) if varying else (p[0].exp()*blocks[beta], p[1].exp())
            loss = posterior(cov, y, noise)[0]
            grad = torch.autograd.grad(loss, p)[0].numpy()
            assert np.isfinite(loss.item()) and np.isfinite(grad).all()
            return loss.item(), grad
        signal.alarm(120)
        fit = minimize(objective, np.log(initial), jac=True, method="L-BFGS-B", bounds=box, options=OPTIONS)
        signal.alarm(0)
        physical = np.exp(fit.x).tolist()
        params = physical if varying else [beta, *physical]
        row = dict(date=date, kind="refine" if varying else "fixed", start=start, beta=params[0], s=params[1], nu=params[2],
                   nll=fit.fun, success=bool(fit.success), reason=fit.message, iterations=fit.nit, gradient=fit.jac.tolist(),
                   projected_inf=float(np.max(abs(fit.x-np.clip(fit.x-fit.jac, *box.T)))),
                   bounds_hit=np.where(np.min(abs(fit.x[:, None]-box), axis=1)<=1e-3)[0].tolist())
        with (OUT/"attempts.jsonl").open("a") as f:
            print(json.dumps(row), file=f)
        return row

    started = perf_counter()
    selected = {}
    for date in DATES:
        profile = []
        for beta in blocks:
            pair = [attempt(date, beta, name, init) for name, init in (("anchor", anchors[date][1:]), ("shared", shared))]
            good = [r for r in pair if r["success"]]
            if good:
                profile.append(min(good, key=lambda r:r["nll"]))
        candidates = [r for r in profile if np.isfinite(r["beta"])]
        starts = []
        if candidates:
            best = min(candidates, key=lambda r:r["nll"])
            starts.append(("grid", [best[k] for k in ("beta", "s", "nu")]))
        for name, init in [*starts, ("anchor", anchors[date])]:
            r = attempt(date, None, name, init)
            if r["success"]:
                candidates.append(r)
        selected[date] = dict(anchor=anchors[date])
        for kind, choices in (("finite", candidates), ("limit", [r for r in profile if np.isinf(r["beta"])])):
            if choices:
                best = min(choices, key=lambda r:r["nll"])
                selected[date][kind] = [best[k] for k in ("beta", "s", "nu")]
    (OUT/"selection.json").write_text(json.dumps(selected, indent=2))
    print("optimization seconds", perf_counter()-started)
    started = perf_counter()
    observed = labels(hdf, prior["test_sensor_ids"])
    rows = []
    for i, date in enumerate(DATES):
        for kind, (beta, s, nu) in selected[date].items():
            weights = (eig == 0).double() if np.isinf(beta) else (-beta*eig).exp()
            cross = (U[test]*(s*weights)) @ V.T
            A = (V*(s*weights)) @ V.T + nu*torch.eye(len(train))
            mean = cross @ torch.cholesky_solve(targets[i, :, None], torch.linalg.cholesky(A))
            rmse = (mean[:, 0]*scale[i]+center[i]-observed[i]).square().mean().sqrt().item()
            if kind == "anchor":
                record = scores.loc[date, winners.set_index("date").loc[date, "arm"]]
                np.testing.assert_allclose(mean[:, 0], json.loads(record["mean"]), atol=1e-9, rtol=1e-8)
                np.testing.assert_allclose(rmse, record.rmse, atol=1e-8, rtol=0)
            rows.append(dict(date=date, kind=kind, rmse=rmse, standardized=rmse/scale[i].item(), mean=mean[:, 0].tolist()))
    pd.DataFrame(rows).to_csv(OUT/"scores.csv", index=False)
    print("scoring seconds", perf_counter()-started)
    fits = pd.read_json(OUT/"attempts.jsonl", lines=True, convert_dates=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for date, group in fits[fits.success & fits.kind.eq("fixed")].groupby("date"):
        profile = group.groupby("beta").nll.min()
        line, = ax.semilogx(profile.index[np.isfinite(profile.index)], profile[np.isfinite(profile.index)], ".-", label=date[5:10])
        if np.inf in profile:
            ax.axhline(profile[np.inf], color=line.get_color(), linestyle=":", linewidth=1)
    ax.set(xlabel="Finite heat time β", ylabel="Profiled training NLL per sensor", title="Dotted lines show each date’s infinite-time fit")
    ax.legend(ncol=5)
    fig.tight_layout()
    fig.savefig(OUT/"profile.png", dpi=160)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    warnings.filterwarnings("error", category=NumericalWarning)
    run()
