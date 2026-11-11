"""Paper C.3 settings with documented notebook defaults; see GRF_PAPER_C3_REPLICATION.md."""

import hashlib
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
from time import perf_counter

import gpytorch
import networkx as nx
import numpy as np
import torch
from chebyshev import apply_chebyshev, heat_coefficients
from experiment_grf_traffic import (CORE, CORE_COMMIT, EXPERIMENT_COMMIT, UPSTREAM,
    DiffusionExactKernel, ExactGraphGP, GRFGP, GRFSampler, get_normalized_laplacian, posterior, score)
from experiments.ablation.grf_ablation_sampler import GRFAblationSampler
from grf_gp.kernels.general import GeneralGRFKernel

OUT = Path(__file__).resolve().parent / "results/paper_c3"
CONFIG = dict(beta_sample=10., signal_variance=10., noise_variance=.1, seed=1, fraction=.1,
    halt=.01, cutoff=10, workers=4, threads=4, dtype="float64", device="cpu",
    optimizer="Adam", lr=.01, weight_decay=0., cholesky=1000, generating_jitter=0.)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def timeout(signum, frame):
    raise TimeoutError("Two-hour wall-time limit")


def data(side):
    L = torch.tensor(get_normalized_laplacian(nx.to_numpy_array(nx.grid_2d_graph(side, side))))
    torch.manual_seed(1)
    latent = torch.distributions.MultivariateNormal(torch.zeros(len(L)), 10*torch.matrix_exp(-10*L)).sample()
    noisy = latent + np.sqrt(.1)*torch.randn_like(latent)
    train = torch.tensor(np.random.RandomState(1).choice(len(L), int(.1*len(L)), replace=False))
    test = torch.tensor(np.setdiff1d(np.arange(len(L)), train.numpy()))
    center, scale = noisy[train].mean(), noisy[train].std()
    y = (noisy-center)/scale
    return L, train, test, y, dict(latent=latent, noisy=noisy, center=center, scale=scale)


def checks():
    L, train, test, y, raw = data(6)
    for a, b in zip((L, train, test, y), data(6)[:4]):
        assert torch.equal(a, b)
    eigenvalues, Q = torch.linalg.eigh(L)
    assert (L-L.T).abs().max() < 1e-14 and eigenvalues.min() > -1e-14 and eigenvalues.max() < 2+1e-14
    ids = torch.cat([train, test])
    reference = 10*(Q*torch.exp(-10*eigenvalues))@Q.T
    torch.testing.assert_close(reference, 10*torch.matrix_exp(-10*L), atol=2e-13, rtol=2e-12)
    kernels = [DiffusionExactKernel(L)]
    matrices = []
    for sampler in (GRFSampler, GRFAblationSampler):
        mats = sampler(L, 32, .01, 10, seed=1, use_tqdm=False, n_processes=4)()
        matrices.append(mats)
        kernels.append(GeneralGRFKernel(mats, 10))
    # Both upstream recursions consume identical draws; only the load changes.
    for a, b in zip(*matrices):
        assert torch.equal(a.sparse_csr_tensor.col_indices(), b.sparse_csr_tensor.col_indices())
    degree = L.count_nonzero(dim=1)
    torch.testing.assert_close(matrices[0][1].to_dense(), matrices[1][1].to_dense()*degree[:, None]/.99)
    errors = []
    for kernel in kernels:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGraphGP(train, y[train], likelihood, kernel) if kernel is kernels[0] else GRFGP(train, y[train], likelihood, kernel)
        candidate = kernel(ids).to_dense()
        if kernel is kernels[0]:
            expected = kernel.sigma_f.square()*(Q*torch.exp(-kernel.beta*eigenvalues))@Q.T
        else:
            phi = torch.stack([m.to_dense() for m in kernel.rw_mats])
            phi = (kernel.modulation_function[:, None, None]*phi).sum(dim=0)
            expected = phi@phi.T
        expected = expected[ids][:, ids]
        nll, mean, variance = posterior(candidate, y[train], likelihood.noise)
        observed = expected[:len(train), :len(train)] + likelihood.noise*torch.eye(len(train))
        reference_nll = -torch.distributions.MultivariateNormal(torch.zeros(len(train)), observed).log_prob(y[train])/len(train)
        mll = -gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)(model(train), y[train])
        params = tuple(model.parameters())
        gradients = torch.autograd.grad(mll, params, retain_graph=True)
        reference_gradients = torch.autograd.grad(reference_nll, params)
        for a, b in zip((candidate, nll, mll, *gradients), (expected, reference_nll, reference_nll, *reference_gradients)):
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-8)
        k = expected.detach().numpy()
        H = k[:len(train), :len(train)] + likelihood.noise.item()*np.eye(len(train))
        np.linalg.cholesky(H)
        cross = k[len(train):, :len(train)]
        mu = cross@np.linalg.solve(H, y[train].numpy())
        var = np.diag(k)[len(train):] - np.sum(cross*np.linalg.solve(H, cross.T).T, axis=1)
        np.testing.assert_allclose(mean.detach(), mu, atol=1e-9, rtol=1e-8)
        np.testing.assert_allclose(variance.detach(), var, atol=1e-9, rtol=1e-8)
        obs = var+likelihood.noise.item()
        nlpd = np.mean(.5*(np.log(2*np.pi*obs)+(y[test].numpy()-mu)**2/obs))
        assert abs(score(mean, variance, y[test], likelihood.noise, 1)[1]-nlpd) < 1e-9
        errors.append(dict(kernel=type(kernel).__name__, covariance_error=float((candidate-expected).detach().abs().max()),
            gradient_error=max(float((a-b).abs().max()) for a,b in zip(gradients,reference_gradients))))
    cheb = []
    for beta in (1.3132616875182228, 4., 10., 20., 40.):
        b = torch.tensor(beta, requires_grad=True)
        c = heat_coefficients(b)
        dc = torch.autograd.functional.jacobian(heat_coefficients, b)
        lam = np.linspace(0, 2, 20001)
        p = np.polynomial.chebyshev.chebval(lam-1, c.detach().numpy())
        derivative = np.polynomial.chebyshev.chebval(lam-1, dc.numpy())
        tiny = apply_chebyshev(L.to_sparse_coo(), c, torch.eye(len(L)))
        cheb.append(dict(beta=beta, error=float(np.max(abs(p-np.exp(-beta*lam)))),
            derivative_error=float(np.max(abs(derivative+lam*np.exp(-beta*lam)))),
            interval_min=float(p.min()), tiny_min=float(torch.linalg.eigvalsh(tiny.detach()).min())))
    save(OUT/"checks.json", dict(gates=errors, cheb_degree20=cheb, cheb_status="deferred: invalid unbounded beta regime"))
    print("PASS exact/GRF semantics and gradients; Cheb deferred", errors, cheb, flush=True)


def fit(stage, arm, L, train, test, y, raw, fingerprint, walkers, steps):
    path = OUT/f"{stage}_{arm}.json"
    checkpoint = path.with_suffix(".pt")
    if path.exists() and json.loads(path.read_text())["status"] == "complete":
        row = json.loads(path.read_text())
        assert row["data_sha"] == fingerprint and row["config"] == CONFIG and row["steps"] == steps and row["walkers"] == walkers
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row["checkpoint_sha"]
        print("SKIP verified", stage, arm, flush=True)
        return
    row = dict(status="running", arm=arm, stage=stage, data_sha=fingerprint, config=CONFIG, steps=steps, walkers=walkers)
    save(path, row)
    try:
        clock = perf_counter()
        if arm != "exact":
            sampler = GRFSampler if arm == "principled" else GRFAblationSampler
            mats = sampler(L, walkers, .01, 10, seed=1, use_tqdm=False, n_processes=4)()
            torch.save([m.sparse_csr_tensor for m in mats], OUT/f"{stage}_{arm}_walks.pt")
        row["sampling_s"] = perf_counter()-clock
        clock = perf_counter()
        torch.manual_seed(1)
        kernel = DiffusionExactKernel(L) if arm == "exact" else GeneralGRFKernel(mats, 10)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGraphGP(train, y[train], likelihood, kernel) if arm == "exact" else GRFGP(train, y[train], likelihood, kernel)
        initial = {name: p.detach().tolist() for name,p in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        row["setup_s"] = perf_counter()-clock
        losses = []
        clock = perf_counter()
        for step in range(steps):
            optimizer.zero_grad()
            loss = -mll(model(train), y[train])
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            losses.append(loss.item())
            if (step+1) % 100 == 0:
                print(stage, arm, step+1, loss.item(), flush=True)
        row["train_s"] = perf_counter()-clock
        clock = perf_counter()
        with torch.no_grad():
            nll, mean, var = posterior(kernel(torch.cat([train, test])).to_dense(), y[train], likelihood.noise)
            rmse, nlpd = score(mean, var, y[test], likelihood.noise, float(raw["scale"]))
        row.update(posterior_s=perf_counter()-clock, nll=float(nll), rmse=rmse, rmse_norm=rmse/float(raw["scale"]),
            nlpd_norm=nlpd, nlpd_raw=nlpd+float(raw["scale"].log()), noise_variance=float(likelihood.noise.detach()),
            initial=initial, final={name:p.detach().tolist() for name,p in model.named_parameters()},
            rss_self_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, rss_child_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        torch.save(dict(state=model.state_dict(), optimizer=optimizer.state_dict(), losses=losses, mean=mean, latent_variance=var), checkpoint)
        row.update(checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), status="complete")
        print(row, flush=True)
    except BaseException as exc:
        row.update(status="failed", error=repr(exc))
        raise
    finally:
        save(path, row)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(7200)
    stage = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    assert stage in ("check", "smoke", "pilot", "full")
    pins = [subprocess.check_output(["git", "-C", str(p), "rev-parse", "HEAD"], text=True).strip() for p in (CORE, UPSTREAM)]
    assert pins == [CORE_COMMIT, EXPERIMENT_COMMIT]
    for p in (CORE, UPSTREAM):
        subprocess.run(["git", "-C", str(p), "diff", "--exit-code"], check=True)
    print(CONFIG, Path(__file__).with_name("GRF_PAPER_C3_REPLICATION.md").read_text().split("## Numerical gates")[0], flush=True)
    with gpytorch.settings.max_cholesky_size(1000), gpytorch.settings.fast_computations(log_prob=False):
        checks()
        if stage != "check":
            side, walkers, steps = (6, 32, 5) if stage == "smoke" else (30, 256, 20) if stage == "pilot" else (30, 10000, 1000)
            clock = perf_counter()
            L, train, test, y, raw = data(side)
            fingerprint = hashlib.sha256(b"".join(a.numpy().tobytes() for a in (L,train,test,y,*raw.values()))).hexdigest()
            torch.save(dict(L=L, train=train, test=test, y=y, **raw), OUT/f"{stage}_data.pt")
            save(OUT/f"{stage}_manifest.json", dict(config=CONFIG, data_sha=fingerprint, pins=pins, data_s=perf_counter()-clock,
                source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), torch=torch.__version__, numpy=np.__version__))
            for arm in ("exact", "principled", "ablated"):
                fit(stage, arm, L, train, test, y, raw, fingerprint, walkers, steps)
