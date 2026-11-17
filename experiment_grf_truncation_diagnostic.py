"""Post-hoc cutoff-3 traffic diagnostic; --check, then --pilot, then full run."""

import csv
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import gpytorch
import numpy as np
import torch
from torch.nn.functional import softplus

from experiment_grf_heldout import OUT as PRIOR, prepare, sha, walks
from experiment_grf_frontier import BoundedDiffusionGRFKernel
from experiment_grf_traffic import (
    CORE, CORE_COMMIT, EXPERIMENT_COMMIT, OUTPUT, UPSTREAM, GRFGP, GRFSampler, HeatGP,
    SparseLinearOperator, chebyshev_kernel, get_normalized_laplacian, posterior, score, train_model,
)

OUT = OUTPUT / "truncation_diagnostic"


def exact_powers(laplacian: torch.Tensor) -> list[SparseLinearOperator]:
    # Upstream Phi=sum sigma*(-beta/2)^k/k!*M_k: I,L,L² give sigma*Q2,
    # so its existing Phi@Phi.T is precisely sigma²*Q2@Q2.T.
    return [SparseLinearOperator(t.to_sparse_csr()) for t in
            [torch.eye(len(laplacian)), laplacian, laplacian @ laplacian]]


def parameters(model):
    kernel = model.covar_module
    if isinstance(model, HeatGP):
        raw = [kernel.base_kernel.raw_heat_time, kernel.raw_outputscale, model.likelihood.raw_noise]
        physical = [kernel.base_kernel.heat_time, kernel.outputscale.sqrt(), model.likelihood.noise]
    else:
        raw = [kernel.raw_beta, kernel.raw_sigma_f, model.likelihood.raw_noise]
        physical = [kernel.beta, kernel.sigma_f, model.likelihood.noise]
    return raw, [float(v.detach()) for v in physical]


def checks(graph, ids, y, label, sampled, records):
    torch.testing.assert_close(graph, graph.T, atol=1e-13, rtol=1e-13)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel = BoundedDiffusionGRFKernel(exact_powers(graph), 3)
    model = GRFGP(ids[:len(y)], y, likelihood, kernel)
    raw, _ = parameters(model)
    for value in [.4, 1.0, 1.9]:
        with torch.no_grad():
            kernel.raw_beta.copy_(torch.logit(torch.tensor(value/2)))
        beta, sigma = 2*raw[0].sigmoid(), softplus(raw[1])
        noise = softplus(raw[2])+1e-4  # Independent expression for default likelihood variance.
        q = torch.eye(len(graph))-beta/2*graph+beta.square()/8*(graph @ graph)
        full = sigma.square()*(q @ q.T)
        reference, actual = full[ids][:, ids], kernel(ids).to_dense()
        noisy = reference[:len(y), :len(y)]+noise*torch.eye(len(y))
        reference_nll = -torch.distributions.MultivariateNormal(torch.zeros_like(y), noisy).log_prob(y)/len(y)
        loss = -gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)(model(ids[:len(y)]), y)
        gradients = torch.autograd.grad(loss, raw)
        expected = torch.autograd.grad(reference_nll, raw)
        row = dict(graph=label, beta=value, covariance_error=float((actual-reference).detach().abs().max()),
                   nll_error=float((loss-reference_nll).detach()),
                   gradient_errors=[float((a-b).abs().max()) for a, b in zip(gradients, expected)],
                   latent_min_eigenvalue=float(torch.linalg.eigvalsh(full.detach()).min()),
                   symmetry_error=float((actual-actual.T).detach().abs().max()),
                   noisy_cholesky_min_diagonal=float(torch.linalg.cholesky(noisy.detach()).diagonal().min()))
        with torch.no_grad():
            heat = sigma.square()*torch.matrix_exp(-beta*graph)[ids][:, ids]
            row["trunc_vs_heat"] = float((reference-heat).norm()/heat.norm())
            row["finite_walks"] = {}
            for seed, matrices in sampled.items():
                finite = BoundedDiffusionGRFKernel(matrices, 3)
                finite.load_state_dict(kernel.state_dict())
                covariance = finite(ids).to_dense()
                row["finite_walks"][seed] = [float((covariance-reference).norm()/reference.norm()),
                                             float((covariance-heat).norm()/heat.norm())]
        records.append(row)
        (OUT / "validation.json").write_text(json.dumps(records, indent=2))
        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(loss, reference_nll, atol=1e-11, rtol=1e-10)
        for actual_gradient, reference_gradient in zip(gradients, expected):
            torch.testing.assert_close(actual_gradient, reference_gradient, atol=1e-10, rtol=1e-9)
        assert row["latent_min_eigenvalue"] > 0 and row["symmetry_error"] < 1e-12
        assert np.isfinite([row["covariance_error"], row["nll_error"], *row["gradient_errors"]]).all()
        print("PASS", row, flush=True)


def fit(date, arm, seed, data, matrices, walk_s, prior_rows):
    path = OUT / f"fit_{int(data['indices'][date])}_{arm}_{seed}.pt"
    result = path.with_suffix(".json")
    data_sha = sha((PRIOR / "data.npz").read_bytes())
    if result.exists():
        old = json.loads(result.read_text())
        if old["status"] == "complete":
            assert old["data_sha"] == data_sha and old["steps"] == 1000
            assert old["checkpoint_sha"] == sha(Path(old["source"]).read_bytes())
            print("SKIP verified complete", path.name, flush=True)
            return
    train, test = torch.from_numpy(data["train"]), torch.from_numpy(data["test"])
    y, yt = torch.from_numpy(data["y"][date]), torch.from_numpy(data["yt"][date])
    clock = perf_counter()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel = chebyshev_kernel(torch.from_numpy(data["laplacian"])) if arm == "Cheb" else BoundedDiffusionGRFKernel(matrices, 3)
    model = HeatGP(train, y, likelihood, kernel) if arm == "Cheb" else GRFGP(train, y, likelihood, kernel)
    setup_s = perf_counter()-clock
    raw, initial = parameters(model)
    np.testing.assert_array_equal([float(p.detach()) for p in raw], [1., 1., 0.])
    torch.manual_seed(0)
    initial_loss = -gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)(model(train), y)
    previous = PRIOR / path.name
    old = next((r for r in prior_rows if r["timestamp"] == str(data["timestamps"][date]) and
                r["arm"] == arm and int(r["feature_seed"]) == seed and r["status"] == "complete"), None)
    source = previous if arm != "Trunc" and previous.exists() and old is not None else path
    row = dict(date=str(data["timestamps"][date]), arm=arm, seed=seed, status="incomplete",
               raw_initial=[1., 1., 0.], initial_physical=initial, initial_nll=float(initial_loss.detach()),
               source=str(source), walk_s=walk_s, reused=source.exists(), data_sha=data_sha)
    result.write_text(json.dumps(row, indent=2))
    if source.exists():
        saved = torch.load(source, weights_only=True)
        assert len(saved["losses"]) == 1000 and np.isfinite(saved["losses"]).all()
        np.testing.assert_allclose(saved["losses"][0], row["initial_nll"], atol=1e-11, rtol=1e-11)
        model.load_state_dict(saved["model"])
    else:
        clock = perf_counter()
        losses = train_model(model, likelihood, train, y, lr=.05, max_iter=1000, print_every=250, progress_desc=arm)
        saved = dict(model=model.state_dict(), losses=losses, setup_s=setup_s, train_s=perf_counter()-clock)
        assert len(losses) == 1000 and np.isfinite(losses).all()
        torch.save(saved, path)
    model.eval()
    raw, physical = parameters(model)
    with torch.no_grad():
        clock = perf_counter()
        nll, mean, variance = posterior(kernel(torch.cat([train, test])).to_dense(), y, likelihood.noise)
        evaluation_s = perf_counter()-clock
        rmse, nlpd = score(mean, variance, yt, likelihood.noise, float(data["std"][date]))
    row.update(steps=len(saved["losses"]), raw_final=[float(p.detach()) for p in raw],
               beta=physical[0], sigma=physical[1], noise_variance=physical[2], nll=float(nll),
               rmse_mph=rmse, nlpd=nlpd, nlpd_mph=nlpd+float(np.log(data["std"][date])),
               last_optimizer_loss=saved["losses"][-1], setup_s=saved["setup_s"], train_s=saved["train_s"],
               posterior_s=evaluation_s, checkpoint_sha=sha(source.read_bytes()), latent_min=float(variance.min()))
    if source == previous:
        np.testing.assert_allclose([nll, rmse, nlpd, *physical],
                                   [float(old[k]) for k in ["kernel_nll", "oracle_rmse", "oracle_nlpd", "beta", "sigma_f", "noise"]], atol=1e-11, rtol=1e-11)
    assert np.isfinite([row[k] for k in ["nll", "rmse_mph", "nlpd", "beta", "sigma", "noise_variance"]]).all()
    row["status"] = "complete"
    torch.save(dict(mean=mean, latent_variance=variance), path.with_name(path.stem+"_posterior.pt"))
    result.write_text(json.dumps(row, indent=2))
    print("COMPLETE", row, flush=True)


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    assert gpytorch.settings.max_cholesky_size.value() == 800
    for path, commit in [(CORE, CORE_COMMIT), (UPSTREAM, EXPERIMENT_COMMIT)]:
        assert subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() == commit
    data = prepare()
    graph = torch.from_numpy(data["laplacian"])
    assert graph.shape == (1016, 1016) and len(data["train"]) == 250 and len(data["test"]) == 75
    clock = perf_counter()
    deterministic = exact_powers(graph)
    power_s = perf_counter()-clock
    sampled = {seed: walks("C", seed, graph) for seed in [101, 202]}
    manifest = dict(base_main="2dbf18d850ebb07618e43f3ff2cf7870740ea0cf", pr8="open/unmerged at start; independent branch",
                    protocol="post-hoc diagnostic; AdamW .05 defaults / 1000 updates / float64 / CPU4 / beta in (0,2)",
                    prior_manifest=json.loads((PRIOR / "manifest.json").read_text()), shared_exact_power_s=power_s,
                    source_sha=sha(Path(__file__).read_bytes()))
    stage = "check" if "--check" in sys.argv else ("pilot" if "--pilot" in sys.argv else "full")
    (OUT / f"{stage}_manifest.json").write_text(json.dumps(manifest, indent=2))
    weights = np.array([[0., 1., 0., 0.], [1., 0., 2., 0.],
                        [0., 2., 0., 3.], [0., 0., 3., 0.]])
    tiny = torch.from_numpy(get_normalized_laplacian(weights))
    tiny_walks = {}
    for seed in [101, 202]:
        sampler = GRFSampler(tiny, 4096, .5, 3, seed=seed, use_tqdm=False, n_processes=4)
        tiny_walks[seed] = sampler.sample_random_walk_matrices()
    records = []
    checks(tiny, torch.tensor([3, 0, 2, 1]), torch.tensor([.3, -.6]), "tiny", tiny_walks, records)
    checks(graph, torch.from_numpy(np.r_[data["train"], data["test"]]), torch.from_numpy(data["y"][0]), "traffic", {s: v[0] for s, v in sampled.items()}, records)
    if "--check" in sys.argv:
        return
    table = PRIOR / "rows.csv"
    prior_rows = list(csv.DictReader(table.open())) if table.exists() else []
    for date in ([0] if "--pilot" in sys.argv else range(3)):
        for arm, seed in [("Cheb", 0), ("Trunc", 0), ("C", 101), ("C", 202)]:
            matrices, cost = (deterministic, 0.) if arm != "C" else sampled[seed]
            fit(date, arm, seed, data, matrices, cost, prior_rows)


if __name__ == "__main__":
    main()
