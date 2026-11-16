"""Paper-first C.2 sparse rings; check, then explicit N/seed cases. See report."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
from time import perf_counter
import warnings

import gpytorch
import numpy as np
import torch
from experiment_grf_large_synthetic import SparseGuard
from experiment_grf_traffic import CORE, CORE_COMMIT, UPSTREAM, EXPERIMENT_COMMIT, GRFSampler, GRFGP, train_model
from grf_gp.kernels.general import GeneralGRFKernel

OUT = Path(__file__).resolve().parent / "results/paper_c2"
CONFIG = dict(walks=100, halt=.1, cutoff=3, noise_variance=.1, fraction=.6, steps=50,
    optimizer="official AdamW", lr=.05, weight_decay=.01, cg=.01, lanczos=1, probes=64,
    max_cg=1000, cholesky=0, preconditioner=0, predictive_paths=64, threads=4, workers=4,
    dtype="float64", device="cpu", limit_seconds=1800, parent_rss_limit_gib=12)


@contextmanager
def controls(cholesky=0):
    with (gpytorch.settings.max_cholesky_size(cholesky), gpytorch.settings.max_eager_kernel_size(0),
          gpytorch.settings.cg_tolerance(.01), gpytorch.settings.eval_cg_tolerance(.01),
          gpytorch.settings.num_trace_samples(64), gpytorch.settings.max_lanczos_quadrature_iterations(1),
          gpytorch.settings.max_cg_iterations(1000), gpytorch.settings.max_preconditioner_size(0),
          gpytorch.settings.min_preconditioning_size(10**10),
          gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=not cholesky)):
        yield


def data(n, seed):
    nodes = torch.arange(n)
    rows = nodes.repeat_interleave(2)
    columns = torch.stack(((nodes-1) % n, (nodes+1) % n), dim=1).flatten()
    adjacency = torch.sparse_coo_tensor(torch.stack((rows, columns)), torch.ones(2*n), (n,n)).coalesce().to_sparse_csr()
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    truth = 2*np.sin(2*angles)+.5*np.cos(4*angles)+.3*np.sin(angles)
    raw = torch.from_numpy(truth+rng.normal(0, np.sqrt(.1), n))
    train = torch.from_numpy(rng.choice(n, int(.6*n), replace=False))
    test = torch.from_numpy(np.setdiff1d(np.arange(n), train.numpy()))
    center, scale = raw[train].mean(), raw[train].std()
    y = (raw-center)/scale
    fingerprint = hashlib.sha256(b"".join(t.numpy().tobytes() for t in
        (adjacency.crow_indices(), adjacency.col_indices(), adjacency.values(), raw, train, test))).hexdigest()
    return adjacency, y, train, test, dict(fingerprint=fingerprint, raw=raw, truth=truth, center=center, scale=scale)


def checks():
    adjacency, y, train, test, meta = data(32, 0)
    independent = torch.eye(32).roll(1,0)+torch.eye(32).roll(-1,0)
    torch.testing.assert_close(adjacency.to_dense(), independent, atol=0, rtol=0)
    assert torch.equal(independent, independent.T) and torch.equal(independent.sum(1), torch.full((32,),2.))
    assert adjacency._nnz() == 64 and meta["fingerprint"] == data(32,0)[4]["fingerprint"]
    sparse = GRFSampler(adjacency, 100, .1, 3, seed=0, n_processes=4, use_tqdm=False)()
    dense = GRFSampler(independent, 100, .1, 3, seed=0, n_processes=4, use_tqdm=False)()
    for a,b in zip(sparse,dense):
        torch.testing.assert_close(a.to_dense(), b.to_dense(), atol=0, rtol=0)
    torch.manual_seed(0)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel = GeneralGRFKernel(sparse, 3)
    model = GRFGP(train, y[train], likelihood, kernel)
    phi = sum(f*r.to_dense() for f,r in zip(kernel.modulation_function,sparse))
    reference = phi@phi.T
    ids = torch.tensor([4,0,0,17])
    torch.testing.assert_close(kernel(ids).to_dense(), reference[ids][:,ids], atol=1e-12, rtol=1e-11)
    rhs = torch.arange(32.)[:,None]/32
    torch.testing.assert_close(kernel(torch.arange(32))@rhs, reference@rhs, atol=1e-12, rtol=1e-11)
    observed = reference[train][:,train]+likelihood.noise*torch.eye(len(train))
    exact = -torch.distributions.MultivariateNormal(torch.zeros(len(train)),observed).log_prob(y[train])/len(train)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,model)
    with controls(1000):
        candidate = -mll(model(train),y[train])
        gradients = torch.autograd.grad(candidate,tuple(model.parameters()))
    expected = torch.autograd.grad(exact,tuple(model.parameters()))
    torch.testing.assert_close(candidate,exact,atol=1e-12,rtol=1e-11)
    for a,b in zip(gradients,expected):
        torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-10)
    with controls():
        torch.manual_seed(0)
        approximate = -mll(model(train),y[train])
        stochastic_gradients = torch.autograd.grad(approximate,tuple(model.parameters()))
        H = kernel(train).evaluate_kernel().add_diagonal(likelihood.noise.expand(len(train)))
        alpha = H.solve(y[train])
    result = dict(passed=True, config=CONFIG, source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,gpytorch=gpytorch.__version__,
        core=CORE_COMMIT,experiments=EXPERIMENT_COMMIT,exact_nll_error=float((candidate-exact).detach().abs()),
        exact_gradient_error=max(float((a-b).abs().max()) for a,b in zip(gradients,expected)),
        authors_nll_error=float((approximate-exact).detach()),
        authors_gradient_errors=[float((a-b).abs().max()) for a,b in zip(stochastic_gradients,expected)],
        true_relative_residual=float((H@alpha-y[train]).detach().norm()/y[train].norm()),
        walk_vs_power_relative=[float((r.to_dense()-torch.linalg.matrix_power(independent,k)).norm()/torch.linalg.matrix_power(independent,k).norm()) for k,r in enumerate(sparse)])
    (OUT/"checks.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)


def timeout(signum, frame):
    raise TimeoutError("1800-second case deadline")


def run(n, seed):
    start = perf_counter()
    path = OUT/f"{n}_{seed}.json"
    row = dict(n=n,seed=seed,effective_walk_seed=seed or 42,status="incomplete",config=CONFIG,
        source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    messages = []
    try:
        with warnings.catch_warnings(record=True) as messages, controls(), SparseGuard(n,int(.6*n)):
            clock = perf_counter()
            adjacency,y,train,test,meta = data(n,seed)
            row.update(preparation_s=perf_counter()-clock,data_sha=meta["fingerprint"],train=len(train),test=len(test),
                train_mean=float(meta["center"]),train_std=float(meta["scale"]))
            if path.exists():
                prior = json.loads(path.read_text())
                if prior["status"] == "complete" and prior["config"] == CONFIG and prior["data_sha"] == row["data_sha"]:
                    assert prior["source_sha"] == row["source_sha"] and prior["steps"] == len(prior["losses"]) == 50
                    print("SKIP verified",n,seed,flush=True)
                    return
            torch.manual_seed(seed)
            clock = perf_counter()
            matrices = GRFSampler(adjacency,100,.1,3,seed=seed,n_processes=4,use_tqdm=False)()
            row["sampling_s"] = perf_counter()-clock
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 12*2**30
            tensors = [t for r in matrices for t in (r.sparse_csr_tensor.values(),r.sparse_csr_tensor.crow_indices(),r.sparse_csr_tensor.col_indices())]
            row.update(feature_bytes=sum(t.numel()*t.element_size() for t in tensors),nnz=sum(r.sparse_csr_tensor._nnz() for r in matrices),
                walks_sha=hashlib.sha256(b"".join(t.numpy().tobytes() for t in tensors)).hexdigest())
            clock = perf_counter()
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            kernel = GeneralGRFKernel(matrices,3)
            model = GRFGP(train,y[train],likelihood,kernel)
            row["setup_s"] = perf_counter()-clock
            row.update(initial_modulation=kernel.modulation_function.detach().tolist(),initial_noise=likelihood.noise.item())
            clock = perf_counter()
            losses = train_model(model,likelihood,train,y[train],lr=.05,max_iter=50,print_every=10)
            row["training_s"] = perf_counter()-clock
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 12*2**30
            assert np.isfinite(losses).all() and all(torch.isfinite(p).all() for p in model.parameters())
            row.update(losses=losses,steps=len(losses))
            model.eval()
            likelihood.eval()
            clock = perf_counter()
            with torch.no_grad():
                prediction = model.predict(test,batch_size=64)
                mean = prediction.mean.detach().cpu()
            row["inference_mean_s"] = perf_counter()-clock
            assert torch.isfinite(mean).all()
            clock = perf_counter()
            model.train()
            with torch.no_grad():
                row["final_stochastic_nll"] = -gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,model)(model(train),y[train]).item()
                assert np.isfinite(row["final_stochastic_nll"])
                H = kernel(train).evaluate_kernel().add_diagonal(likelihood.noise.expand(len(train)))
                alpha = H.solve(y[train])
                row["true_relative_residual"] = float((H@alpha-y[train]).norm()/y[train].norm())
            row.update(audit_s=perf_counter()-clock,noise_variance=likelihood.noise.item(),modulation=kernel.modulation_function.detach().tolist())
            np.savez(OUT/f"data_{n}_{seed}.npz",raw=meta["raw"].numpy(),truth=meta["truth"],train=train.numpy(),test=test.numpy())
        row.update(status="complete",parent_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30,
            child_rss_gib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/2**30)
        assert row["parent_rss_gib"] <= 12
    except BaseException as exc:
        row.update(status="failed",error=repr(exc))
        raise
    finally:
        if row["status"] != "incomplete":
            row.update(total_s=perf_counter()-start,warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in messages])
            path.write_text(json.dumps(row,indent=2,allow_nan=False))
            print(row,flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(1800)
    for source,revision in ((CORE,CORE_COMMIT),(UPSTREAM,EXPERIMENT_COMMIT)):
        assert subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip() == revision
        subprocess.run(["git","-C",str(source),"diff","--exit-code"],check=True)
    if sys.argv[1:] == ["check"]:
        checks()
    else:
        gate = json.loads((OUT/"checks.json").read_text())
        assert gate["passed"] and gate["config"] == CONFIG and gate["source_sha"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        n,seed = map(int,sys.argv[1:])
        assert n in (1024,4096,16384,32768,65536) and seed in range(5)
        run(n,seed)
