"""Fixed learned-kernel scaling protocol; numerical qualification comes first."""

import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import gpytorch
import numpy as np
import scipy.sparse as sp
import torch
from experiment_grf_large_synthetic import graph_data
from experiment_speed_accuracy import controls
from experiment_grf_heldout import sha
from experiment_learned_polynomial_traffic import model_for, polynomial_kernel, check_kernel
from experiment_polynomial_optimizer import checks as kernel_checks
from experiment_grf_traffic import CORE, CORE_COMMIT, UPSTREAM, EXPERIMENT_COMMIT, GRFSampler, SparseLinearOperator, posterior

OUT = Path(__file__).resolve().parent / "results/learned_kernel_scaling"
ARMS = [("Heat", 0), ("Poly", 0), ("General", 1024), ("General", 4096)]


def inputs(n, replicate):
    directory = OUT/f"{'development' if replicate == 9 else 'final'}/n{n}_r{replicate}"
    directory.mkdir(parents=True, exist_ok=True)
    seeds = tuple(base+n+100000*replicate for base in (731000, 741000, 751000))
    graph, _, train, test, metadata = graph_data(n, seeds=seeds, output=directory)
    L = sp.load_npz(directory/f"laplacian_{n}.npz")
    with np.load(directory/f"data_{n}.npz") as saved:
        z2 = np.random.default_rng(761000+n+100000*replicate).standard_normal(n)
        latent = np.stack([saved["f"], (saved["f"]+L@z2/2)/np.sqrt(2)])
        observations = latent+(saved["y"]-saved["f"])[None, :]
        mean, scale = observations[:, train.numpy()].mean(1), observations[:, train.numpy()].std(1)
        targets = (observations-mean[:, None])/scale[:, None]
        fingerprint = sha(L.data.tobytes()+L.indices.tobytes()+L.indptr.tobytes()+targets.tobytes()+train.numpy().tobytes()+test.numpy().tobytes())
        values = dict(z2=z2, latent=latent, observations=observations, normalized=targets, mean=mean, scale=scale)
    path = directory/"regimes.npz"
    if path.exists():
        with np.load(path) as saved:
            for key, value in values.items():
                np.testing.assert_array_equal(saved[key], value)
    else:
        np.savez(path, **values)
    np.testing.assert_allclose(targets[:, train.numpy()].mean(1), 0, atol=1e-13)
    np.testing.assert_allclose(targets[:, train.numpy()].std(1), 1, atol=1e-13)
    assert not np.intersect1d(train.numpy(), test.numpy()).size
    return graph, torch.from_numpy(targets), train, test, scale, directory, fingerprint, metadata


def construct(graph, train, target, arm, walks, directory, replicate):
    matrices, walk_s = None, 0.
    if arm == "General":
        path = directory/f"walks_{walks}.pt"
        if not path.exists():
            clock = perf_counter()
            sampler = GRFSampler(graph, walks, .5, 3, seed=771000+len(target)+100000*replicate, use_tqdm=False, n_processes=4)
            matrices = sampler.sample_random_walk_matrices()
            torch.save(dict(matrices=[m.sparse_csr_tensor for m in matrices], walk_s=perf_counter()-clock,
                seed=sampler.seed, W=walks, J=3, halt=.5, workers=4), path)
        saved = torch.load(path, weights_only=True)
        assert (saved["seed"],saved["W"],saved["J"],saved["halt"],saved["workers"]) == (771000+len(target)+100000*replicate,walks,3,.5,4)
        matrices, walk_s = [SparseLinearOperator(m) for m in saved["matrices"]], saved["walk_s"]
    torch.manual_seed(0)
    model = model_for("Cheb" if arm == "Heat" else arm, graph, matrices, train, target[train])
    return model, walk_s


def second_state(model, arm):
    kernel = model.covar_module
    model.likelihood.noise = .05
    if arm == "General":
        kernel.initialize(raw_modulation_function=torch.tensor([1., -.35-4*.45, 2*.45])*1.5)
    else:
        kernel.outputscale = 1.5**2
        if arm == "Poly":
            kernel.base_kernel.initialize(a1=-.35, a2=.45)
        else:
            kernel.base_kernel.initialize(raw_heat_time=kernel.base_kernel.raw_heat_time_constraint.inverse_transform(torch.tensor(1.8)))


def residual(model, train, targets):
    alpha = model.prediction_strategy.mean_cache
    error = model.covar_module(train)@alpha+model.likelihood.noise*alpha-targets[train]
    return float((error.norm()/targets[train].norm()).detach())


def tiny():
    graph, targets, train, test, _, directory, _, _ = inputs(36, 9)
    L = graph.to_dense()
    torch.testing.assert_close(L, L.T, atol=1e-13, rtol=1e-13)
    eigenvalues = torch.linalg.eigvalsh(L)
    assert eigenvalues.min() > -1e-12 and eigenvalues.max() < 2+1e-12
    with np.load(directory/"data_36.npz") as original, np.load(directory/"regimes.npz") as data:
        heat = torch.matrix_exp(-.75*L)@torch.from_numpy(original["z"])
        multiscale = (heat+L@torch.from_numpy(data["z2"])/2)/np.sqrt(2)
        np.testing.assert_allclose(data["latent"], torch.stack([heat,multiscale]).numpy(), atol=1e-12, rtol=1e-11)
        np.testing.assert_allclose(data["observations"]-data["latent"], np.broadcast_to(original["y"]-original["f"],(2,36)), atol=1e-15, rtol=1e-13)
    check_kernel(polynomial_kernel(graph))
    general, _ = construct(graph,train,targets[0],"General",4096,directory,9)
    diagnostics = kernel_checks(L,{303:general.covar_module.rw_mats,404:general.covar_module.rw_mats})
    heat, _ = construct(graph,train,targets[0],"Heat",0,directory,9)
    poly, _ = construct(graph,train,targets[0],"Poly",0,directory,9)
    with controls(dict(degree=6,cg=1e-4,probes=64,lanczos=24)):
        exact = heat.covar_module.outputscale*torch.matrix_exp(-heat.covar_module.base_kernel.heat_time*L)
        error = float((heat.covar_module(torch.arange(36)).to_dense()-exact).detach().abs().max())
        assert error <= 7e-5*float(heat.covar_module.outputscale.detach())
        discrepancy = float((poly.covar_module(torch.arange(36)).to_dense()-general.covar_module(torch.arange(36)).to_dense()).detach().norm()/poly.covar_module(torch.arange(36)).to_dense().detach().norm())
    (OUT/"tiny.json").write_text(json.dumps(dict(diagnostics=diagnostics, heat_max_error=error, initial_poly_general_relative=discrepancy,
        checker_note="303/404 helper keys alias the same prescribed tiny draw; not independent replicates",walk_seed=771000+36+900000,
        parameter_counts={arm:sum(p.numel() for p in model.parameters()) for arm,model in [("Heat",heat),("Poly",poly),("General",general)]},
        initial={arm:{n:p.detach().tolist() for n,p in model.named_parameters()} for arm,model in [("Heat",heat),("Poly",poly),("General",general)]}),indent=2))
    print("PASS tiny graph, same-vector signals, independent kernels/gradients; Heat error",error,flush=True)


def qualify():
    graph, targets, train, test, _, directory, fingerprint, _ = inputs(1000,9)
    rows = []
    for arm,walks in ARMS:
        for regime in range(2):
            model, _ = construct(graph,train,targets[regime],arm,walks,directory,9)
            for state in range(2):
                if state:
                    second_state(model,arm)
                for probes,lanczos in [(32,16),(64,24)]:
                    with controls(dict(degree=6,cg=1e-4,probes=probes,lanczos=lanczos)):
                        reference, exact_mean, _ = posterior(model.covar_module(torch.cat([train,test])).to_dense(),targets[regime,train],model.likelihood.noise)
                        names = [f"{name}[{i}]" for name,p in model.named_parameters() for i in range(p.numel())]
                        gradients = torch.cat([g.flatten() for g in torch.autograd.grad(reference,tuple(model.parameters()))]).detach()
                        for seed in (71,72,73):
                            model.train()
                            torch.manual_seed(seed)
                            loss = -gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood,model)(model(train),targets[regime,train])
                            estimated = torch.cat([g.flatten() for g in torch.autograd.grad(loss,tuple(model.parameters()))]).detach()
                            model.eval()
                            with torch.no_grad(),gpytorch.settings.skip_posterior_variances(True):
                                mean = model(test).mean
                                solve_residual = residual(model,train,targets[regime])
                            errors = estimated-gradients
                            row = dict(arm=arm,W=walks,regime=regime,state=state,probes=probes,lanczos=lanczos,trace_seed=seed,
                                nll_error=float(loss.detach()-reference.detach()), names=names,reference_gradients=gradients.tolist(),
                                estimated_gradients=estimated.tolist(),gradient_errors=errors.tolist(),residual=solve_residual,
                                mean_max_error=float((mean-exact_mean.detach()).abs().max()),data_fingerprint=fingerprint)
                            row["passed"] = abs(row["nll_error"]) <= .02 and errors.abs().max().item() <= .02 and solve_residual <= 2e-4 and row["mean_max_error"] <= .002 and bool(((estimated*gradients >= 0)|(gradients.abs() <= .02)).all())
                            rows.append(row)
                            (OUT/"qualification.json").write_text(json.dumps(rows,indent=2))
                            print("GATE",arm,walks,regime,state,probes,seed,row["passed"],flush=True)
    gates = []
    for arm,walks in ARMS:
        for probes,lanczos in [(32,16),(64,24)]:
            chosen = [r for r in rows if (r["arm"],r["W"],r["probes"]) == (arm,walks,probes)]
            gates.append(dict(arm=arm,W=walks,probes=probes,lanczos=lanczos,passed=len(chosen) == 12 and all(r["passed"] for r in chosen)))
    eligible = [p for p in (32,64) if all(g["passed"] for g in gates if g["probes"] == p and g["W"] != 1024)]
    (OUT/"gates.json").write_text(json.dumps(dict(gates=gates,common_probes=eligible[0] if eligible else None),indent=2))
    assert eligible, "Neither prescribed common budget qualifies; stop before development/final fits"


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    for path,revision in ((CORE,CORE_COMMIT),(UPSTREAM,EXPERIMENT_COMMIT)):
        assert subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip() == revision
        subprocess.run(["git","-C",str(path),"diff","--exit-code"],check=True)
    sources = [Path(__file__).name,"experiment_grf_large_synthetic.py","experiment_speed_accuracy.py","experiment_learned_polynomial_traffic.py",
        "experiment_polynomial_optimizer.py","learned_polynomial.py","chebyshev.py","graph_gp.py","dense_gp.py","experiment_grf_traffic.py"]
    protocol = dict(base="6554aaef34704d2a4d49a5012e432de1bd15ce1b",work_order_sha=sha(Path("/Users/josephortega/Downloads/CODEX_LEARNED_KERNEL_SCALING_STUDY.md").read_bytes()),
        sources={s:sha(Path(s).read_bytes()) for s in sources},packages=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True).splitlines())
    path = OUT/"protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol, "Source/config/environment changed; preserve existing run"
    else:
        path.write_text(json.dumps(protocol,indent=2)+"\n")
    with gpytorch.settings.max_cg_iterations(1000):
        if sys.argv[1] == "tiny":
            tiny()
        elif sys.argv[1] == "qualify":
            assert (OUT/"tiny.json").exists()
            qualify()
        else:
            raise ValueError("Run tiny, then qualify before development/scaling")
