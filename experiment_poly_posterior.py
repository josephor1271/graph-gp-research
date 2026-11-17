"""Development-only qualification of centered squared-polynomial posterior paths."""

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch
import warnings

import numpy as np
import torch
from linear_operator.utils.linear_cg import linear_cg
from linear_operator.utils.warnings import NumericalWarning
from chebyshev import apply_chebyshev
from graph_gp import HeatOperator
from experiment_grf_large_synthetic import SparseGuard
from experiment_grf_traffic import posterior, score
import experiment_learned_kernel_scaling as prior
import experiment_learned_kernel_numerical_learning as learning

OUT = Path(__file__).resolve().parent/"results/poly_posterior_qualification"
SCHEDULE = [(36,9,s) for s in ("initial","stress")]+[(1000,r,s) for r in (10,11) for s in ("initial","stress","fitH","fitM")]


@contextmanager
def sparse(n, m, t):
    guard = SparseGuard(n,m)
    guard.forbidden |= {(t,m),(m,t),(t,t),(m+t,m+t),(m+t,n),(n,m+t),(m,m+t),(m+t,m)}
    with guard, patch.object(HeatOperator,"_get_indices",side_effect=AssertionError("Exact-entry fallback")), \
            patch.object(HeatOperator,"_diagonal",side_effect=AssertionError("Diagonal fallback")):
        yield


def load(n, replicate, state):
    with patch.object(prior,"OUT",OUT):
        graph,targets,train,test,scales,directory,fingerprint,_ = prior.inputs(n,replicate)
    regime = int(state == "fitM")
    model,_ = prior.construct(graph,train,targets[regime],"Poly",0,directory,replicate)
    consumed = {}
    if n == 1000:
        protocol = json.loads((learning.OUT/"protocol.json").read_text())
        assert protocol["data"][str(replicate)]["fingerprint"] == fingerprint
        for name in ("laplacian_1000.npz","data_1000.npz","regimes.npz"):
            path = learning.OUT/f"final/n1000_r{replicate}"/name
            consumed[str(path.relative_to(OUT.parent.parent))] = prior.sha(path.read_bytes())
            assert consumed[str(path.relative_to(OUT.parent.parent))] == protocol["files"][f"final/n1000_r{replicate}/{name}"]
    if state.startswith("fit"):
        path = learning.OUT/f"r{replicate}_{regime}_Poly0_32_91.json"
        saved = json.loads(path.read_text())
        assert saved["status"] == "complete" and saved["steps"] == 100 and saved["seed"] == 91 and not saved["numerical_warning"]
        assert saved["fingerprint"] == fingerprint and path.stem in protocol["schedule"]
        consumed[str(path.relative_to(OUT.parent.parent))] = prior.sha(path.read_bytes())
        consumed[str(path.with_suffix(".pt").relative_to(OUT.parent.parent))] = prior.sha(path.with_suffix(".pt").read_bytes())
        assert consumed[str(path.with_suffix(".pt").relative_to(OUT.parent.parent))] == saved["checkpoint_sha"]
        model.load_state_dict(torch.load(path.with_suffix(".pt"),weights_only=True)["model"])
        assert learning.raw(model) == saved["final"]
        torch.testing.assert_close(model.covar_module.base_kernel.laplacian,graph,atol=0,rtol=0)
    elif state == "stress":
        prior.second_state(model,"Poly")
    identity = dict(fingerprint=fingerprint,regime=regime,raw=learning.raw(model),consumed=consumed,
        inputs={p.name:prior.sha(p.read_bytes()) for p in directory.iterdir() if p.is_file()})
    return graph,train,test,targets[regime],float(scales[regime]),model,identity


def root(model):
    kernel = model.covar_module.base_kernel
    return model.covar_module.outputscale.sqrt()*torch.stack((1+kernel.a1-kernel.a2,kernel.a1,kernel.a2))


def reference(graph, train, test, y, model):
    L, kernel, noise = graph.to_dense(), model.covar_module.base_kernel, model.likelihood.noise
    B = model.covar_module.outputscale.sqrt()*(torch.eye(len(graph))+kernel.a1*L+kernel.a2*(2*L@L-4*L))
    ids = torch.cat([train,test])
    covariance = B[ids]@B[ids].T
    _,mean,variance = posterior(covariance,y[train],noise)
    factor = torch.linalg.cholesky(covariance[:len(train),:len(train)]+noise*torch.eye(len(train)))
    cross = covariance[len(train):,:len(train)]
    checks = {}
    if len(graph) == 36:
        model_covariance = model.covar_module(torch.arange(36)).to_dense()
        torch.testing.assert_close(B@B.T,model_covariance,atol=1e-10,rtol=1e-9)
        gain = torch.cholesky_solve(cross.T,factor).T
        mapping = B[test]-gain@B[train]
        mapped = mapping@mapping.T+noise*gain@gain.T
        torch.testing.assert_close(mapped,covariance[len(train):,len(train):]-gain@cross.T,atol=1e-10,rtol=1e-9)
        z = torch.randn(36,32,generator=torch.Generator().manual_seed(0))
        with sparse(36,len(train),len(test)):
            action = apply_chebyshev(graph,root(model),z)
        torch.testing.assert_close(action,B@z,atol=1e-10,rtol=1e-9)
        checks = dict(root_max=float((action-B@z).abs().max()),covariance_max=float((B@B.T-model_covariance).abs().max()),map_max=float((mapped-covariance[len(train):,len(train):]+gain@cross.T).abs().max()))
    return B,factor,cross,mean,variance,checks


def errors(estimate, exact):
    difference = (estimate-exact).abs()
    relative = difference/exact  # All analytic marginal variances are strictly positive; no floor.
    return dict(l2=float(difference.norm()/exact.norm()),max_abs=float(difference.max()),max_rel=float(relative.max()),p95_rel=float(torch.quantile(relative,.95)))


def scores(mean, variance, y, noise, scale):
    rmse,nlpd = score(mean,variance,y,noise,scale)
    coverage = float(((mean-y).abs() <= 1.959963984540054*(variance+noise).sqrt()).double().mean())
    return dict(rmse_raw=rmse,rmse_normalized=rmse/scale,nlpd_normalized=nlpd,nlpd_raw=nlpd+np.log(scale),coverage95=coverage)


def run(n, replicate, state, frozen):
    tag = f"n{n}_r{replicate}_{state}"
    path = OUT/f"{tag}.json"
    assert not path.exists(), "Preserve prior attempts; never silently retry"
    result = dict(status="failed",rows=[],case=[n,replicate,state])
    with warnings.catch_warnings(record=True) as caught, torch.no_grad(), learning.inference():
        warnings.simplefilter("always",NumericalWarning)
        try:
            graph,train,test,y,scale,model,identity = load(n,replicate,state)
            assert identity == frozen["cases"][tag]
            B,factor,cross,exact_mean,variance,checks = reference(graph,train,test,y,model)
            noise, times, original_residual = model.likelihood.noise, [], prior.residual
            def residual_audit(*args):
                times.append(perf_counter())
                return original_residual(*args)
            with sparse(n,len(train),len(test)), patch.object(prior,"residual",side_effect=residual_audit):
                start = perf_counter()
                mean,mean_residual = learning.prediction(model,train,test,y)
            mean_s, mean_error = times[0]-start, float((mean-exact_mean).abs().max())
            np.savez(OUT/f"{tag}_reference.npz",mean=exact_mean.numpy(),latent=variance.numpy(),observation=(variance+noise).numpy(),raw_latent=(variance*scale**2).numpy(),raw_observation=((variance+noise)*scale**2).numpy(),scale=scale)
            result.update(identity=identity,deterministic=checks,mean_s=mean_s,mean_error=mean_error,mean_residual=mean_residual,reference_scores=scores(exact_mean,variance,y[test],noise,scale))
            for seed in ((421,) if n == 36 else (421,422,423)):
                generator = torch.Generator().manual_seed(seed)
                paths, exact_paths = torch.empty(len(test),1024), torch.empty(len(test),1024)
                elapsed, maximum, zeros, zero_absolute = 0., 0., 0, 0.
                for offset in range(0,32 if n == 36 else 1024,32):
                    with sparse(n,len(train),len(test)):
                        start = perf_counter()
                        z,e = torch.randn(n,32,generator=generator),torch.randn(len(train),32,generator=generator)
                        f0 = apply_chebyshev(graph,root(model),z)
                        rhs = f0[train]+noise.sqrt()*e
                        A = model.covar_module(train).add_diagonal(noise.expand(len(train)))
                        alpha = linear_cg(A.matmul,rhs,n_tridiag=0,tolerance=1e-4,max_iter=1000,preconditioner=None)
                        paths[:,offset:offset+32] = f0[test]-model.covar_module(test,train)@alpha
                        elapsed += perf_counter()-start
                        residual = (A@alpha-rhs).norm(dim=0)
                    norms = rhs.norm(dim=0)
                    maximum = max(maximum,float((residual[norms>0]/norms[norms>0]).max()) if (norms>0).any() else 0.)
                    zeros += int((norms == 0).sum())
                    zero_absolute = max(zero_absolute,float(residual[norms == 0].max()) if (norms == 0).any() else 0.)
                    exact_prior = B@z
                    exact_paths[:,offset:offset+32] = exact_prior[test]-cross@torch.cholesky_solve(exact_prior[train]+noise.sqrt()*e,factor)
                    np.savez(OUT/f"{tag}_s{seed}_b{offset}.npz",z=z.numpy(),e=e.numpy(),residual=residual.numpy(),rhs_norm=norms.numpy())
                    count = offset+32
                    if count not in ((32,) if n == 36 else (128,512,1024)):
                        continue
                    start = perf_counter()
                    estimate = paths[:,:count].var(dim=1,unbiased=True)
                    observation = estimate+noise
                    moment_s = perf_counter()-start
                    exact_draw_variance = exact_paths[:,:count].var(dim=1,unbiased=True)
                    same_draw = float((estimate-exact_draw_variance).norm()/exact_draw_variance.norm())
                    row = dict(case=tag,seed=seed,P=count,latent_error=errors(estimate,variance),observation_error=errors(observation,variance+noise),same_draw_l2=same_draw,
                        mean_error=mean_error,centered_sample_mean_max=float(paths[:,:count].mean(1).abs().max()),max_residual=maximum,zero_rhs=zeros,zero_rhs_max_absolute=zero_absolute,
                        sampling_s=elapsed+moment_s,incremental_sampling_s=elapsed+moment_s-(result["rows"][-1]["sampling_s"] if result["rows"] and result["rows"][-1]["seed"] == seed else 0),mean_variance_s=mean_s+elapsed+moment_s,
                        scores=scores(mean,estimate,y[test],noise,scale),latent=estimate.tolist(),observation=observation.tolist(),raw_latent=(estimate*scale**2).tolist(),raw_observation=(observation*scale**2).tolist(),warnings=[f"{w.category.__name__}: {w.message}" for w in caught])
                    row["numerical_pass"] = bool(torch.isfinite(paths[:,:count]).all() and torch.isfinite(exact_paths[:,:count]).all() and torch.isfinite(estimate).all() and (estimate >= 0).all() and (observation > 0).all()) and maximum <= 2e-4 and zero_absolute == 0 and mean_error <= .002 and mean_residual <= 2e-4 and same_draw <= .002 and not any(issubclass(w.category,NumericalWarning) for w in caught)
                    row["mc_pass"] = row["latent_error"]["l2"] <= .1 and row["observation_error"]["l2"] <= .05
                    result["rows"].append(row)
                    (OUT/f"{tag}_s{seed}_p{count}.json").write_text(json.dumps(row,indent=2))
                    np.savez(OUT/f"{tag}_s{seed}_p{count}_paths.npz",iterative=paths[:,:count].numpy(),exact=exact_paths[:,:count].numpy())
                    assert row["numerical_pass"], "Frozen numerical gate failed; stop"
            result["status"] = "complete"
        except Exception as error:
            result["error"] = repr(error)
        result["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
        path.write_text(json.dumps(result,indent=2))
        print(tag,result["status"],result.get("error",""),flush=True)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    for path,revision in ((prior.CORE,prior.CORE_COMMIT),(prior.UPSTREAM,prior.EXPERIMENT_COMMIT)):
        assert subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip() == revision
        subprocess.run(["git","-C",str(path),"diff","HEAD","--exit-code"],check=True)
    provenance = dict(base="3989d58b8fe081446d4fe89b39bced47261d0dfc",order=prior.sha((OUT/"work_order.md").read_bytes()),
        sources={p.name:prior.sha(p.read_bytes()) for p in Path(__file__).resolve().parent.glob("*.py")},
        packages=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True).splitlines(),pins=[prior.CORE_COMMIT,prior.EXPERIMENT_COMMIT],
        seeds=[421,422,423],prefixes=[128,512,1024],batch=32,cg=1e-4,maxCG=1000,threads=4)
    if sys.argv[1] == "freeze":
        assert not (OUT/"manifest.json").exists()
        cases = {f"n{n}_r{r}_{s}":load(n,r,s)[-1] for n,r,s in SCHEDULE}
        (OUT/"manifest.json").write_text(json.dumps(dict(provenance=provenance,schedule=SCHEDULE,cases=cases),indent=2))
    else:
        frozen = json.loads((OUT/"manifest.json").read_text())
        assert frozen["provenance"] == provenance
        n,r,state = sys.argv[1:]
        assert [int(n),int(r),state] in frozen["schedule"]
        run(int(n),int(r),state,frozen)
