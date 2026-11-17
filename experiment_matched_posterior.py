"""Matched fixed-checkpoint posterior paths and residual-bracket variance audits."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch
import warnings

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils._python_dispatch import _disable_current_modes
from linear_operator.operators import LinearOperator
from linear_operator.utils.linear_cg import linear_cg
from linear_operator.utils.warnings import NumericalWarning
from linear_operator import settings
import experiment_poly_posterior as poly
import experiment_learned_kernel_scaling as prior
import experiment_learned_kernel_scaling_final as fitted
from grf_gp.inference import pathwise_conditioning

OUT = Path(__file__).resolve().parent/"results/matched_posterior_scaling"
RUN = OUT/"continuation"
SEEDS = (521,522,523)


def load(n, r, regime, arm, state):
    source = poly.learning.OUT if n == 1000 else fitted.OUT
    directory = source/f"final/n{n}_r{r}"
    consumed, record, saved_mean = {}, {}, None
    if n == 36:
        with patch.object(prior,"OUT",OUT):
            graph,targets,train,test,scales,directory,fingerprint,_ = prior.inputs(n,r)
    else:
        L = sp.load_npz(directory/f"laplacian_{n}.npz")
        coo = L.tocoo()
        graph = torch.sparse_coo_tensor(np.stack((coo.row,coo.col)),coo.data,L.shape).coalesce()
        with np.load(directory/f"data_{n}.npz") as data, np.load(directory/"regimes.npz") as signals:
            train,test = torch.from_numpy(data["train"]),torch.from_numpy(data["test"])
            targets,scales = torch.from_numpy(signals["normalized"]),signals["scale"].copy()
        fingerprint = prior.sha(L.data.tobytes()+L.indices.tobytes()+L.indptr.tobytes()+targets.numpy().tobytes()+train.numpy().tobytes()+test.numpy().tobytes())
    for name in (f"laplacian_{n}.npz",f"data_{n}.npz","regimes.npz"):
        consumed[str((directory/name).relative_to(OUT.parent.parent))] = prior.sha((directory/name).read_bytes())
    assert len(train) == len(test) == n//4 and not np.intersect1d(train.numpy(),test.numpy()).size
    np.testing.assert_allclose(targets[:,train].mean(1),0,atol=1e-13)
    np.testing.assert_allclose(targets[:,train].std(1,unbiased=False),1,atol=1e-13)
    if n == 1000:
        protocol = json.loads((source/"protocol.json").read_text())
        assert protocol["data"][str(r)]["fingerprint"] == fingerprint
        assert all(prior.sha((directory/name).read_bytes()) == protocol["files"][f"final/n{n}_r{r}/{name}"] for name in (f"laplacian_{n}.npz",f"data_{n}.npz","regimes.npz"))
    if state == "fit":
        tag = f"r{r}_{regime}_{arm}{1024 if arm == 'General' else 0}_32_91" if n == 1000 else f"n{n}_r{r}_{regime}_{arm}_W{1024 if arm == 'General' else 0}_100"
        path = source/f"{tag}.json"
        record = json.loads(path.read_text())
        assert record["status"] == "complete" and record["steps"] == 100 and record["fingerprint"] == fingerprint
        assert n != 1000 or record["seed"] == 91 and not record["numerical_warning"]
        assert all(prior.sha((directory/name).read_bytes()) == digest for name,digest in record.get("input_hashes",{}).items())
        assert prior.sha(path.with_suffix(".pt").read_bytes()) == record["checkpoint_sha"]
        point = torch.load(path.with_suffix(".pt"),weights_only=True)
        saved_mean = point.get("mean")
        consumed.update({str(p.relative_to(OUT.parent.parent)):prior.sha(p.read_bytes()) for p in (path,path.with_suffix(".pt"))})
    if arm == "General":
        cache = directory/"walks_1024.pt"
        assert n == 36 or cache.exists(), "Missing original walks: never resample"
        if n == 1000:
            assert prior.sha(cache.read_bytes()) == protocol["files"][f"final/n{n}_r{r}/walks_1024.pt"]
        elif n > 1000:
            assert prior.sha(cache.read_bytes()) == record["input_hashes"][cache.name]
    model,_ = prior.construct(graph,train,targets[regime],arm,1024 if arm == "General" else 0,directory,r)
    if arm == "General":
        consumed[str(cache.relative_to(OUT.parent.parent))] = prior.sha(cache.read_bytes())
    if state == "fit":
        model.load_state_dict(point["model"])
        assert poly.learning.raw(model) == record["final"]
    elif state == "stress":
        prior.second_state(model,arm)
    if arm == "Poly":
        torch.testing.assert_close(model.covar_module.base_kernel.laplacian,graph,atol=0,rtol=0)
    assert sum(p.numel() for p in model.parameters()) == 4 and all(torch.isfinite(p).all() for p in model.parameters())
    positions = np.random.default_rng(891000+n+100000*r).choice(len(test),min(256,len(test)),replace=False)
    if n == 36:
        positions = positions[:3]  # Local 3-column bases cannot collide with full 9x9 forbidden blocks.
    identity = dict(fingerprint=fingerprint,consumed=consumed,raw=poly.learning.raw(model),positions=positions.tolist(),vertices=test[positions].tolist())
    return graph,train,test,targets[regime],float(scales[regime]),model,saved_mean,record,identity


def prepare(graph, train, test, model, arm):
    start = perf_counter()
    nu = model.likelihood.noise
    if arm == "Poly":
        B = poly.root(model)
        A,C = model.covar_module(train).add_diagonal(nu.expand(len(train))),model.covar_module(test,train)
    else:
        B = model.covar_module._get_feature_matrix()
        assert isinstance(B,LinearOperator) and B.shape == graph.shape
        observed,heldout = B[train],B[test]
        A,C = (observed@observed.T).add_diagonal(nu.expand(len(train))),heldout@observed.T
    assert isinstance(A,LinearOperator) and isinstance(C,LinearOperator)
    return B,A,C,nu,perf_counter()-start


def solve(A, rhs):
    return linear_cg(A.matmul,rhs,n_tridiag=0,tolerance=1e-4,max_iter=1000,preconditioner=None)


def residual(A, x, rhs, matmul=None):
    r = rhs-(A@x if matmul is None else matmul(x))
    norms,error = rhs.norm(dim=0),r.norm(dim=0)
    positive = norms > 0
    values = error[positive]/norms[positive]
    stats = dict(max=float(values.max()) if values.numel() else 0.,mean=float(values.mean()) if values.numel() else 0.,
        p95=float(torch.quantile(values,.95)) if values.numel() else 0.,zeros=int((~positive).sum()),zero_absolute=float(error[~positive].max()) if (~positive).any() else 0.)
    return r,stats,values


def selected_audit(graph, train, test, model, A, nu, positions):
    start = perf_counter()
    K = model.covar_module(torch.arange(len(graph)))
    lower,upper,diagnostics = [],[],[]
    for offset in range(0,len(positions),3 if len(graph) == 36 else 32):
        ids = test[positions[offset:offset+(3 if len(graph) == 36 else 32)]]
        basis = torch.zeros(len(graph),len(ids))
        basis[ids,torch.arange(len(ids))] = 1
        columns = K@basis
        c,d = columns[train],columns[ids,torch.arange(len(ids))]
        x = solve(A,c)
        r,stats,_ = residual(A,x,c)
        hi = d-(c*x).sum(0)-(x*r).sum(0)
        lo = hi-r.square().sum(0)/nu
        assert torch.isfinite(lo).all() and torch.isfinite(hi).all() and (lo > 0).all() and ((hi-lo)/lo <= .002).all() and stats["zero_absolute"] == 0, f"Reference-unqualified: lower={lo.tolist()}, upper={hi.tolist()}, residual={stats}"
        lower.append(lo)
        upper.append(hi)
        diagnostics.append(stats)
    return torch.cat(lower),torch.cat(upper),dict(seconds=perf_counter()-start,residuals=diagnostics)


def dense_reference(graph, train, test, y, model, arm):
    if arm == "Poly":
        return poly.reference(graph,train,test,y,model)[:5]
    B = sum(f*m.sparse_csr_tensor.to_dense() for f,m in zip(model.covar_module.modulation_function,model.covar_module.rw_mats))
    ids,nu = torch.cat((train,test)),model.likelihood.noise
    covariance = B[ids]@B[ids].T
    _,mean,var = poly.posterior(covariance,y[train],nu)
    factor = torch.linalg.cholesky(covariance[:len(train),:len(train)]+nu*torch.eye(len(train)))
    return B,factor,covariance[len(train):,:len(train)],mean,var


def tiny_checks(graph, train, test, model, arm, reference):
    dense,factor,cross,_,_ = reference
    ids,nu = torch.cat((train,test)),model.likelihood.noise
    K = dense[ids]@dense[ids].T
    gain = torch.cholesky_solve(cross.T,factor).T
    mapping = dense[test]-gain@dense[train]
    mapped = mapping@mapping.T+nu*gain@gain.T
    torch.testing.assert_close(mapped,K[len(train):,len(train):]-gain@cross.T,atol=1e-10,rtol=1e-9)
    actual = model.covar_module(torch.arange(len(graph))).to_dense()
    torch.testing.assert_close(dense@dense.T,actual,atol=1e-10,rtol=1e-9)
    return dict(covariance_max=float((dense@dense.T-actual).abs().max()),map_max=float((mapped-K[len(train):,len(train):]+gain@cross.T).abs().max()))


def native_audit(B, A, C, nu, train, test, z, e, f0, rhs, x, u, reference, label):
    """Tiny-only native capture; deterministic algebra stays strict, paths use residual bounds."""
    captured = []
    def capture(matmul, b, **kwargs):
        trace = []
        def product(v):
            result = matmul(v)
            trace.append(dict(denominator=(v*result).sum(0).tolist(),input=v.tolist()))
            return result
        solution = linear_cg(product,b,**kwargs)
        captured.append(dict(operator=matmul.__self__,rhs=b.clone(),x=solution,trace=trace))
        return solution
    with patch("grf_gp.inference.linear_cg",side_effect=capture), patch("torch.randn",side_effect=[z.T,e.T]):
        native = pathwise_conditioning(train,test,B,torch.zeros(len(train)),nu.sqrt(),z.shape[1],torch.device("cpu")).T
    repeated = capture(A.matmul,rhs,n_tridiag=0,tolerance=1e-4,max_iter=1000,preconditioner=None)
    original,adapted = captured
    with patch("grf_gp.inference.linear_cg",side_effect=capture), patch("torch.randn",side_effect=[z.T.contiguous(),e.T.contiguous()]):
        contiguous = pathwise_conditioning(train,test,B,torch.zeros(len(train)),nu.sqrt(),z.shape[1],torch.device("cpu")).T
    traces = dict(native=original["trace"],adapter=adapted["trace"])
    N,xn,bn = original["operator"],-original["x"],-original["rhs"]
    CN = B[test]@B[train].T
    dense,factor,cross,_,_ = reference
    m,n = len(train),len(dense)
    eye = torch.eye(m)
    A0 = dense[train]@dense[train].T+nu*eye
    b0 = (dense@z)[train]+nu.sqrt()*e
    fp = (dense@z)[test]
    exact_x = torch.cholesky_solve(b0,factor)
    exact = fp-cross@exact_x
    native_prior = (z.T@B[test].T).T
    selected = (z.T@B[train].T).T
    matrices = dict(adapter_A=A@eye,native_A=N._matmul(eye),adapter_C=C@eye,native_C=CN@eye)
    deterministic = [(matrices["adapter_A"],A0),(matrices["native_A"],A0),
        (matrices["adapter_C"],cross),(matrices["native_C"],cross),(rhs,b0),(bn,b0),
        (f0,dense@z),(native_prior,fp),(selected,(dense@z)[train]),(B[train]@z,f0[train]),
        (A.matmul(eye),A._matmul(eye)),(N.matmul(eye),N._matmul(eye)),
        (repeated,x),(native,native_prior-CN@xn)]
    # Gram assembly error bounds lambda_min(A0); gamma terms also cover norm/subtraction rounding.
    eps = torch.finfo(torch.float64).eps/2
    gram_gamma = n*eps/(1-n*eps)
    gamma = (8*m+4)*eps/(1-(8*m+4)*eps)
    lower = nu-gram_gamma*(dense[train].abs()@dense[train].abs().T).norm()-eps*(A0.norm()+nu)
    assert lower > 0
    chol_residual = (b0-A0@exact_x).norm(dim=0)+gamma*(b0.norm(dim=0)+(A0.abs()@exact_x.abs()).norm(dim=0))
    metrics,arrays,bounds = {},dict(z=z,e=e,A0=A0,C0=cross,b0=b0,exact_x=exact_x,exact_paths=exact,
        native_cg_rhs=original["rhs"],native_cg_solution=original["x"],contiguous_paths=contiguous),[]
    for name,operator,correction,bi,xi,prior_i,paths_i in (
            ("adapter",A,C,rhs,x,f0[test],u),("native",N,CN,bn,xn,native_prior,native)):
        action = operator._matmul if name == "native" else operator.matmul
        ax,ci = action(xi),correction@xi
        ri,stats,values = residual(operator,xi,bi,action)
        operator_difference = (ax-A0@xi).norm(dim=0)
        rhs_difference = (bi-b0).norm(dim=0)
        solution_budget = (ri.norm(dim=0)+operator_difference+rhs_difference+
            gamma*(bi.norm(dim=0)+ax.norm(dim=0)+(A0.abs()@xi.abs()).norm(dim=0)))/lower
        budget = (prior_i-fp).norm(dim=0)+cross.norm()*solution_budget+(ci-cross@xi).norm(dim=0)
        budget += cross.norm()*chol_residual/lower+gamma*(prior_i.norm(dim=0)+fp.norm(dim=0)+ci.norm(dim=0)+(cross.abs()@xi.abs()).norm(dim=0))
        centered,exact_centered = paths_i-paths_i.mean(1,keepdim=True),exact-exact.mean(1,keepdim=True)
        covariance,exact_covariance = centered@centered.T/(z.shape[1]-1),exact_centered@exact_centered.T/(z.shape[1]-1)
        same_draw = float((paths_i.var(1,unbiased=True)-exact.var(1,unbiased=True)).norm()/exact.var(1,unbiased=True).norm())
        Ai,Ci = matrices[f"{name}_A"],matrices[f"{name}_C"]
        gain = torch.cholesky_solve(Ci.T,torch.linalg.cholesky(Ai)).T
        mapping = dense[test]-gain@dense[train]
        map_covariance = mapping@mapping.T+nu.sqrt().square()*gain@gain.T
        expected_covariance = dense[test]@dense[test].T-cross@torch.cholesky_solve(cross.T,factor)
        deterministic.append((map_covariance,expected_covariance))
        bounds.append(budget)
        metrics[name] = dict(residual=stats,per_rhs=values.tolist(),same_draw_l2=same_draw,
            covariance_relative=float((covariance-exact_covariance).norm()/exact_covariance.norm()),
            path_error_max=float((paths_i-exact).abs().max()),path_norm_errors=(paths_i-exact).norm(dim=0).tolist(),
            path_bounds=budget.tolist(),solution_error_max=float((xi-exact_x).abs().max()),
            operator_action_difference_max=float(operator_difference.max()),rhs_difference_max=float(rhs_difference.max()),
            iterations=len(traces[name])-1)
        arrays.update({f"{name}_{key}":value for key,value in dict(A=Ai,C=Ci,rhs=bi,x=xi,paths=paths_i,residual=ri,map_covariance=map_covariance).items()})
    pair_error = (u-native).norm(dim=0)
    pair_bound = bounds[0]+bounds[1]
    report = dict(metrics=metrics,arrays={k:v.tolist() for k,v in arrays.items()},
        deterministic_errors=[float((a-b).abs().max()) for a,b in deterministic],
        native_adapter_max=float((u-native).abs().max()),pair_errors=pair_error.tolist(),pair_bounds=pair_bound.tolist(),
        noise=float(nu),squared_std_difference=float(nu.sqrt().square()-nu),rhs_strides=[list(rhs.stride()),list(bn.stride())],
        contiguous_native_difference=float((contiguous-native).abs().max()),contiguous_adapter_difference=float((contiguous-u).abs().max()),
        operators=[type(A).__name__,type(N).__name__],trace=[c["trace"] for c in captured],
        cg=dict(tolerance=settings.cg_tolerance.value(),max_iter=settings.max_cg_iterations.value(),n_tridiag=0,
            eps=1e-10,stop_updating_after=1e-10,terminate_by_size=settings.terminate_cg_by_size.on(),minimum_iterations=11),
        lambda_lower=float(lower),gamma=gamma,gram_gamma=gram_gamma)
    path = RUN/f"native_{label}.json"
    assert not path.exists()
    path.write_text(json.dumps(report,indent=2))
    for a,b in deterministic:
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)
    assert (pair_error <= pair_bound).all()
    for name,paths_i,budget in zip(("adapter","native"),(u,native),bounds):
        assert torch.isfinite(paths_i).all() and ((paths_i-exact).norm(dim=0) <= budget).all()
        assert metrics[name]["residual"]["max"] <= 2e-4 and metrics[name]["residual"]["zero_absolute"] == 0
        assert metrics[name]["same_draw_l2"] <= .002
    return {k:v for k,v in report.items() if k not in ("arrays","trace")}


def explain_native():
    """Analyze captured arrays only; do not rerun the failed diagnostic or any sampling."""
    source = RUN/"native_diagnostic_n36_r9_0_General_initial_s521.json"
    saved = json.loads(source.read_text())
    arrays = {k:torch.tensor(v,dtype=torch.float64) for k,v in saved["arrays"].items()}
    nu = saved["noise"]
    delta = float(torch.tensor(nu,dtype=torch.float32).double()-nu)
    difference = arrays["native_A"]-arrays["adapter_A"]
    torch.testing.assert_close(difference,delta*torch.eye(len(difference)),atol=1e-10,rtol=1e-9)
    own_paths,own_errors = {},{}
    for name in ("adapter","native"):
        Ai,Ci,bi,xi = (arrays[f"{name}_{key}"] for key in ("A","C","rhs","x"))
        exact_x = torch.cholesky_solve(bi,torch.linalg.cholesky(Ai))
        prior_paths = arrays[f"{name}_paths"]+Ci@xi
        own_paths[name] = prior_paths-Ci@exact_x
        own_errors[name] = float((arrays[f"{name}_paths"]-own_paths[name]).abs().max())
    observed = arrays["adapter_paths"]-arrays["native_paths"]
    exact_difference = own_paths["adapter"]-own_paths["native"]
    noise_only = arrays["C0"]@(torch.cholesky_solve(arrays["b0"],torch.linalg.cholesky(arrays["A0"]+delta*torch.eye(len(difference))))-arrays["exact_x"])
    numerical_pass = all(row["residual"]["max"] <= 2e-4 and row["residual"]["zero_absolute"] == 0 and row["same_draw_l2"] <= .002 and
        all(e <= b for e,b in zip(row["path_norm_errors"],row["path_bounds"])) for row in saved["metrics"].values())
    numerical_pass = numerical_pass and all(e <= b for e,b in zip(saved["pair_errors"],saved["pair_bounds"]))
    result = dict(source_sha=prior.sha(source.read_bytes()),code_sha=prior.sha(Path(__file__).read_bytes()),
        noise=nu,noise_float32=float(np.float32(nu)),diagonal_delta=delta,
        unexplained_operator_max=float((difference-delta*torch.eye(len(difference))).abs().max()),
        own_exact_path_difference_max=float(exact_difference.abs().max()),own_iterative_errors=own_errors,
        differential_iterative_error_max=float((observed-exact_difference).abs().max()),
        noise_only_prediction_max=float(noise_only.abs().max()),noise_only_unexplained_path_max=float((observed-noise_only).abs().max()),
        numerical_pass=numerical_pass,operator_equivalence_pass=False,decision="STOP: native float32 noise diagonal fails strict operator equivalence")
    path = RUN/"native_explanation.json"
    assert not path.exists()
    path.write_text(json.dumps(result,indent=2))
    assert numerical_pass
    return result


def stream(graph, train, test, B, A, C, nu, arm, prefixes, seed, reference, label):
    generator = torch.Generator().manual_seed(seed)
    count,average,M2 = 0,torch.zeros(len(test)),torch.zeros(len(test))
    paths_s,moments_s,audits_s = 0.,0.,0.
    actual,exact,relative,zero_count,zero_abs = [],[],[],0,0.
    for offset in range(0,max(prefixes),32):
        start = perf_counter()
        width = min(32,max(prefixes)-offset)
        z,e = torch.randn(len(graph),width,generator=generator),torch.randn(len(train),width,generator=generator)
        f0 = poly.apply_chebyshev(graph,B,z) if arm == "Poly" else B@z
        rhs = f0[train]+nu.sqrt()*e
        x = solve(A,rhs)
        u = f0[test]-C@x
        paths_s += perf_counter()-start
        start = perf_counter()
        batch_mean = u.mean(1)
        delta = batch_mean-average
        M2 = M2+(u-batch_mean[:,None]).square().sum(1)+delta.square()*count*width/(count+width)
        average = average+delta*width/(count+width)
        count += width
        variance = M2/(count-1)
        moments_s += perf_counter()-start
        start = perf_counter()
        _,stats,values = residual(A,x,rhs)
        relative.append(values)
        zero_count += stats["zeros"]
        zero_abs = max(zero_abs,stats["zero_absolute"])
        assert torch.isfinite(u).all() and stats["max"] <= 2e-4 and zero_abs == 0, f"Unsafe posterior solve: {stats}; per-RHS {values.tolist()}"
        if reference is not None:
            with _disable_current_modes():  # Dense validation only; candidate operations remain guarded/timed separately.
                dense,factor,cross,_,_ = reference
                f = dense@z
                exact.append(f[test]-cross@torch.cholesky_solve(f[train]+nu.sqrt()*e,factor))
            actual.append(u)
            if len(graph) == 36 and arm == "General":
                torch.testing.assert_close(f0,f,atol=1e-10,rtol=1e-9)
                with _disable_current_modes():
                    native = native_audit(B,A,C,nu,train,test,z,e,f0,rhs,x,u,reference,label)
        audits_s += perf_counter()-start
        if count in prefixes:
            assert torch.isfinite(variance).all() and (variance >= 0).all() and (variance+nu > 0).all()
            rr = torch.cat(relative)
            details = dict(P=count,paths_s=paths_s,moments_s=moments_s,residual_audit_s=audits_s,max_residual=float(rr.max()) if rr.numel() else 0.,
                mean_residual=float(rr.mean()) if rr.numel() else 0.,p95_residual=float(torch.quantile(rr,.95)) if rr.numel() else 0.,zero_rhs=zero_count,zero_rhs_absolute=zero_abs,per_rhs=rr.tolist())
            if reference is not None:
                torch.testing.assert_close(variance,torch.cat(actual,1).var(1,unbiased=True),atol=1e-10,rtol=1e-9)
                same = torch.cat(exact,1).var(1,unbiased=True)
                details.update(same_draw_l2=float((variance-same).norm()/same.norm()),latent_error=poly.errors(variance,reference[4]),observation_error=poly.errors(variance+nu,reference[4]+nu),same_draw_reference=same.tolist(),streaming_error_max=float((variance-torch.cat(actual,1).var(1,unbiased=True)).abs().max()))
                if len(graph) == 36 and arm == "General":
                    details.update(native=native,root_max_error=float((f0-f).abs().max()))
                assert details["same_draw_l2"] <= .002
            yield variance.clone(),details


def worker(job, manifest):
    mode,n,r,g,arm,state,seed,P = job
    tag = f"{mode}_n{n}_r{r}_{g}_{arm}_{state}_s{seed}"
    output = RUN/f"{tag}.json"
    assert not output.exists(), "Preserve prior attempts"
    result = dict(job=job,status="failed",rows=[])
    with torch.no_grad(), poly.learning.inference(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",NumericalWarning)
        try:
            graph,train,test,y,scale,model,saved,old,identity = load(n,r,g,arm,state)
            assert identity == manifest["identities"][f"{n}_{r}_{g}_{arm}_{state}"]
            result.update(identity=identity)
            guard = poly.sparse(n,len(train),len(test)) if mode in ("qual","diagnostic","smoke") else nullcontext()
            reference = dense_reference(graph,train,test,y,model,arm) if mode in ("qual","diagnostic") else None
            if reference is not None:
                result["reference_scores"] = poly.scores(reference[3],reference[4],y[test],model.likelihood.noise,scale)
                np.savez(RUN/f"{tag}_reference.npz",mean=reference[3].numpy(),variance=reference[4].numpy())
                if n == 36:
                    result["deterministic"] = tiny_checks(graph,train,test,model,arm,reference)
            with guard:
                audit_path = RUN/f"audit_n{n}_r{r}_{g}_{arm}_{state}.npz"
                if mode in ("qual","diagnostic","smoke","audit") and not audit_path.exists():
                    _,A,_,nu,_ = prepare(graph,train,test,model,arm)
                    lo,hi,audit = selected_audit(graph,train,test,model,A,nu,identity["positions"])
                    if reference is not None:
                        v = reference[4][identity["positions"]]
                        tolerance = 1e-10+1e-9*v.abs()
                        assert (lo <= v+tolerance).all() and (v <= hi+tolerance).all(), "Dense variance outside bracket"
                    np.savez(audit_path,lower=lo.numpy(),upper=hi.numpy())
                    audit_path.with_suffix(".json").write_text(json.dumps(dict(identity=identity,sha=prior.sha(audit_path.read_bytes()),audit=audit),indent=2))
                else:
                    audit_info = json.loads(audit_path.with_suffix(".json").read_text())
                    assert audit_info["identity"] == identity and audit_info["sha"] == prior.sha(audit_path.read_bytes())
                    audit = audit_info["audit"]
                    with np.load(audit_path) as data:
                        lo,hi = torch.from_numpy(data["lower"]),torch.from_numpy(data["upper"])
                result.update(audit=audit,audit_sha=prior.sha(audit_path.read_bytes()),width_max=float(((hi-lo)/lo).max()))
                if mode == "audit":
                    result["status"] = "complete"
                else:
                    seeds = SEEDS if mode == "qual" and n == 1000 and arm == "General" else (seed,)
                    prefixes = (512,1024) if mode == "qual" and n == 1000 else (P,)
                    for stream_seed in seeds:
                        B,A,C,nu,prep_s = prepare(graph,train,test,model,arm)
                        start = perf_counter()
                        alpha = solve(A,y[train,None])
                        mean = (C@alpha)[:,0]
                        mean_s = perf_counter()-start
                        _,mean_stats,_ = residual(A,alpha,y[train,None])
                        target = reference[3] if reference is not None else saved
                        mean_error = float((mean-target).abs().max())
                        result.update(mean_error=mean_error,mean_relative=float((mean-target).norm()/target.norm()),mean_residual=mean_stats,prep_s=prep_s,mean_s=mean_s)
                        assert mean_error <= .002 and mean_stats["max"] <= 2e-4 and torch.isfinite(mean).all(), f"Unsafe mean: error={mean_error}, residual={mean_stats}"
                        for variance,row in stream(graph,train,test,B,A,C,nu,arm,prefixes,stream_seed,reference,tag):
                            h = variance[identity["positions"]]
                            numerator = (h-hi).norm()+(hi-lo).norm()
                            row.update(seed=stream_seed,prep_s=prep_s,mean_s=mean_s,mean_error=mean_error,mean_residual=mean_stats,prediction_s=prep_s+mean_s+row["paths_s"]+row["moments_s"],
                                latent_bound=float(numerator/lo.norm()),observation_bound=float(numerator/(lo+nu).norm()),audit_error=poly.errors(h,hi),observation_audit_error=poly.errors(h+nu,hi+nu),
                                scores=poly.scores(mean,variance,y[test],nu,scale),mean_observation_sd_raw=float((variance+nu).sqrt().mean()*scale))
                            row["mc_pass"] = (row["latent_error"]["l2"] <= .10 and row["observation_error"]["l2"] <= .05) if reference is not None else row["latent_bound"] <= .10 and row["observation_bound"] <= .05
                            row["mean_interval95_width_raw"] = 3.919927969080108*row["mean_observation_sd_raw"]
                            row.update(reconstructed_standalone_s=old["setup_s"]+old["walk_s"]+old["training_s"]+row["prediction_s"] if n>1000 else None,amortized_s=old["setup_s"]+old["training_s"]+row["prediction_s"] if n>1000 else None)
                            assert not any(issubclass(w.category,NumericalWarning) for w in caught)
                            result["rows"].append(row)
                            np.savez(RUN/f"{tag}_s{stream_seed}_p{row['P']}.npz",mean=mean.numpy(),latent=variance.numpy(),observation=(variance+nu).numpy(),raw_latent=(variance*scale**2).numpy(),raw_observation=((variance+nu)*scale**2).numpy())
                    result["status"] = "complete"
            assert not any(issubclass(w.category,NumericalWarning) for w in caught)
        except Exception as error:
            result.update(status="failed",error=repr(error))
        result["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
        output.write_text(json.dumps(result,indent=2))
        print(tag,result["status"],result.get("error",""),flush=True)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    manifest = json.loads((RUN/"manifest.json").read_text())
    assert all(prior.sha(Path(p).read_bytes()) == h for p,h in manifest["sources"].items())
    job = json.loads(sys.argv[1])
    assert job in manifest["diagnostic"]+manifest["qualification"]+manifest.get("final",[])
    worker(job,manifest)
