"""Frozen degree-four, mean-only follow-up to PR17. CPU float64."""
import json
import signal
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import gpytorch
import numpy as np
import torch
from numpy.polynomial import Chebyshev, Polynomial
from linear_operator.utils.warnings import NumericalWarning
import experiment_learned_kernel_numerical_learning as learning
from experiment_learned_kernel_scaling_final import write
from experiment_learned_polynomial_traffic import GeneralGRFKernel
from experiment_matched_posterior import load
from experiment_poly_posterior import sparse
from experiment_spectral_expressivity import Poly4
from spectral_checks import check

prior = learning.prior
OUT = Path("results/higher_degree_scaling")
DEV = [(r, g, w) for r in (10, 11) for g in (0, 1) for w in (0, 1024, 4096)]


def walks(graph, r, w):
    if not w:
        return None, 0.
    n = graph.size(0)
    path = OUT/f"n{n}_r{r}_J5_W{w}.pt"
    meta = path.with_suffix(".json")
    if not meta.exists():
        assert not path.exists()
        start = perf_counter()
        matrices = prior.GRFSampler(graph, w, .5, 5, seed=771000+n+100000*r, use_tqdm=False, n_processes=4)()
        seconds = perf_counter()-start
        tensors = [m.sparse_csr_tensor for m in matrices]
        torch.save(tensors, path)
        write(meta, dict(seconds=seconds, nnz=[m._nnz() for m in tensors], sha=prior.sha(path.read_bytes())))
    record = json.loads(meta.read_text())
    assert record["sha"] == prior.sha(path.read_bytes())
    return [prior.SparseLinearOperator(m) for m in torch.load(path, weights_only=True)], record["seconds"]


def model4(L, train, y, matrices):
    torch.manual_seed(0)
    arm = "General" if matrices else "Poly"
    model = prior.model_for(arm, L, matrices, train, y[train])
    if matrices:
        initial = model.covar_module.modulation_function.detach()
        model.covar_module = GeneralGRFKernel(matrices, 5)
        model.covar_module.initialize(raw_modulation_function=torch.cat((initial, initial.new_zeros(2))))
    else:
        model.covar_module.base_kernel = Poly4(model.covar_module.base_kernel.laplacian)
    assert sum(p.numel() for p in model.parameters()) == 6 and abs(model.likelihood.noise.item()-.6932471805599453)<1e-14
    return model


def algebra():
    check()
    weights = torch.diag(torch.arange(1., 6), 1)
    weights = weights + weights.T
    d = weights.sum(1).rsqrt()
    L = (torch.eye(6)-d[:, None]*weights*d).to_sparse()
    train, rows, y = torch.tensor([0, 2, 4]), torch.tensor([3, 1, 1, 0]), torch.arange(6.)/6
    matrices, _ = walks(L, 0, 1024)
    poly, grf = (model4(L, train, y, m) for m in (None, matrices))
    for stress in (False, True):
        kernel = poly.covar_module
        if stress:
            kernel.base_kernel.initialize(a1=-.35, a2=.15, a3=.03, a4=.01)
            kernel.outputscale = 2.25
            grf.likelihood.noise = .05
        a = np.array([getattr(kernel.base_kernel, f"a{j}").item() for j in range(1, 5)])
        q = Chebyshev(np.r_[1-a@np.array([-1,1,-1,1]), a], domain=[0,2]).convert(kind=Polynomial)
        coeff = torch.tensor(np.pad(q.coef,(0,5-len(q.coef))))*kernel.outputscale.detach().sqrt()
        if stress:
            grf.covar_module.initialize(raw_modulation_function=coeff)
        mod = grf.covar_module.modulation_function
        phi = sum(c*m.sparse_csr_tensor.to_dense() for c,m in zip(mod,matrices))
        K = phi @ phi.T
        assert torch.linalg.eigvalsh(K.detach()).min() >= -1e-10
        H = K[train][:,train]+grf.likelihood.noise*torch.eye(len(train))
        ref = -torch.distributions.MultivariateNormal(torch.zeros(3), H).log_prob(y[train])/3
        loss = -gpytorch.mlls.ExactMarginalLogLikelihood(grf.likelihood,grf)(grf(train),y[train])
        op, rhs = grf.covar_module(rows,train), torch.arange(6.).reshape(3,2)
        pairs = [(mod,coeff),(grf.covar_module(torch.arange(6)).to_dense(),K),(op@rhs,K[rows][:,train]@rhs),
                 (op.mT.to_dense(),K[train][:,rows]),(loss,ref)]
        grads = [learning.gradient(z,grf) for z in (loss,ref)]
        for x,z in pairs:
            torch.testing.assert_close(x,z,atol=1e-10,rtol=1e-9)
        torch.testing.assert_close(*grads,atol=1e-9,rtol=1e-8)
        print("J5 initial/stress",stress,"value errors",[(x-z).abs().max().item() for x,z in pairs],"gradient",(grads[0]-grads[1]).abs().max().item())


def develop():
    algebra()
    data = {(r,g):load(1000,r,g,"Poly","fit") for r,g,w in DEV if not w}
    refs = {}
    def case(r,g,w):
        L, train, val, y, scale = data[r,g][:5]
        matrices, _ = walks(L,r,w)
        return model4(L,train,y,matrices), train, val, y, scale
    for r,g,w in DEV:
        args = case(r,g,w)
        row = learning.fit(*args,32,16,0)
        write(OUT/f"reference_{r}_{g}_{w}.json",row)
        assert row["status"] == "complete" and not row["numerical_warning"] and row["residual"] <= 2e-4
        refs[r,g,w] = row
    W = 1024 if all(refs[r,g,1024]["exact_rmse"] <= 1.01*refs[r,g,4096]["exact_rmse"] for r,g in data) else 4096
    write(OUT/"walk_selection.json",dict(W=W))
    for probes,lanczos in learning.MENU:
        passed = True
        for r,g,w in (k for k in DEV if k[2] in (0,W)):
            model,train,val,y,_ = case(r,g,w)
            endpoint = {k:torch.tensor(v) for k,v in refs[r,g,w]["final"].items()}
            with patch.object(prior,"second_state",lambda m,arm:m.initialize(**endpoint)), patch.object(learning,"MENU",[(probes,lanczos)]):
                groups = learning.numerical(model,"General" if w else "Poly",train,val,y)["groups"]
            write(OUT/f"numerical_{probes}_{r}_{g}_{w}.json",groups)
            assert all(t["residual"]<=2e-4 and np.isfinite([t["nll_error"],t["mean_error"],*t["errors"]]).all() and not t["warnings"] for group in groups for t in group["trials"] if not t["safe"])
            passed &= all(group["passed"] for group in groups)
        if not passed:
            continue
        for r,g,w in (k for k in DEV if k[2] in (0,W)):
            row = learning.fit(*case(r,g,w),probes,lanczos,91)
            write(OUT/f"learning_{probes}_{r}_{g}_{w}.json",row)
            assert row["status"] == "complete" and not row["numerical_warning"] and row["residual"] <= 2e-4
            ref = refs[r,g,w]
            passed &= row["nll"]-ref["nll"]<=.02 and row["exact_rmse"]<=1.02*ref["exact_rmse"] and row["mean_error"]<=.002
        if passed:
            schedule = [[n,r,g,w] for i,n in enumerate((5000,25000,50000)) for r in range(3) for g in range(2)
                        for w in ((W,0) if (i+r+g)%2 else (0,W))]
            write(OUT/"selection.json",dict(W=W,budget=[probes,lanczos],schedule=schedule))
            return
    write(OUT/"selection.json",dict(W=W,budget=None,schedule=[],reason="No common passing budget"))


def final(n,r,g,w):
    selection = json.loads((OUT/"selection.json").read_text())
    assert [n,r,g,w] in selection["schedule"]
    clock = perf_counter()
    L,train,test,y,scale,_,_,historical,identity = load(n,r,g,"General" if w else "Poly","fit")
    matrices, seconds = walks(L,r,w)
    tag = OUT/f"final_{n}_{r}_{g}_{w}"
    for guarded in ([True,False] if r==g==0 and n in (5000,50000) else [False]):
        with learning.inference(*selection["budget"]), sparse(n,len(train),len(test)) if guarded else nullcontext():
            start = perf_counter()
            model = model4(L,train,y,matrices)
            opt = torch.optim.AdamW(model.parameters(),lr=.05)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood,model)
            setup = perf_counter()-start
            initial = learning.raw(model)
            torch.manual_seed(781000+n+100000*r)
            losses, norms = [], []
            start = perf_counter()
            for _ in range(1 if guarded else 100):
                opt.zero_grad()
                loss = -mll(model(train),y[train])
                loss.backward()
                norms.append(torch.cat([p.grad.flatten() for p in model.parameters()]).norm().item())
                losses.append(loss.item())
                assert np.isfinite([norms[-1],losses[-1]]).all()
                opt.step()
                assert all(torch.isfinite(p).all() for p in model.parameters())
            training = perf_counter()-start
            model.eval()
            with torch.no_grad(), gpytorch.settings.skip_posterior_variances(True):
                start = perf_counter()
                mean = model(test).mean
                prediction = perf_counter()-start
                residual = prior.residual(model,train,y)
            assert torch.isfinite(mean).all() and residual<=2e-4
            if guarded:
                write(tag.with_suffix(".guard.json"), dict(residual=residual))
    if n==50000 and r==g==0:
        model.train()
        with learning.inference(*selection["budget"]), sparse(n,len(train),len(test)):
            replay, res = learning.prediction(model,train,test,y)
            torch.testing.assert_close(replay,mean,atol=.002,rtol=1e-8)
            assert res<=2e-4
    rmse = (mean-y[test]).square().mean().sqrt().item()
    torch.save(dict(model=model.state_dict(),mean=mean),tag.with_suffix(".pt"))
    write(tag.with_suffix(".json"),dict(status="complete",setup=setup,training=training,prediction=prediction,walk=seconds,
          rmse=scale*rmse,standardized=rmse,residual=residual,losses=losses,gradient_norm=norms,initial=initial,
          final=learning.raw(model),noise=model.likelihood.noise.item(),historical=historical,inputs=identity,wall=perf_counter()-clock))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    warnings.filterwarnings("error",category=NumericalWarning)
    signal.alarm(1800)
    if sys.argv[1]=="develop":
        assert not list(OUT.glob("reference*")) and not (OUT/"selection.json").exists()
        develop()
    else:
        final(*map(int,sys.argv[1:]))
