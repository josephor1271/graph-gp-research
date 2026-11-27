"""Degree-two operator pilot. Run from the repository root in the traffic environment."""
import json
import resource
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import gpytorch
import linear_operator.utils as utils
import numpy as np
import scipy.sparse as sp
import torch
import experiment_learned_kernel_numerical_learning as learning
from experiment_learned_polynomial_traffic import LearnedPolynomialKernel
from grf_gp.kernels.base import BaseGRFKernel

prior = learning.prior
OUT = Path("results/operator_pilot")
SEEDS = (217101, 517101, 817101)


def root_coeff(kernel):
    return torch.stack((torch.ones_like(kernel.a1), kernel.a1-4*kernel.a2, 2*kernel.a2))


def square_coeff(kernel):
    a,b,c = root_coeff(kernel)
    return torch.stack((a*a, 2*a*b, b*b+2*a*c, 2*b*c, c*c))


def horner(L, coeff, rhs):
    x = coeff[-1]*rhs
    for c in coeff[:-1].flip(0):
        x = torch.sparse.mm(L,x)+c*rhs
    return x


class MatchedFeatures(LearnedPolynomialKernel):
    """Use official feature action with Poly's two normalized shape parameters."""
    modulation_function = property(root_coeff)
    _get_feature_matrix = BaseGRFKernel._get_feature_matrix
    forward = BaseGRFKernel.forward


def make(method,L,ids,y,mats):
    model = prior.model_for("General" if method=="Native" else "Poly", L, mats, ids, y[ids])
    if method in ("Powers","Walk"):
        base = MatchedFeatures(L)
        base.rw_mats = mats
        model.covar_module.base_kernel = base
    return model


def basis(method,L,w,seed):
    if method=="Walk":
        return prior.GRFSampler(L,w,.5,3,seed=seed,n_processes=4,use_tqdm=False)()
    if method!="Powers":
        return None
    csr = L.to_sparse_csr()
    degrees = csr.crow_indices().diff()
    assert (degrees*degrees).sum().item() <= 100*L.size(0), "Sparse square fill bound"
    a = sp.csr_matrix((csr.values().numpy(),csr.col_indices().numpy(),csr.crow_indices().numpy()),shape=L.shape)
    powers = [sp.eye(L.size(0),format="csr"),a,a@a]
    for power in powers:
        power.sort_indices()
    return [prior.SparseLinearOperator(torch.sparse_csr_tensor(p.indptr,p.indices,p.data,size=p.shape)) for p in powers]


@contextmanager
def coordinates(method):
    if method=="Mono":
        with patch("graph_gp.apply_chebyshev",horner), patch.object(LearnedPolynomialKernel,"coefficients",property(square_coeff)):
            yield
    else:
        yield


def run():
    records, checks, fits = [], [], []
    for n,family in [(36,"road"),(1000,"road"),(1000,"ring"),(5000,"road"),(5000,"ring")]:
        if family=="road":
            with patch.object(prior,"OUT",OUT):
                L,ys,train,test,scale,_,fp,_ = prior.inputs(n,12)
        else:
            nodes = torch.arange(n)
            ij = torch.stack((nodes.repeat(3),torch.cat((nodes,(nodes-1)%n,(nodes+1)%n))))
            L = torch.sparse_coo_tensor(ij,torch.tensor([1.,-.5,-.5]).repeat_interleave(n),(n,n)).coalesce()
            train = torch.randperm(n,generator=torch.Generator().manual_seed(751000+n+1200000))[:n//4]
            ys,fp = torch.zeros(2,n),prior.sha(L.indices().numpy().tobytes()+L.values().numpy().tobytes())
        saved = {}
        cases = [(m,0,0) for m in ("Cheb","Mono","Powers")]+[("Walk",w,s) for w in (100,256,1024) for s in SEEDS]
        for method,w,seed in cases:
            if n==36 and method=="Walk" and (seed!=SEEDS[0] or w!=100):
                continue
            t = perf_counter()
            mats = basis(method,L,w,seed)
            setup = perf_counter()-t
            if seed in (0,SEEDS[0]):
                saved[method,w] = mats
            tensors = [m.sparse_csr_tensor for m in mats] if mats else [L]
            nnz = [m._nnz() for m in tensors]
            storage = 0
            for m in tensors:
                storage += m.values().nbytes+(m.indices().nbytes if m.layout==torch.sparse_coo else m.crow_indices().nbytes+m.col_indices().nbytes)
            with coordinates(method):
                start = perf_counter()
                model = make(method,L,train,ys[0],mats)
                model_setup = perf_counter()-start
                kernel = model.covar_module
                for state in ("initial","stress"):
                    if state=="stress":
                        prior.second_state(model,"Poly")
                    for width in (1,33):
                        rhs = torch.randn(len(train),width,generator=torch.Generator().manual_seed(702+width))
                        times = []
                        for repeat in range(6):
                            start = perf_counter()
                            product = kernel(train).evaluate_kernel() @ rhs
                            middle = perf_counter()
                            grads = torch.autograd.grad(product.square().mean(),tuple(kernel.parameters()))
                            times.append([middle-start,perf_counter()-middle])
                        if method=="Cheb":
                            saved[state,width] = product.detach()
                        error = ((product-saved[state,width]).norm()/saved[state,width].norm()).item()
                        if method!="Walk":
                            torch.testing.assert_close(product,saved[state,width],atol=1e-10,rtol=1e-9)
                        if n==36:
                            b = kernel.base_kernel
                            if method=="Walk":
                                phi = sum(c*m.sparse_csr_tensor.to_dense() for c,m in zip(root_coeff(b),mats))
                                K = phi @ phi.T
                            else:
                                eig,U = torch.linalg.eigh(L.to_dense())
                                q = 1+b.a1*eig+b.a2*(2*(eig-1).square()-2)
                                K = (U*q.square()) @ U.T
                            ref = kernel.outputscale*K[train][:,train] @ rhs
                            expected = torch.autograd.grad(ref.square().mean(),tuple(kernel.parameters()))
                            torch.testing.assert_close(product,ref,atol=1e-10,rtol=1e-9)
                            for a,z in zip(grads,expected):
                                torch.testing.assert_close(a,z,atol=1e-9,rtol=1e-8)
                            checks.append([method,state,width,(product-ref).abs().max().item(),max((a-z).abs().max().item() for a,z in zip(grads,expected))])
                        records.append(dict(n=n,family=family,fingerprint=fp,method=method,W=w,seed=seed,state=state,width=width,
                            preprocessing_s=setup,model_setup_s=model_setup,nnz=nnz,bytes=storage,relative_error=error,seconds=times[1:]))
        if n==1000 and family=="road":
            dev = L,ys,train,test,scale,saved
    (OUT/"actions.json").write_text(json.dumps(records))
    (OUT/"checks.json").write_text(json.dumps(checks))
    L,ys,train,test,scale,saved = dev
    for g,y in enumerate(ys):
        for method,w in [("Cheb",0),("Mono",0),("Powers",0),("Walk",100),("Walk",1024),("Native",100),("Native",1024)]:
            mats = saved["Walk" if method=="Native" else method,w]
            counts = []
            original = utils.linear_cg
            def counted(action,*args,**kwargs):
                calls = 0
                def multiply(rhs):
                    nonlocal calls
                    calls += 1
                    return action(rhs)
                answer = original(multiply,*args,**kwargs)
                counts.append(calls-1)  # exclude initial residual product
                return answer
            with coordinates(method), patch.object(utils,"linear_cg",counted):
                model = make(method,L,train,y,mats)
                row = learning.fit(model,train,test,y,scale[g],32,16,91)
            row.update(regime="HM"[g],method=method,W=w,cg_iterations=counts)
            fits.append(row)
            (OUT/"fits.json").write_text(json.dumps(fits))
            assert row["status"]=="complete" and not row["numerical_warning"] and row["residual"]<=2e-4 and row["mean_error"]<=.002
            print(row["regime"],method,w,row["training_s"],row["exact_rmse"],flush=True)
    print("whole-process peak RSS bytes",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def export():
    import matplotlib.pyplot as plt
    import pandas as pd
    fig, axes = plt.subplots(2,2,figsize=(8,5),sharex=True,layout="constrained")
    with (OUT/"measurements.jsonl").open("w") as raw:
        for stage in ("actions","checks","fits"):
            for row in json.loads((OUT/f"{stage}.json").read_text()):
                print(json.dumps({stage:row},allow_nan=False),file=raw)
    actions = pd.read_json(OUT/"actions.json")
    walks = actions[(actions.method=="Walk") & (actions.width==33) & (actions.n>36)]
    for ax, ((family,n), part) in zip(axes.flat,walks.groupby(["family","n"])):
        for state, group in part.groupby("state"):
            err = group.groupby("W").relative_error.agg(["median","min","max"])
            ax.plot(err.index,err["median"],".-",label=state)
            ax.fill_between(err.index,err["min"],err["max"],alpha=.15)
        ax.set(title=f"{family}, {n:,} vertices",xscale="log",yscale="log",xticks=[100,256,1024],xticklabels=["100","256","1024"])
    axes[0,0].legend()
    fig.supxlabel("Walks per vertex")
    fig.supylabel("Relative action error (33 RHS)")
    fig.savefig(OUT/"walk-error.png",dpi=160)


if __name__=="__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=False)
    with gpytorch.settings.max_eager_kernel_size(0):
        run()
    export()
