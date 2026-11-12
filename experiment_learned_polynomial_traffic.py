"""Frozen post-hoc traffic comparison: check, 20-update pilot, then full fits."""

import csv
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import gpytorch
from linear_operator.operators import LinearOperator
import numpy as np
import torch
from experiment_grf_heldout import OUT as HELD, prepare, sha, walks
from experiment_grf_frontier import BoundedDiffusionGRFKernel
from experiment_grf_traffic import (CORE, CORE_COMMIT, UPSTREAM, EXPERIMENT_COMMIT,
    GRFGP, HeatGP, chebyshev_kernel, signal_variance, inverse_signal_variance, posterior, score, train_model)
from grf_gp.kernels.general import GeneralGRFKernel
from learned_polynomial import LearnedPolynomialKernel

OUT = Path(__file__).resolve().parent / "results/traffic/learned_polynomial"
sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
from test_learned_polynomial import check_kernel


def polynomial_kernel(graph):
    constraint = gpytorch.constraints.Positive(transform=signal_variance, inv_transform=inverse_signal_variance)
    kernel = gpytorch.kernels.ScaleKernel(LearnedPolynomialKernel(graph.to_sparse_coo()), outputscale_constraint=constraint)
    kernel.initialize(raw_outputscale=torch.tensor(1.))
    return kernel


def model_for(arm, graph, matrices, train, y):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if arm == "General":
        kernel = GeneralGRFKernel(matrices, 3)
        beta = 2*torch.sigmoid(torch.tensor(1.))
        modulation = torch.nn.functional.softplus(torch.tensor(1.))*torch.stack((beta.new_tensor(1.),-beta/2,beta.square()/8))
        kernel.initialize(raw_modulation_function=modulation)
    elif arm == "C":
        kernel = BoundedDiffusionGRFKernel(matrices, 3)
    else:
        kernel = polynomial_kernel(graph) if arm == "Poly" else chebyshev_kernel(graph)
    cls = HeatGP if arm in ("Poly", "Cheb") else GRFGP
    return cls(train, y, likelihood, kernel)


def checks(data):
    graph = torch.from_numpy(data["laplacian"])
    vertices = [0]
    while len(vertices) < 6:
        neighbors = (graph[vertices].abs().sum(0) > 0).nonzero().flatten().tolist()
        vertices.extend(v for v in neighbors if v not in vertices)
    vertices = vertices[:6]
    small = graph[vertices][:, vertices]
    diagnostics = check_kernel(polynomial_kernel(small))
    ids = torch.from_numpy(np.r_[data["train"],data["test"]])
    initial = polynomial_kernel(graph)(ids).to_dense().detach()
    discrepancies = {}
    for seed in (101,202):
        assert (HELD/f"walks_C_{seed}.pt").exists(), "Use cached PR6 walks, do not resample"
        matrices,_ = walks("C",seed,graph)
        model = model_for("General",graph,matrices,ids[:250],torch.from_numpy(data["y"][0]))
        discrepancies[seed] = float((model.covar_module(ids).to_dense().detach()-initial).norm()/initial.norm())
    (OUT/"checks.json").write_text(json.dumps(dict(slice_vertices=vertices,diagnostics=diagnostics,initial_grf_relative=discrepancies),indent=2))
    print("PASS traffic-slice gates",diagnostics,"initial GeneralGRF discrepancies",discrepancies,flush=True)


def run(data, date, arm, seed, steps, old_rows, data_sha):
    index = int(data["indices"][date])
    tag = f"{index}_{arm}_{seed}_{steps}"
    path, checkpoint = OUT/f"{tag}.json", OUT/f"{tag}.pt"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["status"] == "complete":
            assert saved["data_sha"] == data_sha and saved["steps"] == steps
            assert sha(Path(saved["checkpoint"]).read_bytes()) == saved["checkpoint_sha"]
            print("SKIP verified",tag,flush=True)
            return
    row = dict(index=index,timestamp=str(data["timestamps"][date]),arm=arm,seed=seed,steps=steps,data_sha=data_sha,status="incomplete")
    try:
        graph = torch.from_numpy(data["laplacian"])
        train,test = torch.from_numpy(data["train"]),torch.from_numpy(data["test"])
        y,yt = torch.from_numpy(data["y"][date]),torch.from_numpy(data["yt"][date])
        matrices,walk_s = walks("C",seed,graph) if seed else (None,0.)
        row.update(walk_s_historical=walk_s,walk_sha=None if not seed else sha((HELD/f"walks_C_{seed}.pt").read_bytes()))
        torch.manual_seed(0)
        clock = perf_counter()
        model = model_for(arm,graph,matrices,train,y)
        row["setup_s"] = perf_counter()-clock
        row["initial"] = {name:p.detach().tolist() for name,p in model.named_parameters()}
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood,model)
        initial_loss = -mll(model(train),y).item()
        prior = next((r for r in old_rows if int(r["index"]) == index and r["arm"] == arm and int(r["feature_seed"]) == seed),None)
        assert arm not in ("Cheb","C") or prior is not None, "PR6 baseline record required; never silently refit"
        row["timing_provenance"] = "new" if prior is None else "PR6 historical setup/train/deployment"
        if prior is not None:
            checkpoint = HELD/f"fit_{index}_{arm}_{seed}.pt"
            assert prior["status"] == "complete"
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing historical PR6 checkpoint: {checkpoint}; baseline not refitted")
        if checkpoint.exists():
            saved = torch.load(checkpoint,weights_only=True)
            assert len(saved["losses"]) == steps and np.isfinite(saved["losses"]).all()
            np.testing.assert_allclose(saved["losses"][0],initial_loss,atol=1e-11,rtol=1e-11)
            model.load_state_dict(saved["model"])
            row.update(setup_s=saved["setup_s"],train_s=saved["train_s"])
        else:
            clock = perf_counter()
            losses = train_model(model,model.likelihood,train,y,lr=.05,max_iter=steps,print_every=250,progress_desc=tag)
            row["train_s"] = perf_counter()-clock
            assert np.isfinite(losses).all()
            torch.save(dict(model=model.state_dict(),losses=losses,setup_s=row["setup_s"],train_s=row["train_s"]),checkpoint)
        row.update(checkpoint=str(checkpoint),checkpoint_sha=sha(checkpoint.read_bytes()),initial_nll=initial_loss,
            parameters={name:p.detach().tolist() for name,p in model.named_parameters()})
        kernel = model.covar_module
        row.update(a1=float(kernel.base_kernel.a1.detach()) if arm == "Poly" else None,
            a2=float(kernel.base_kernel.a2.detach()) if arm == "Poly" else None,
            signal_std=float(kernel.outputscale.detach().sqrt()) if arm in ("Poly","Cheb") else None)
        if arm == "C":
            row.update(beta=float(kernel.beta.detach()),signal_std=float(kernel.sigma_f.detach()))
        model.eval()
        noise = model.likelihood.noise.detach()
        clock = perf_counter()
        with torch.no_grad():
            covariance = model.covar_module(torch.cat([train,test])).to_dense()
            nll,mean,variance = posterior(covariance,y,noise)
            rmse,nlpd = score(mean,variance,yt,noise,float(data["std"][date]))
        row.update(oracle_s=perf_counter()-clock,nll=float(nll),rmse=rmse,nlpd=nlpd,nlpd_mph=nlpd+np.log(float(data["std"][date])),noise=float(noise))
        torch.save(dict(mean=mean,latent_variance=variance),OUT/f"{tag}_posterior.pt")
        if prior is not None:
            np.testing.assert_allclose([rmse,nlpd,float(nll),float(noise)],
                [float(prior[k]) for k in ("oracle_rmse","oracle_nlpd","kernel_nll","noise")],atol=1e-10,rtol=1e-10)
            row.update(pred_s=float(prior["pred_s"]),status="complete",historical_row=prior)
        else:
            clock = perf_counter()
            with torch.no_grad():
                if arm == "Poly":
                    prediction = model(test)
                    deployed_mean,deployed_variance = prediction.mean,prediction.variance
                else:
                    torch.manual_seed(100)
                    with gpytorch.settings.cg_tolerance(1e-4):
                        samples = model.predict_sample(test,n_samples=512)
                    deployed_mean,deployed_variance = samples.mean(0),samples.var(0,unbiased=True)
            row.update(pred_s=perf_counter()-clock,deployed_mean_rmse_error=float((deployed_mean-mean).square().mean().sqrt()),
                deployed_variance_rmse_error=float((deployed_variance-variance).square().mean().sqrt()))
            if arm == "Poly":
                torch.testing.assert_close(deployed_mean,mean,atol=1e-9,rtol=1e-8)
                torch.testing.assert_close(deployed_variance,variance,atol=1e-9,rtol=1e-8)
            row.update(deployed_scores=score(deployed_mean,deployed_variance,yt,noise,float(data["std"][date])),status="complete")
        print(row,flush=True)
    except BaseException as exc:
        row.update(status="failed",error=repr(exc))
        raise
    finally:
        path.write_text(json.dumps(row,indent=2,allow_nan=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    stage = sys.argv[1] if len(sys.argv)>1 else "check"
    assert stage in ("check","pilot","full")
    for source,revision in ((CORE,CORE_COMMIT),(UPSTREAM,EXPERIMENT_COMMIT)):
        assert subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()==revision
        subprocess.run(["git","-C",str(source),"diff","--exit-code"],check=True)
    assert (HELD/"data.npz").exists(), "Use the frozen data, do not select new snapshots"
    data = prepare()
    manifest = json.loads((HELD/"manifest.json").read_text())
    assert manifest["data_sha"] == "802e5e523b5db6ff9d13da3388c566de7d0212b09b23497a3d0e50a0385ad71e"
    (OUT/f"{stage}_manifest.json").write_text(json.dumps(dict(frozen=manifest,script_sha=sha(Path(__file__).read_bytes()),
        kernel_sha=sha(Path(__file__).with_name("learned_polynomial.py").read_bytes()),torch=torch.__version__,gpytorch=gpytorch.__version__),indent=2))
    original_dense = LinearOperator.to_dense

    def selected_dense(operator, *args, **kwargs):
        assert operator.shape != data["laplacian"].shape, "Full graph kernel/identity forbidden"
        return original_dense(operator,*args,**kwargs)

    with gpytorch.settings.max_cholesky_size(800), gpytorch.settings.max_eager_kernel_size(0), patch.object(LinearOperator,"to_dense",selected_dense):
        checks(data)
        if stage != "check":
            old_rows = list(csv.DictReader((HELD/"rows.csv").open()))
            cases = [(0,"Poly",0)] if stage == "pilot" else [(d,a,s) for d in range(3) for a,s in
                [("Cheb",0),("C",101),("C",202),("Poly",0),("General",101),("General",202)]]
            for date,arm,seed in cases:
                run(data,date,arm,seed,20 if stage == "pilot" else 1000,old_rows,manifest["data_sha"])
