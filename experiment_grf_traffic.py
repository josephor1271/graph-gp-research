"""Official PEMS-BAY reproduction first; local checkouts/environment live in results/."""

import csv
import hashlib
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import gpytorch
import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "results/traffic_upstream/experiments"
CORE = ROOT / "results/traffic_upstream/Fast-Graph-GP"
OUTPUT = ROOT / "results/traffic"
EXPERIMENT_COMMIT = "274b7ee393ee516585552f053e9d4a9c6ad2222f"
CORE_COMMIT = "268fff9a3724cdd480dce4b2946068d8211b4f30"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(UPSTREAM / "experiments/regression/traffic_prediction"))

from experiments.utils import train_model, evaluate_model
from traffic_utils.preprocessing import load_PEMS
from grf_gp import GRFSampler, GRFGP, ExactGraphGP
from grf_gp.kernels.diffusion import DiffusionGRFKernel, DiffusionExactKernel
from grf_gp.utils.spectral import get_normalized_laplacian
from grf_gp.utils.sparse_lo import SparseLinearOperator
from dense_gp import GraphGP as HeatGP
from graph_gp import SparseHeatKernel
from demo import plt


class TrafficExactKernel(DiffusionExactKernel):
    """The traffic notebook's bounded beta; all other code is upstream (MIT)."""

    @property
    def beta(self):
        return 2 * torch.sigmoid(self.raw_beta)


def signal_variance(raw):
    return torch.nn.functional.softplus(raw).square()


def inverse_signal_variance(value):
    return torch.log(torch.expm1(value.sqrt()))


def chebyshev_kernel(laplacian):
    """Align all three raw parameters and physical starts with traffic exact."""
    base = SparseHeatKernel(laplacian.to_sparse_coo())
    base.register_constraint("raw_heat_time", gpytorch.constraints.Interval(0.0, 2.0))
    base.initialize(raw_heat_time=torch.tensor(1.0))
    constraint = gpytorch.constraints.Positive(transform=signal_variance, inv_transform=inverse_signal_variance)
    kernel = gpytorch.kernels.ScaleKernel(base, outputscale_constraint=constraint)
    kernel.initialize(raw_outputscale=torch.tensor(1.0))
    return kernel


def posterior(covariance, y, noise):
    """Observed-block Cholesky NLL, posterior mean and latent marginal variance."""
    m = y.numel()
    factor = torch.linalg.cholesky(covariance[:m, :m] + noise * torch.eye(m))
    alpha = torch.cholesky_solve(y[:, None], factor)[:, 0]
    nll = (y @ alpha + 2 * factor.diagonal().log().sum() + m * np.log(2 * np.pi)) / (2 * m)
    cross = covariance[m:, :m]
    mean = cross @ alpha
    variance = covariance.diagonal()[m:] - (cross * torch.cholesky_solve(cross.T, factor).T).sum(dim=1)
    assert (variance > 0).all()
    return nll, mean, variance


def score(mean, latent_variance, target, noise, scale):
    observation_variance = latent_variance + noise
    rmse = scale * (target - mean).square().mean().sqrt()
    nlpd = (0.5 * (np.log(2 * np.pi) + observation_variance.log() + (target-mean).square()/observation_variance)).mean()
    return rmse.item(), nlpd.item()


def numerical_checks(laplacian, train, test, y, walk_matrices):
    weights = np.array([[0., 1., 0., 0.], [1., 0., 2., 0.], [0., 2., 0., 3.], [0., 0., 3., 0.]])
    small = torch.from_numpy(get_normalized_laplacian(weights))
    for graph, ids, targets in [(small, torch.arange(4), y[:2]), (laplacian, torch.cat([train, test]), y)]:
        exact, cheb = TrafficExactKernel(graph), chebyshev_kernel(graph)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        exact_full = exact._full_kernel_matrix()
        reference = exact_full[ids][:, ids]
        candidate = cheb(ids).to_dense()
        reference_loss = posterior(reference, targets, likelihood.noise)[0]
        candidate_loss = posterior(candidate, targets, likelihood.noise)[0]
        torch.testing.assert_close(candidate_loss, reference_loss, atol=3e-12, rtol=3e-11)
        exact_parameters = (exact.raw_beta, exact.raw_sigma_f, likelihood.raw_noise)
        cheb_parameters = (cheb.base_kernel.raw_heat_time, cheb.raw_outputscale, likelihood.raw_noise)
        reference_gradients = torch.autograd.grad(reference_loss, exact_parameters)
        candidate_gradients = torch.autograd.grad(candidate_loss, cheb_parameters)
        torch.testing.assert_close(candidate, reference.detach(), atol=3e-12, rtol=3e-11)
        for actual, expected in zip(candidate_gradients, reference_gradients):
            torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-10)
        rhs = torch.randn(graph.shape[0], 3, generator=torch.Generator().manual_seed(123))
        action_error = cheb(torch.arange(graph.shape[0])) @ rhs - exact_full.detach() @ rhs
        print(f"Oracle n={graph.shape[0]}: block max/relative errors={(candidate-reference.detach()).abs().max():.3e}/"
              f"{(candidate-reference.detach()).norm()/reference.detach().norm():.3e}; action max/relative="
              f"{action_error.abs().max():.3e}/{action_error.norm()/(exact_full.detach()@rhs).norm():.3e}; "
              f"NLL error={(candidate_loss-reference_loss).item():+.3e}; "
              f"raw beta/sigma/noise gradient errors={[float((a-b).abs().max()) for a,b in zip(candidate_gradients,reference_gradients)]}", flush=True)
    with torch.no_grad():
        traffic_reference = TrafficExactKernel(laplacian)
        beta, amplitude = traffic_reference.beta, traffic_reference.sigma_f
        q = torch.eye(laplacian.shape[0]) - beta/2*laplacian + beta.square()/8*(laplacian@laplacian)
        truncated = amplitude.square() * (q@q.T)[train][:, train]
        heat = traffic_reference._full_kernel_matrix()[train][:, train]
        grf = DiffusionGRFKernel(walk_matrices, 3)
        grf.initialize(raw_beta=torch.log(torch.expm1(beta)), raw_sigma_f=1.0)
        sampled = grf(train).to_dense()
        print(f"Fixed beta={beta:.6f}, sigma_f={amplitude:.6f}: truncation vs heat max/relative="
              f"{(truncated-heat).abs().max():.6f}/{(truncated-heat).norm()/heat.norm():.6f}; "
              f"GRF vs q2-squared max/relative={(sampled-truncated).abs().max():.6f}/{(sampled-truncated).norm()/truncated.norm():.6f}")
        for power, matrix in enumerate(walk_matrices):
            target = torch.linalg.matrix_power(laplacian, power)
            print(f"Sampled walk matrix {power} vs L^{power}, relative error={(matrix.to_dense()-target).norm()/target.norm():.6f}")


def evaluate_common(model, row, data, train, test, y, yt, laplacian):
    model.eval()
    kernel = model.covar_module
    is_grf = isinstance(model, GRFGP)
    is_cheb = isinstance(model, HeatGP)
    beta = kernel.base_kernel.heat_time if is_cheb else kernel.beta
    amplitude = kernel.outputscale.sqrt() if is_cheb else kernel.sigma_f
    noise = model.likelihood.noise
    ids = torch.cat([train, test])
    clock = perf_counter()
    with torch.no_grad():
        covariance = kernel(ids).to_dense()
        nll, mean, variance = posterior(covariance, y, noise)
        row.update(beta=beta.item(), sigma_f=amplitude.item(), noise=noise.item(), kernel_nll=nll.item())
        oracle_rmse, oracle_nlpd = score(mean, variance, yt, noise, float(data["train_std"]))
        heat = amplitude.square()*torch.matrix_exp(-beta*laplacian)[ids][:, ids]
        row["heat_nll_same_parameters"] = posterior(heat, y, noise)[0].item()
    row["audit_s"] = perf_counter()-clock
    row.update(oracle_rmse_mph=oracle_rmse, oracle_nlpd_norm=oracle_nlpd, pred_s=0.0)
    if is_grf:
        metrics = []
        for sample_seed in [100, 101]:
            torch.manual_seed(sample_seed)
            clock = perf_counter()
            with gpytorch.settings.cg_tolerance(1e-4):
                samples = model.predict_sample(test, n_samples=512)
            elapsed = perf_counter()-clock
            sample_mean, sample_variance = samples.mean(dim=0), samples.var(dim=0, unbiased=True)
            metrics.append(score(sample_mean, sample_variance, yt, noise, float(data["train_std"])))
            row[f"mc{sample_seed}_rmse"], row[f"mc{sample_seed}_nlpd"] = metrics[-1]
            row[f"mc{sample_seed}_seconds"] = elapsed
            row["pred_s"] += elapsed
            print(f"{row['method']} MC seed={sample_seed}: mean/variance RMSE vs own Phi posterior="
                  f"{(sample_mean-mean).square().mean().sqrt():.5f}/{(sample_variance-variance).square().mean().sqrt():.5f}; "
                  f"common mph RMSE/NLPD={metrics[-1]}", flush=True)
        row["rmse_mph"], row["marginal_nlpd_norm"] = np.mean(metrics, axis=0)
        # Identical perturbations isolate default-CG error from posterior MC noise.
        torch.manual_seed(100)
        default_samples = model.predict_sample(test, n_samples=512)
        torch.manual_seed(100)
        with gpytorch.settings.cg_tolerance(1e-4):
            controlled_samples = model.predict_sample(test, n_samples=512)
        print(f"{row['method']} native CG=1 vs common CG=1e-4, same draws: max difference={(default_samples-controlled_samples).abs().max():.3e}")
    else:
        clock = perf_counter()
        with torch.no_grad():
            _, mean, variance = posterior(kernel(ids).to_dense(), y, noise)
            row["rmse_mph"], row["marginal_nlpd_norm"] = score(mean, variance, yt, noise, float(data["train_std"]))
        row["pred_s"] = perf_counter()-clock
    row["marginal_nlpd_mph"] = row["marginal_nlpd_norm"] + np.log(float(data["train_std"]))
    row["total_s"] = float(data["preprocessing_seconds"]) + row["walk_s"] + row["setup_s"] + row["train_s"] + row["author_eval_s"] + row["pred_s"]
    row["status"] = "complete"
    print(row, flush=True)
    return row


def load_data(seed):
    path = OUTPUT / f"seed{seed}.npz"
    if not path.exists():
        np.random.seed(seed)
        torch.manual_seed(seed)
        clock = perf_counter()
        graph, (train, y), (test, yt), _ = load_PEMS(num_train=250, path=str(UPSTREAM / "data/traffic"))
        assert list(graph.nodes) == list(range(len(graph)))
        adjacency = nx.to_numpy_array(graph)
        laplacian = get_normalized_laplacian(adjacency)
        np.savez(path, train_ids=train.astype(np.int64).ravel(), test_ids=test.astype(np.int64).ravel(),
                 y_train=((y-y.mean())/y.std()).ravel(), y_test=((yt-y.mean())/y.std()).ravel(),
                 train_mean=y.mean(), train_std=y.std(), laplacian=laplacian,
                 adjacency_hash=hashlib.sha256(adjacency.tobytes()).hexdigest(),
                 laplacian_hash=hashlib.sha256(laplacian.tobytes()).hexdigest(), data_seed=seed,
                 experiment_commit=EXPERIMENT_COMMIT, core_commit=CORE_COMMIT,
                 preprocessing_seconds=perf_counter()-clock)
    with np.load(path) as archive:
        data = dict(archive)
    assert str(data["experiment_commit"]) == EXPERIMENT_COMMIT and str(data["core_commit"]) == CORE_COMMIT
    assert not np.intersect1d(data["train_ids"], data["test_ids"]).size
    assert hashlib.sha256(data["laplacian"].tobytes()).hexdigest() == str(data["laplacian_hash"])
    spectrum = np.linalg.eigvalsh(data["laplacian"])
    assert np.allclose(data["laplacian"], data["laplacian"].T, atol=1e-14)
    assert spectrum.min() > -1e-12 and spectrum.max() <= 2+1e-12
    print(f"seed={seed}, n={len(spectrum)}, m={len(data['train_ids'])}, t={len(data['test_ids'])}; "
          f"spectrum={spectrum[[0,-1]]}, no overlap; train mean/std={data['train_mean']}/{data['train_std']}", flush=True)
    print(f"Weighted adjacency SHA256={data['adjacency_hash']}; normalized L SHA256={data['laplacian_hash']}", flush=True)
    return data


def main():
    OUTPUT.mkdir(exist_ok=True)
    for path, expected in [(UPSTREAM, EXPERIMENT_COMMIT), (CORE, CORE_COMMIT)]:
        assert subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() == expected
    torch.set_default_dtype(torch.float64)  # The upstream sweep's declared dtype.
    torch.set_num_threads(4)
    seed = 2 if "--seed2" in sys.argv else 0
    data = load_data(seed)
    train, test = torch.from_numpy(data["train_ids"]), torch.from_numpy(data["test_ids"])
    y, yt = torch.from_numpy(data["y_train"]), torch.from_numpy(data["y_test"])
    laplacian = torch.from_numpy(data["laplacian"])
    print(f"Native defaults: Cholesky threshold={gpytorch.settings.max_cholesky_size.value()}, "
          f"CG={gpytorch.settings.cg_tolerance.value()}, eval CG={gpytorch.settings.eval_cg_tolerance.value()}, "
          f"probes={gpytorch.settings.num_trace_samples.value()}, Lanczos={gpytorch.settings.max_lanczos_quadrature_iterations.value()}, "
          f"fast log_prob={gpytorch.settings.fast_computations.log_prob.on()}")
    print("Traffic notebook does not invoke set_gp_defaults; optimizer AdamW lr=.05, GRF 1000 steps, 256 walks/node, halt .5, cutoff 3, four sampler processes.")
    np.random.seed(seed)
    torch.manual_seed(seed)
    native_path = OUTPUT/f"native{seed}.csv"
    if native_path.exists():
        with native_path.open() as handle:
            native_row = next(csv.DictReader(handle))
        for key in native_row:
            if key not in ["method", "status"]:
                native_row[key] = float(native_row[key])
        walk_matrices = [SparseLinearOperator(tensor) for tensor in torch.load(OUTPUT/f"walks{seed}.pt", weights_only=True)]
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GRFGP(train, y, likelihood, DiffusionGRFKernel(walk_matrices, 3))
        model.load_state_dict(torch.load(OUTPUT/f"native_grf{seed}.pt", weights_only=True))
        print("Reusing completed native reproduction, not retraining or resampling", flush=True)
    else:
        clock = perf_counter()
        sampler = GRFSampler(laplacian, 256, 0.5, 3, seed=seed, use_tqdm=False, n_processes=4)
        print(f"Upstream sampler effective base seed={sampler.seed} (source uses seed or 42)", flush=True)
        walk_matrices = sampler.sample_random_walk_matrices()
        walk_seconds = perf_counter()-clock
        torch.save([operator.sparse_csr_tensor for operator in walk_matrices], OUTPUT/f"walks{seed}.pt")
        clock = perf_counter()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GRFGP(train, y, likelihood, DiffusionGRFKernel(walk_matrices, 3))
        setup_seconds = perf_counter()-clock
        clock = perf_counter()
        losses = train_model(model, likelihood, train, y, lr=0.05, max_iter=1000, print_every=100, progress_desc="Native GRF diffusion")
        train_seconds = perf_counter()-clock
        torch.save(model.state_dict(), OUTPUT/f"native_grf{seed}.pt")
        np.save(OUTPUT/f"native_losses{seed}.npy", losses)
        clock = perf_counter()
        author_lml, author_rmse, author_nlpd = evaluate_model(model, likelihood, train, y, test, yt, float(data["train_std"]))
        native_row = dict(seed=seed, method="GRF native", steps=1000, walks=256, walk_s=walk_seconds,
                          setup_s=setup_seconds, train_s=train_seconds, author_eval_s=perf_counter()-clock,
                          author_lml=author_lml, author_rmse_mph=author_rmse, author_nlpd=author_nlpd)
        with native_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(native_row))
            writer.writeheader()
            writer.writerow(native_row)
    print(native_row, flush=True)
    if "--native-only" in sys.argv:
        return
    numerical_checks(laplacian, train, test, y, walk_matrices)
    rows_path = OUTPUT/"traffic_rows.csv"
    rows = []
    if rows_path.exists():
        with rows_path.open() as handle:
            rows = list(csv.DictReader(handle))
    cases = [("GRF P1000", 1000), ("Exact P100", 100), ("Cheb M100", 100), ("GRF M100", 100), ("Cheb CG100", 100)]
    if "--long" in sys.argv:
        cases.append(("Cheb M1000", 1000))
    native_model = model
    for method, steps in cases:
        if any(int(float(r["seed"])) == seed and r["method"] == method for r in rows):
            print(f"Already recorded seed={seed}, {method}; not repeated", flush=True)
            continue
        print(f"START {method}, seed={seed}, steps={steps}", flush=True)
        if method == "GRF P1000":
            model = native_model
            row = dict(native_row)
            row["method"] = method
        else:
            clock = perf_counter()
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            if method == "Exact P100":
                model = ExactGraphGP(train, y, likelihood, TrafficExactKernel(laplacian))
            elif method == "GRF M100":
                kernel = DiffusionGRFKernel(walk_matrices, 3)
                kernel.initialize(raw_beta=torch.log(torch.expm1(2*torch.sigmoid(torch.tensor(1.0)))))
                model = GRFGP(train, y, likelihood, kernel)
            else:
                model = HeatGP(train, y, likelihood, chebyshev_kernel(laplacian))
            row = dict(seed=seed, method=method, steps=steps, walks=256 if method == "GRF M100" else 0,
                       walk_s=float(native_row["walk_s"]) if method == "GRF M100" else 0.0,
                       setup_s=perf_counter()-clock, author_eval_s=0.0)
            torch.manual_seed(seed)
            clock = perf_counter()
            with (
                gpytorch.settings.max_cholesky_size(0 if method == "Cheb CG100" else 800),
                gpytorch.settings.cg_tolerance(1e-4 if method == "Cheb CG100" else 1.0),
                gpytorch.settings.num_trace_samples(64 if method == "Cheb CG100" else 10),
                gpytorch.settings.max_lanczos_quadrature_iterations(24 if method == "Cheb CG100" else 20),
                gpytorch.settings.max_preconditioner_size(0),
                gpytorch.settings.fast_computations(log_prob=True),
            ):
                losses = train_model(model, likelihood, train, y, lr=.05, max_iter=steps, print_every=100, progress_desc=method)
                row["train_s"] = perf_counter()-clock
                row["reported_nll"] = -gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)(model(train), y).item()
            if method == "Exact P100":
                clock = perf_counter()
                lml, rmse, nlpd = evaluate_model(model, likelihood, train, y, test, yt, float(data["train_std"]))
                row.update(author_lml=lml, author_rmse_mph=rmse, author_nlpd=nlpd, author_eval_s=perf_counter()-clock)
            torch.save(model.state_dict(), OUTPUT/f"{method.replace(' ', '_')}_{seed}.pt")
            np.save(OUTPUT/f"{method.replace(' ', '_')}_{seed}_losses.npy", losses)
        rows.append(evaluate_common(model, row, data, train, test, y, yt, laplacian))
        fields = []
        for entry in rows:
            for field in entry:
                if field not in fields:
                    fields.append(field)
        with rows_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, restval="nan")
            writer.writeheader()
            writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    for row in rows:
        marker = "o" if int(float(row["seed"])) == 0 else "^"
        color = f"C{methods.index(row['method'])}"
        # CSV total_s also includes shared preprocessing/validation. Exclude that
        # common cost from this plot; all measured method-specific work remains.
        method_seconds = sum(float(row[field]) for field in ["walk_s", "setup_s", "train_s", "author_eval_s", "pred_s"])
        for axis, metric in zip(axes, ["rmse_mph", "marginal_nlpd_norm"]):
            axis.scatter(method_seconds, float(row[metric]), marker=marker, color=color,
                         label=row["method"] if marker == "o" else None)
    axes[0].set(ylabel="Held-out RMSE (mph)")
    axes[1].set(ylabel="Marginal observation NLPD (normalized targets)")
    for axis in axes:
        axis.set(xlabel="Method total seconds (shared data setup excluded)", xscale="log")
        axis.legend(fontsize=7)
    figure.suptitle("Common metrics; P: source training, M: aligned physical starts; circles: seed 0, triangles: seed 2")
    figure.tight_layout()
    figure.savefig(OUTPUT/"grf_traffic.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
