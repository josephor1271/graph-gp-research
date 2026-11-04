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
from grf_gp import GRFSampler, GRFGP
from grf_gp.kernels.diffusion import DiffusionGRFKernel
from grf_gp.utils.spectral import get_normalized_laplacian
from grf_gp.utils.sparse_lo import SparseLinearOperator


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
          f"probes={gpytorch.settings.num_trace_samples.value()}, Lanczos={gpytorch.settings.max_lanczos_quadrature_iterations.value()}")
    print("Traffic notebook does not invoke set_gp_defaults; optimizer AdamW lr=.05, GRF 1000 steps, 256 walks/node, halt .5, cutoff 3, four sampler processes.")
    np.random.seed(seed)
    torch.manual_seed(seed)
    clock = perf_counter()
    sampler = GRFSampler(laplacian, 256, 0.5, 3, seed=seed, use_tqdm=False, n_processes=4)
    print(f"Upstream sampler effective base seed={sampler.seed} (source uses seed or 42)", flush=True)
    walk_matrices = sampler.sample_random_walk_matrices()
    walk_seconds = perf_counter()-clock
    torch.save([operator.sparse_csr_tensor for operator in walk_matrices], OUTPUT/f"walks{seed}.pt")
    print(f"Sampling finished: {walk_seconds:.3f} seconds", flush=True)
    clock = perf_counter()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel = DiffusionGRFKernel(walk_matrices, 3)
    model = GRFGP(train, y, likelihood, kernel)
    setup_seconds = perf_counter()-clock
    print(f"Native initial beta={kernel.beta.item()}, signal std={kernel.sigma_f.item()}, noise variance={likelihood.noise.item()}", flush=True)
    clock = perf_counter()
    losses = train_model(model, likelihood, train, y, lr=0.05, max_iter=1000, print_every=100, progress_desc="Native GRF diffusion")
    train_seconds = perf_counter()-clock
    torch.save(model.state_dict(), OUTPUT/f"native_grf{seed}.pt")
    np.save(OUTPUT/f"native_losses{seed}.npy", losses)
    clock = perf_counter()
    author_lml, author_rmse, author_nlpd = evaluate_model(model, likelihood, train, y, test, yt, float(data["train_std"]))
    prediction_seconds = perf_counter()-clock
    row = dict(seed=seed, method="GRF native", steps=1000, walks=256, walk_s=walk_seconds,
               setup_s=setup_seconds, train_s=train_seconds, author_eval_s=prediction_seconds,
               beta=kernel.beta.item(), sigma_f=kernel.sigma_f.item(), noise=likelihood.noise.item(),
               author_lml=author_lml, author_rmse_mph=author_rmse, author_nlpd=author_nlpd,
               status="author-protocol complete; common metrics pending")
    with (OUTPUT/f"native{seed}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row, flush=True)


if __name__ == "__main__":
    main()
