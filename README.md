# A readable graph heat GP

Question: can a sparse Chebyshev heat covariance replace a dense heat kernel in
the same ordinary GPyTorch ExactGP, with checked values and heat-time gradients?

This is a small research example, not a toolbox or a performance study.
For an unweighted undirected grid, use `L = (D - A) / maximum_degree` with
spectrum in `[0, 2]`. This scalar scaling is different from symmetric degree
normalization. The covariance is `exp(-tau L)`, amplitude one, zero mean, and
Gaussian observation **variance** 0.04. All calculations use CPU float64;
trainable heat time is restricted to `(0.05, 4)`.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python demo.py
```

`dense_gp.py` contains the ordinary ExactGP and a small-graph eigendecomposition
reference. `demo.py` constructs a 6 × 6 grid, draws an exact synthetic GP at
tau=1.4 with seed 12, fits heat time from 24 observations, and predicts the 12
held-out vertices. It writes one figure and its inputs under ignored `results/`.
Fitting one draw is a usage demonstration, not a parameter-recovery guarantee.

Initial stages: (1) dense GP, (2) differentiable Chebyshev recurrence,
(3) sparse GPyTorch integration. Each stage is tested and committed separately.

Stage 1 validation (Python 3.12.0, pinned dependencies): both independent dense
tests passed; `demo.py` reduced negative MLL from 0.83903702 to 0.60669030 in
40 updates, fitted tau=2.85602691, and obtained latent held-out RMSE 0.56974583.
No tests failed. The first run emitted a Torch sparse-invariant warning; graph
construction now explicitly enables invariant checks.

`chebyshev.py` computes 128-node cosine-quadrature coefficients with Torch, then
applies `T_0 V = V`, `T_1 V = (L-I)V`, and
`T_k V = 2(L-I)T_{k-1}V - T_{k-2}V`. Degree is fixed at 20; gradients pass
through the coefficients to heat time. No dense graph covariance is needed.

Stage 2: all five tests passed. Actions and heat derivatives at tau=0.05, 0.8,
2.85602691, and 4 agree with independent SciPy `expm` and its analytic derivative.
A 17-time × 2,001-point spectral check measured maximum absolute polynomial
error 2.887e-15 and minimum value 3.355e-4. These are sampled checks, not continuous
bounds or a general positivity guarantee. No failures; the Torch warning still
appears despite constructor invariant checks, and has not been filtered.

`graph_gp.py` supplies the small `HeatOperator` and `SparseHeatKernel`.
Products scatter an observed RHS onto the graph, run that same recurrence, and
gather the requested vertices. Tensor representation arguments let
`linear_operator` supply autograd; GPyTorch supplies inference, CG, and SLQ.
There is no custom solver. Only the covariance module changes in `GraphGP`.
The demo fits the dense model once, then evaluates both at identical parameters;
it does not claim two independently optimized fits agree.

Stage 3 validation, October 28, 2026: all eight tests passed, including independent
SciPy MLL finite differences, rectangular/repeated IDs, diagonals, RHS and heat
gradients, and held-out mean/variance comparisons. A guarded training/backward/
prediction test observed 1,031 recurrence calls and forbade graph densification,
a full graph basis, and full-graph operator materialization. The sparse inference
path is verified for this tiny unbatched CPU float64 case.

Final demo measurements (absolute differences unless stated otherwise):

| Comparison | MLL or action | Raw heat-time gradient |
| --- | ---: | ---: |
| Dense/sparse covariance action | 1.443e-15 | — |
| Dense/sparse with Cholesky | 0.000e+00 | 2.776e-17 |
| Dense/sparse iterative, common probes | 2.554e-15 | 1.284e-16 |
| Sparse iterative versus dense direct | 6.509e-03 | 3.133e-03 |

Held-out sparse-CG versus dense-direct maximum mean/variance differences are
3.762e-6 / 3.150e-10. Dense latent RMSE is 0.56974583. Fitted heat time and dense
losses match the Stage 1 values to the digits printed above. Iterative evaluation
uses requested CG tolerance 1e-10, no preconditioner, 32 Gaussian probes, seed 71,
and at most 24 Lanczos steps. MLL is GPyTorch's per-observation value.

Failures and caveats: three intermediate integration test runs each failed the
independent MLL comparison. Python-scalar noise initialization and float32
constraint buffers caused reference mismatches. Explicit noise dtype and
matching constraint dtype fixed these; assertion tolerances were unchanged.
The scoped graph invariant check now also covers coalescing, resolving the
initial sparse warning. CG numerical warnings remain visible: the library's
default division safeguard (`eps=1e-10`) can stall before the requested residual.
The demo reported average residuals about 7.0e-6 to 8.0e-6 after 1,000 iterations;
the requested 1e-10 convergence was **not** achieved. The direct/iterative row
includes solver and stochastic trace errors, not polynomial error alone.
Exact diagonals cost one recurrence per requested entry; standard posterior
variance calculation materializes a small test/train cross block. No scaling,
speedup, GPU, parameter-recovery, or global PSD claim is made. No further stage
is implemented.

Actual setup/validation commands (from this directory unless absolute):

```sh
python3 -m venv /Users/josephortega/graph-gp-research/.venv
.venv/bin/python -m pip install torch==2.11.0 gpytorch==1.15.2 linear_operator==0.6.1 numpy==2.4.4 scipy==1.17.1 matplotlib==3.10.8
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python demo.py
git diff --check
```

Manifest: four implementation/demo Python files, one test file, README, AGENTS,
requirements, and gitignore created; nothing deleted. The old repository was kept
read-only. Local `.venv/` and `results/` are ignored; figure inputs include truth,
observations, vertex splits, both posterior moments, fitted heat time, and noise.
Open limitation: strict CG convergence remains unmet as recorded above; larger
graphs, other parameter ranges, batching, and devices remain untested.

## CG settings experiment (October 29, 2026)

Run `.venv/bin/python experiment_cg.py`. It reuses the demo draw, degree 20,
tau=2.85602691 and noise variance 0.04; no refit. The exact/polynomial observed
condition numbers are 14.797948; covariance max error is 1.110e-15. Single/batched
default CG stalls near 7.04e-6 / 7.64e-6 mean residual at requests 1e-6 and 1e-10
(1,001 matvec calls, including initial residual). Requests 1e-2 take 12 calls,
achieving 9.62e-4 / 1.72e-3; 1e-4 takes 16 calls, achieving 3.79e-5 / 8.55e-5.
Batch maximum residual at 1e-4 is 1.55e-4: stopping controls the mean. One explicit
`eps=1e-20` sensitivity check reaches 7.61e-11 / 8.13e-11 in 28/27 calls, supporting
the default division safeguard as the floor's cause, rather than ill-conditioning.

GRF-code settings were verified at [commit 268fff9](https://github.com/MatthewZhang473/Fast-Graph-GP/blob/268fff9a3724cdd480dce4b2946068d8211b4f30/src/grf_gp/utils/config.py).
All changes are scoped, with no preconditioner or private mutations. Dense
negative MLL=0.606690299, d/dtau=+0.022568925, independently checked with matrix-exp
Gaussian algebra. Seeds 71–73: strict 1e-10/32 probes/24 Lanczos has MLL errors
-0.00651 to +0.03614 and gradient errors -0.00600 to +0.00385 (~190 ms, 1,002
recurrence calls); GRF-code 1e-2/64/1 has errors +0.13757 to +0.18322 and -0.00560
to +0.00572 (~4–5 ms, 13 calls). Signs match in all cases; accuracy is not assured
by speed. A 1e-2/64/24 control reduces MLL errors to -0.01311 to +0.03704 (27 calls,
~9 ms). Dense is fastest here (~0.44 ms). Timings are warmed, instrumented forward
+ gradient only, not scaling claims. Numerical CG warnings remain visible; no
assertions failed. Baseline: eight tests passed. Figure/data/log are ignored
`results/cg_residual.png`, `cg_inputs.npz`, `experiment_cg*.txt`. No core files changed.

## Pathwise sampling experiment (October 29, 2026)

Run `.venv/bin/python experiment_pathwise.py`. On an 8x8 grid (22 train, 42 test),
use the same tau/noise, degree-20 `q(L)` at tau/2, data seed 41, sample seed 42.
The sample covariance and update consistently use **K_q=q(L)^2**, not p_tau(L).
Two sparse applications build only 64x22 training columns; one 22x22 Cholesky
factorization handles all sample RHS. Independent dense spectral/polynomial
references verify the action and sampling-transform covariance with unchanged
tight numerical tolerances. Full graph matrices occur only in tiny references
and data generation. K_q/exact-heat max covariance error is 1.082e-15; analytic
posterior mean/variance differences are 1.721e-15 / 4.441e-16.

512/1,024 draws give mean RMSE 0.015257/0.008189 and latent variance RMSE
0.007445/0.005435. At 1,024, mean/variance standardized-error RMS is 0.837/1.121
(max absolute 1.802/3.747), consistent with sampling noise. Counts share a prefix;
this is not independent replication. Block setup/factorization took ~0.52 ms,
and 1,024 draws/actions/conditioning ~3.98 ms in one tiny CPU run. Optional
CG 1e-2 on the same observed block/RHS achieves mean/max residual 1.36e-5/2.30e-5;
sample mean/variance changes from Cholesky are at most 4.27e-6/1.37e-7. Its
unwarmed solve/update took ~20 ms, not a useful speed comparison. No warnings or
assertion failures. The square-root action avoids per-vertex posterior diagonals;
training-column construction, dense observed factorization, sample workspace,
Monte Carlo error, and reliable large-system iterative solves still cost work.
Neither experiment establishes a GRF kernel comparison or resolves historical
gradient-sign failures. A moderate CG budget such as 1e-4 with longer SLQ is a
reasonable next investigation; defaults remain unchanged. Figure/inputs/log are
`results/pathwise_posterior.png`, `pathwise_inputs.npz`, `experiment_pathwise.txt`.
Manifest: two scripts and README added/updated; nothing deleted; core code and old
repository kept unchanged. Stopped after these two experiments.
