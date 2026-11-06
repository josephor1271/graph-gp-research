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

## Learning fidelity (after merged PR #1)

Run `.venv/bin/python experiment_learning.py`: exactly the demo draw, CPU float64,
60 Adam updates at lr=0.08 from 0.8/3.0; sparse CG=1e-4, 64 probes, 24 Lanczos,
no preconditioner, degree 20, fixed noise variance 0.04, and streams 71–73 seeded
once per run. Only heat time is learned; every step is audited with independent
matrix-exponential/Cholesky Gaussian algebra. Eight tests pass before/after;
experiment numerical checks pass with no warnings, while baseline CG warnings
remain visible. Figure, all trajectories/training estimates, endpoint table and
console output are in ignored `results/learning_fidelity.png`, `learning_inputs.npz`,
and `experiment_learning.txt`; exact predictions are used for every endpoint.

The best profile/dense-endpoint reference is NLL=0.604599300 at tau=2.6706553,
not the generating tau=1.4 or a certified optimum. Sparse endpoint gaps are at
most 7.31e-4 per observation, supporting similar learning on this draw; the
profile is locally shallow, so small tau differences need not imply material
model differences. Across seeds, tau spans 2.76043–2.78063 from 0.8 and
2.66539–2.69668 from 3.0, with exact NLL spreads 2.47e-4 and 3.56e-5 respectively.
Final stochastic estimates understate exact NLL by 0.0060–0.0086 and misrank some
fits, but the exact audits show only small endpoint differences here. Sparse
latent RMSE ranges 0.567696–0.568948 versus dense 0.567754/0.568735; exact
predictions isolate learned-parameter effects, and one draw cannot establish
generalization performance. Dense from 0.8 remains unconverged (exact dNLL/dtau
=+0.0110), while dense from 3.0 is nearer stationary (-0.000272), so neither
complete optimizer agreement nor a definitive success is established. Fixed-tau
Cholesky values/gradients agree with the independent reference within 3e-16,
whereas seed-71 iterative raw-gradient errors are 0.0104/0.00388 at 0.8/3.0.
Measured training time excluding audits is ~0.50–0.55 s sparse and ~0.022–0.024 s
dense (unwarmed; 60 updates plus final estimate), with no scalability claim.
Manifest: added one script and updated README; no deletions, core/default changes,
or scaling experiments. Stopped after this fixed experiment.

## Scaling crossover (after merged PR #2)

Apple M4, 16 GB RAM, macOS 26.6.2 arm64, Torch 2.11.0/GPyTorch 1.15.2,
CPU float64, four threads. All arms use AdamW defaults except lr=0.05, 100
updates, tau start=0.8, amplitude one, zero mean, and noise variance 0.04.
Data use degree-20 heat-root actions at tau_true=1.4, seed 12+n, split seed 23+n;
the same latent/noise draw and 64 held-out vertices are used when varying m.
The block baseline rebuilds its observed covariance in batches of at most 32
RHS every update. GRF-code means Chebyshev CG=1e-2/64 probes/1 Lanczos; control
means 1e-4/64/24. Both force iterative inference without preconditioning and
seed the probe stream once at 71. No GRF kernel was run or core code changed.

Measured **100-update training seconds**, excluding audits and prediction:

| n | m | Dense-exact | Block Cholesky | GRF-code | Control | GRF audited NLL gap | GRF latent RMSE ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 400 | 128 | 0.084 | 0.895 | 1.804 | 3.674 | -1.06e-6 | 1.000017 |
| 400 | 256 | 0.161 | 1.849 | 1.913 | — | -1.11e-6 | 1.000037 |
| 1600 | 128 | — | 3.837 | 6.950 | — | +9.34e-6 | 0.999928 |
| 1600 | 512 | — | 16.166 | 7.102 | 14.394 | +6.04e-6 | 1.000031 |
| 3600 | 128 | — | 6.794 | 17.060 | — | +6.48e-5 | 0.999931 |
| 3600 | 512 | — | 27.217 | 16.141 | — | +6.32e-6 | 0.999997 |
| 6400 | 128 | — | 11.104 | 28.824 | — | -2.22e-7 | 0.999999 |
| 6400 | 512 | — | 46.700 | 27.951 | — | +7.53e-6 | 0.999864 |
| 10000 | 128 | — | — | 58.625 | — | unmatched | unmatched |
| 10000 | 512 | — | — | 56.633 | — | unmatched | unmatched |

All eight matched GRF-code runs pass the preset gates: polynomial-Cholesky NLL
gap <=0.02 per observation and latent RMSE ratio <=1.05 against block training.
Qualified measured crossovers occur at (1600,512), (3600,512), (6400,512):
training ratios block/GRF=2.276/1.686/1.671; setup-inclusive ratios=2.276/1.686/1.670.
There is no measured win for m=128 or (400,256); the latter is only 3.5% slower
and therefore marginal. Control also passes both cases; its 1.123x win at
(1600,512) is provisional. Dense-exact is fastest at n=400, including setup;
no full-graph dense crossover was measured. Full dense-exact above 400, control
outside its two prescribed cases, and block training at 10000 were not scheduled.
Both 10000 cases are audited but unmatched sparse-only feasibility, not speedups.

Setup includes common graph/data cost charged to each arm, kernel/optimizer
construction, and one forward/backward warmup without an update. CSV separately
records setup, warmup, training, their setup+training sum, learned tau, endpoint
reported/audited NLL, latent RMSE, and prediction time. Audits use independent
untimed polynomial-block Cholesky; fixed-tau dense/block NLL/raw-gradient errors
at (400,128) were 1.11e-16/8.33e-17. All sparse predictions use existing GPyTorch
mean-only CG at tolerance 1e-4; dense predictions use Cholesky. No variance quality
was checked. GRF reported NLL differs from its audit by up to 0.05843, despite
passing learned-model gates. These are single-stream, single-timing 100-step
endpoints, not convergence guarantees, asymptotic laws, or GRF performance results.

Pilot then full schedule completed: 22 rows / 2,200 updates, zero experiment
warnings, failures, incomplete or pending rows. Existing eight tests pass before
and after; their previous strict-CG warnings remain visible. Bounded sparse
products were checked in warmups (maximum width 32 for block, 65 for iterative).
Generated CSV, figure and console logs stay ignored under `results/`:

```sh
set -o pipefail
caffeinate -i .venv/bin/python -u experiment_scaling.py --pilot 2>&1 | tee results/scaling_pilot.txt
caffeinate -i .venv/bin/python -u experiment_scaling.py 2>&1 | tee results/scaling_run.txt
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

The script resumes recorded arms without retries and saves `scaling_rows.csv`
after each arm; the one figure is `scaling_crossover.png`. A 15-minute per-arm
guard labels incomplete rows and excludes them; it was never reached. Source
settings: [scaling/AdamW](https://github.com/MatthewZhang473/Graph-Random-Features-for-Scalable-Gaussian-Processes/tree/274b7ee393ee516585552f053e9d4a9c6ad2222f/experiments),
[iterative defaults](https://github.com/MatthewZhang473/Fast-Graph-GP/blob/268fff9a3724cdd480dce4b2946068d8211b4f30/src/grf_gp/utils/config.py).
Manifest: added one 214-line script and this finding; no deletions, dependencies,
core/default changes, or follow-on experiments. Review the PR separately.

## Official GRF / PEMS-BAY comparison

`experiment_grf_traffic.py` reuses the pinned official preprocessing, sampler,
training and native evaluator. The bundled cached road graph gives 1,016 nodes,
1,173 edges and 325 sensors; seeds 0/2 use 250 train / 75 test, with training-only
normalization and identical weighted symmetric normalized L in every arm.
Both snapshots have adjacency SHA256
`96026f51a092dac8461388374081628aa1e85349a33214dbcd51cb88ae156a64`.
Sources: [traffic notebook/preprocessing, 274b7ee](https://github.com/MatthewZhang473/Graph-Random-Features-for-Scalable-Gaussian-Processes/tree/274b7ee393ee516585552f053e9d4a9c6ad2222f/experiments/regression/traffic_prediction),
[GRF core, 268fff9](https://github.com/MatthewZhang473/Fast-Graph-GP/tree/268fff9a3724cdd480dce4b2946068d8211b4f30).
No upstream source edits, live road-network substitution, or core GP changes.

One frozen walk budget: 256/node, cutoff 3 (powers 0–2), halt 0.5, four sampler
processes. Native diffusion uses AdamW lr .05 / 1000 updates; exact traffic heat
uses 100 updates. The notebook does not call the scaling-default configuration:
installed defaults train these 250-point native models with Cholesky (threshold
800); pathwise prediction uses CG tolerance 1. The upstream sampler maps seed 0
to effective base seed 42; this behavior was preserved. The downloaded arXiv
appendix instead describes halt 0.1, cutoff 10 and five seeds: this is a pinned
**official-code reproduction**, not verified reproduction of the paper's figures.

Heat models learn beta, signal standard deviation and noise variance, with beta
in (0,2) and raw parameterization/physical starts aligned with traffic exact.
GRF M100 aligns physical starts but retains upstream's unbounded softplus beta;
GRF-vs-heat results are complete-model comparisons, not pure approximation tests.
P denotes original code training; M denotes aligned physical starts. CPU float64,
four Torch threads, Apple M4. Values below are **seed 0 / seed 2**:

| Method | Updates | Training seconds | RMSE (mph) | Common marginal observation NLPD (normalized) |
| --- | ---: | ---: | ---: | ---: |
| GRF P1000 | 1000 | 14.85 / 16.91 | 16.7596 / 15.3858 | 1.41236 / 1.29841 |
| Exact P100 | 100 | 40.86 / 41.98 | 15.7566 / 14.8653 | 1.36857 / 1.26311 |
| Cheb M100, Cholesky | 100 | 3.50 / 3.44 | 15.7566 / 14.8653 | 1.36857 / 1.26311 |
| GRF M100 | 100 | 1.72 / 1.74 | 16.7960 / 15.4736 | 1.41498 / 1.30411 |
| Cheb CG100 | 100 | 14.42 / 13.73 | 15.7614 / 14.8610 | 1.36882 / 1.26396 |
| Cheb M1000, Cholesky | 1000 | 29.82 / 34.15 | 15.7103 / 14.8323 | 1.36665 / 1.26077 |

GRF common metrics average two 512-sample evaluations (seeds 100/101), adding
learned observation noise; source 64-sample latent NLPD is retained separately.
Its original evaluator also computes its reported LML on a training posterior,
not the prior observed-data likelihood. Independent block-Cholesky NLL and
posterior mean/latent variances are reported for every fitted model; GRF's own
Phi posterior is the MC oracle. Common CG tolerance 1e-4 gives identical draws
to native tolerance 1 with the same perturbations here, not a general guarantee.
Cheb CG training uses 1e-4/64 probes/24 Lanczos and Cholesky evaluation; it is
slower than Cholesky at m=250, with audited NLL differing from its estimate by
0.00231/0.00125. Cholesky-trained Chebyshev is not matrix-free inference.

At fixed beta=1.462117 and sigma_f=1.313262, Chebyshev block/action errors are
about 1e-15 and NLL/three raw-gradient errors about 1e-16. Cutoff-3 q2(L)^2 versus
heat has relative observed-block error 0.291/0.293; sampled GRF versus q2(L)^2
has error 0.430/0.394. These are separate Frobenius errors, not additive bounds.
The heat fits improve RMSE/NLPD on these two splits, while matched-budget GRF
is cheaper; no general GRF superiority claim, GeneralGRF baseline, or scalability
claim follows. The exact source recomputes matrix_exp per step; no optimized
eigendecomposition timing baseline was run. Per-arm peak memory was not measured.

All 12 fits / 4,800 updates completed; no long run is pending. Eight existing
tests pass in both environments. An initial NumPy/OSMnx dependency conflict was
resolved using an isolated NumPy 1.26.4 environment; a local scalar-initialization
TypeError was fixed before numerical checks. Upstream Shapely/CSR/deprecation
and GPInput warnings remain visible; no numerical check failed after that fix.
Generated snapshots, states, logs, environment freeze, the one `traffic_rows.csv`
and `grf_traffic.png` stay under ignored `results/traffic/`. The figure plots
method-specific total seconds (walks/setup/train/author+common evaluation),
excluding shared preprocessing and validation; CSV `total_s` also includes the
recorded shared snapshot preparation. Audits are separately timed and excluded.
NLPD in original mph units adds log(train_std); native scores remain separate.
Exact environment/checkout commands and all numerical rows are in the PR.
Manifest: one script plus README; core implementation and project requirements
kept unchanged, no deletions or follow-on experiments. Leave review unmerged.

## GRF accuracy–cost frontier on frozen PEMS-BAY splits

H (J=10, halt=0.1, 4096 walks/node) closes the predictive gap on both splits at
1000 updates, at **69.97 / 75.18 s**, versus Cheb **28.70 / 27.00 s** (seed 0/2):
**2.44× / 2.78× the cost**. It is the only consistently qualifying bounded-beta
setting by the primary predictive oracle at that budget. H's oracle RMSE/NLPD is
15.6266/1.363449 and 14.9092/1.267163; Cheb's is 15.7103/1.366652 and
14.8323/1.260774. H slightly improves seed 0 and worsens seed 2; no significance
claim follows. Its fixed-beta=1.9 kernel error remains 12.46% / 13.14%.

At 100 updates G and H qualify by oracle; the cheapest consistent GRF is G at
8.76 / 9.52 s versus Cheb 3.10 / 2.58 s (2.83× / 3.69×). At 1000, G fails both
thresholds narrowly on seed 2. The cheapest individually qualifying oracle is G
on seed 0 (57.26 s, 1.99× Cheb) and B on seed 2 (15.03 s, 0.56×); neither
consistently qualifies across splits.

**Deployed MC gives a cheaper, sensitive result:** B and C qualify on both splits
at 1000 updates with sample seed 100. B costs 15.24 / 15.03 s (0.53× / 0.56× Cheb).
With replication seed 101 only C and H consistently qualify; C costs 15.90 /
16.02 s (0.55× / 0.59×; costs use seed-100 prediction). B's seed-0 oracle misses
the RMSE limit by 0.0048 mph; C's misses NLPD by 0.00196. These borderline oracle
failures and MC gate flips are material uncertainty. At 100 updates only H
qualifies in both MC replications on both splits. No thresholds/settings changed.

Same PR #4 weighted normalized L, 1016 nodes / 1173 edges / 325 sensors, seeds
0/2, 250 train / 75 test, train-only normalization. Each arm is one AdamW lr=.05
fit (default weight decay .01), checkpointed at 100/1000 with optimizer state
preserved. Primary GRF overrides only beta=2*sigmoid(raw_beta); all other GRF
code is imported unchanged. Cheb uses degree 20 and observed Cholesky. Raw
beta/signal/noise starts 1/1/0 give physical 1.462117157/1.313261688/0.693247181
(last is noise variance) in both. Native keeps softplus beta (initial 1.313261688)
only at G/seed 0, as labeled context outside matched-frontier selection. No
prior fitted GRF state was reused; new Cheb states reproduce PR #4.

Gate: RMSE <= 1.02*Cheb **and** normalized marginal observation NLPD <= Cheb+.02,
separately per split/budget, at full precision. This is descriptive, not an
equivalence test. O/D/R means oracle / deployed seed 100 / replication seed 101.
Oracle means exact own-kernel Cholesky; GRF deploys upstream pathwise CG=1e-4,
512 draws per seed. Cheb deploys its oracle. Training uses native Cholesky
threshold 800 at m=250. Observation variance adds learned noise **once**; no
latent or joint NLPD is substituted. NLPD_mph adds log(train_std), where std is
16.902929904605298 / 17.35662266225777. Test metrics never select a fit/checkpoint.

**1000 updates.** MC entries are RMSE mph / normalized observation NLPD for
seeds 100;101. J includes power zero for GRF. Total = walk + model/optimizer setup
+ cumulative training + one deployed prediction. It excludes data preparation,
dense audits, checkpoint I/O and replication 101. Cached walks are charged at
original measured cost once per standalone checkpoint, including native G.

| Seed | Arm | J | Halt | W | Beta | Oracle RMSE | Oracle NLPD | Deployed MC (100; 101) | Train s | Total s | O/D/R |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0 | Cheb | degree 20 | — | — | 1.995168 | 15.7103 | 1.366652 | Cholesky | 28.38 | 28.70 | baseline |
| 0 | A | 3 | 0.5 | 256 | 0.993908 | 16.8283 | 1.416443 | 16.7665/1.413152; 16.7527/1.411567 | 13.96 | 14.62 | N/N/N |
| 0 | B | 3 | 0.5 | 1024 | 1.366774 | 16.0293 | 1.373958 | 15.9782/1.368456; 15.9511/1.369086 | 14.59 | 15.24 | N/Y/Y |
| 0 | C | 3 | 0.5 | 4096 | 1.498688 | 15.8379 | 1.388607 | 15.7755/1.384951; 15.7668/1.383355 | 13.72 | 15.90 | N/Y/Y |
| 0 | D | 10 | 0.5 | 256 | 0.826383 | 16.7860 | 1.410680 | 16.7156/1.407597; 16.6961/1.405940 | 37.94 | 38.55 | N/N/N |
| 0 | E | 10 | 0.5 | 1024 | 1.242162 | 16.4571 | 1.392973 | 16.4269/1.391674; 16.3800/1.388705 | 40.87 | 41.98 | N/N/N |
| 0 | F | 10 | 0.1 | 256 | 1.553682 | 16.2635 | 1.418792 | 16.2249/1.418161; 16.1959/1.415566 | 52.97 | 54.36 | N/N/N |
| 0 | G | 10 | 0.1 | 1024 | 1.990765 | 15.8286 | 1.376225 | 15.7982/1.375521; 15.7622/1.371908 | 53.89 | 57.26 | Y/Y/Y |
| 0 | H | 10 | 0.1 | 4096 | 1.995508 | 15.6266 | 1.363449 | 15.6033/1.363035; 15.5647/1.359858 | 57.57 | 69.97 | Y/Y/Y |
| 0 | native | 10 | 0.1 | 1024 | 2.130473 | 15.7972 | 1.376975 | 15.7692/1.376224; 15.7330/1.372337 | 60.60 | 64.07 | Y/Y/Y |
| 2 | Cheb | degree 20 | — | — | 1.994967 | 14.8323 | 1.260774 | Cholesky | 26.98 | 27.00 | baseline |
| 2 | A | 3 | 0.5 | 256 | 0.918940 | 15.3373 | 1.294140 | 15.3802/1.295475; 15.3912/1.301325 | 13.15 | 13.46 | N/N/N |
| 2 | B | 3 | 0.5 | 1024 | 1.359263 | 14.9690 | 1.276010 | 15.0127/1.275584; 15.0209/1.282725 | 14.34 | 15.03 | Y/Y/N |
| 2 | C | 3 | 0.5 | 4096 | 1.453515 | 14.9556 | 1.272092 | 14.9926/1.271699; 15.0051/1.278716 | 13.83 | 16.02 | Y/Y/Y |
| 2 | D | 10 | 0.5 | 256 | 0.815142 | 15.5316 | 1.309360 | 15.5645/1.311571; 15.5764/1.315840 | 37.49 | 38.12 | N/N/N |
| 2 | E | 10 | 0.5 | 1024 | 1.620250 | 15.1388 | 1.290821 | 15.1804/1.292668; 15.1791/1.294825 | 43.26 | 44.40 | N/N/N |
| 2 | F | 10 | 0.1 | 256 | 1.309213 | 15.4015 | 1.310455 | 15.4252/1.312623; 15.4451/1.314351 | 52.95 | 54.39 | N/N/N |
| 2 | G | 10 | 0.1 | 1024 | 1.777601 | 15.1501 | 1.281844 | 15.1795/1.284334; 15.2025/1.285426 | 56.89 | 60.45 | N/N/N |
| 2 | H | 10 | 0.1 | 4096 | 1.995407 | 14.9092 | 1.267163 | 14.9493/1.269090; 14.9684/1.271352 | 62.19 | 75.18 | Y/Y/Y |

**100 updates of the same fits.** Additional training helps some held-out scores
and worsens others; compare both fixed checkpoints without selecting one.

| Seed | Arm | Beta | Oracle RMSE/NLPD | Deployed MC (100; 101) | Train s | Total s | O/D/R |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| 0 | Cheb | 1.938369 | 15.7566/1.368571 | Cholesky | 2.77 | 3.10 | baseline |
| 0 | A | 1.062073 | 16.8652/1.419324 | 16.8072/1.416191; 16.7954/1.414526 | 1.38 | 2.05 | N/N/N |
| 0 | B | 1.413039 | 16.2570/1.387853 | 16.2156/1.383799; 16.1859/1.382508 | 1.48 | 2.13 | N/N/N |
| 0 | C | 1.527222 | 16.0412/1.397486 | 15.9863/1.394300; 15.9725/1.391866 | 1.39 | 3.56 | N/N/N |
| 0 | D | 0.873889 | 16.7878/1.411574 | 16.7207/1.408656; 16.7017/1.406786 | 3.78 | 4.39 | N/N/N |
| 0 | E | 1.255154 | 16.4578/1.393124 | 16.4286/1.391884; 16.3816/1.388827 | 4.12 | 5.23 | N/N/N |
| 0 | F | 1.563778 | 16.2699/1.418736 | 16.2317/1.418099; 16.2028/1.415489 | 5.27 | 6.66 | N/N/N |
| 0 | G | 1.922472 | 15.8579/1.376560 | 15.8263/1.375853; 15.7906/1.372387 | 5.35 | 8.76 | Y/Y/Y |
| 0 | H | 1.944227 | 15.6670/1.364710 | 15.6421/1.364245; 15.6037/1.361175 | 5.71 | 18.10 | Y/Y/Y |
| 0 | native | 2.126108 | 15.8000/1.376925 | 15.7720/1.376174; 15.7358/1.372300 | 5.39 | 8.81 | Y/Y/Y |
| 2 | Cheb | 1.935924 | 14.8653/1.263113 | Cholesky | 2.56 | 2.58 | baseline |
| 2 | A | 0.996838 | 15.4277/1.300039 | 15.4687/1.302068; 15.4813/1.306171 | 1.32 | 1.61 | N/N/N |
| 2 | B | 1.370568 | 15.1228/1.286055 | 15.1650/1.286449; 15.1721/1.292130 | 1.42 | 2.12 | N/N/N |
| 2 | C | 1.454878 | 15.1270/1.284692 | 15.1629/1.285159; 15.1764/1.290586 | 1.36 | 3.56 | N/N/N |
| 2 | D | 0.913368 | 15.5415/1.309971 | 15.5722/1.312260; 15.5864/1.315658 | 3.65 | 4.25 | N/N/N |
| 2 | E | 1.618794 | 15.1386/1.290909 | 15.1803/1.292752; 15.1790/1.294917 | 4.35 | 5.50 | N/N/N |
| 2 | F | 1.311040 | 15.4034/1.310608 | 15.4271/1.312775; 15.4469/1.314490 | 4.97 | 6.39 | N/N/N |
| 2 | G | 1.763181 | 15.1562/1.282373 | 15.1856/1.284841; 15.2086/1.285977 | 5.86 | 9.52 | Y/N/N |
| 2 | H | 1.940075 | 14.9339/1.268815 | 14.9732/1.270720; 14.9926/1.273019 | 6.79 | 19.82 | Y/Y/Y |

**Fixed-beta errors, signal variance 1.** Same 325-sensor block (permuted between
splits). T=||Ktrunc-Kheat||F/||Kheat||F is common to both splits;
S=||Kgrf-Ktrunc||F/||Ktrunc||F; E=||Kgrf-Kheat||F/||Kheat||F. Paired entries are
seed 0 / seed 2; diagonal diagnostics are maximum absolute differences.
These errors are not additive. At finite W, E[Phi Phi^T] need not equal
(E Phi)(E Phi)^T; no unbiased-covariance claim is made.

| Arm | Beta | T (both splits) | S (0/2) | E (0/2) | Max diag vs heat (0/2) | Max diag vs trunc (0/2) | Max abs Kgrf (0/2) |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| A | 1.0 | 0.096611 | 0.234694 / 0.21841 | 0.280286 / 0.26334 | 0.534161 / 0.439279 | 0.479971 / 0.383445 | 1.08314 / 0.997663 |
| A | 1.9 | 0.770209 | 0.624322 / 0.548993 | 1.39175 / 1.28492 | 2.40965 / 1.75277 | 2.04113 / 1.37063 | 2.88509 / 2.24492 |
| B | 1.0 | 0.096611 | 0.115055 / 0.106105 | 0.160209 / 0.154484 | 0.21593 / 0.185739 | 0.176263 / 0.141666 | 0.774821 / 0.693742 |
| B | 1.9 | 0.770209 | 0.268998 / 0.254891 | 0.921846 / 0.915004 | 0.970567 / 0.866795 | 0.670901 / 0.564442 | 1.36629 / 1.26339 |
| C | 1.0 | 0.096611 | 0.0585548 / 0.0575435 | 0.116952 / 0.116206 | 0.139797 / 0.159574 | 0.08778 / 0.104053 | 0.702249 / 0.716017 |
| C | 1.9 | 0.770209 | 0.145624 / 0.135184 | 0.822671 / 0.812693 | 0.660692 / 0.647995 | 0.282017 / 0.269783 | 1.16087 / 1.13617 |
| D | 1.0 | 1.19006e-07 | 0.265023 / 0.224952 | 0.265023 / 0.224952 | 0.571488 / 0.350905 | 0.571488 / 0.350905 | 1.12046 / 0.881443 |
| D | 1.9 | 3.52405e-05 | 1.31004 / 1.10654 | 1.31001 / 1.10651 | 2.86212 / 2.3625 | 2.86214 / 2.36252 | 3.33757 / 2.84524 |
| E | 1.0 | 1.19006e-07 | 0.130063 / 0.121411 | 0.130063 / 0.121411 | 0.194621 / 0.17443 | 0.194621 / 0.17443 | 0.705987 / 0.706203 |
| E | 1.9 | 3.52405e-05 | 0.549683 / 0.523276 | 0.549669 / 0.523261 | 0.924822 / 0.933173 | 0.92483 / 0.93318 | 1.3728 / 1.35588 |
| F | 1.0 | 1.19006e-07 | 0.151839 / 0.172523 | 0.151838 / 0.172523 | 0.296446 / 0.377706 | 0.296446 / 0.377706 | 0.833212 / 0.921811 |
| F | 1.9 | 3.52405e-05 | 0.505419 / 0.599708 | 0.505405 / 0.599693 | 0.716624 / 1.49927 | 0.716643 / 1.49929 | 1.20039 / 1.96177 |
| G | 1.0 | 1.19006e-07 | 0.0732668 / 0.0811104 | 0.0732668 / 0.0811104 | 0.122264 / 0.147806 | 0.122264 / 0.147806 | 0.65846 / 0.695444 |
| G | 1.9 | 3.52405e-05 | 0.250789 / 0.26733 | 0.250784 / 0.267323 | 0.390032 / 0.413883 | 0.390046 / 0.4139 | 0.81704 / 0.876384 |
| H | 1.0 | 1.19006e-07 | 0.0395144 / 0.0407103 | 0.0395144 / 0.0407103 | 0.0653002 / 0.0597879 | 0.0653001 / 0.0597878 | 0.599045 / 0.609401 |
| H | 1.9 | 3.52405e-05 | 0.124553 / 0.131433 | 0.124551 / 0.131429 | 0.242752 / 0.205508 | 0.242733 / 0.205523 | 0.629863 / 0.655588 |

More walks reduce realized J=3 sampling error, but leave the deterministic
truncation floor: T=0.770209 at beta=1.9, versus 0.00003524 for J=10. For seed 0,
A→C reduces S=.62432→.14562 while E remains .82267. Increasing J alone B→E
reduces truncation but increases S=.26900→.54968; 1000-step predictive quality
worsens on both splits. At fixed J,W, E→G reduces S=.54968→.25079 while sampling
cost rises .633→2.651 s; changing halt leaves the deterministic target unchanged.
F→G→H further reduces S with W on both seeds. These are realized comparisons,
not a sampling-distribution estimate or monotonicity theorem.

Kernel fidelity, fitted hyperparameters and posterior MC error are separated,
but do not identify a single cause of held-out improvement. J=3 partly
compensates using other hyperparameters; H retains kernel error while improving
seed 0. Native G reaches beta=2.130473, oracle 15.7972/1.376975 in 64.07 s;
its different parameter domain does not establish a matched-parameter win.

<details>
<summary>Complete checkpoint audits and runtime components (CSV)</summary>

Columns omitted from primary tables: sigma_f is std, noise is variance; latent
and observation variances/moment errors use normalized units. MC errors are
moment RMSEs against the own-kernel oracle. Oracle time is an excluded GRF audit
and the included Cheb prediction; MC101 time is excluded. Six significant digits
are shown; local CSV preserves full precision, statuses and all gate flags.

```csv
seed,arm,updates,sigma_f,noise,kernel_nll,latent_var_min,latent_var_mean,latent_var_max,observation_var_mean,walk_s,setup_s,pred_s,mc101_s,oracle_s,mc100_mean_error,mc100_variance_error,mc101_mean_error,mc101_variance_error,oracle_nlpd_mph,deployed_nlpd_mph
0,Cheb,100,1.38218,0.317172,1.37676,0.284749,0.549727,0.957627,0.866898,0,0.308752,0.018513,,0.018513,,,,,4.19606,4.19606
0,Cheb,1000,1.39088,0.31744,1.37594,0.275101,0.545954,0.966939,0.863394,0,0.308752,0.0182886,,0.0182886,,,,,4.19414,4.19414
0,A,100,1.04698,0.404425,1.40436,0.324065,0.58506,0.987578,0.989485,0.174263,0.333456,0.165341,0.138081,0.0142698,0.038523,0.041366,0.0322491,0.0312034,4.24681,4.24368
0,A,1000,1.09963,0.339783,1.40424,0.384338,0.648773,1.05351,0.988556,0.174263,0.333456,0.151799,0.126572,0.0137719,0.0405349,0.0454914,0.0341395,0.0345015,4.24393,4.24064
0,B,100,1.22761,0.192157,1.3883,0.413631,0.72218,1.18352,0.914337,0.503008,0.000296375,0.145714,0.148059,0.0163057,0.0422779,0.0510696,0.0357505,0.0386951,4.21534,4.21129
0,B,1000,1.3625,0.0237153,1.38575,0.51615,0.872325,1.4183,0.896041,0.503008,0.000296375,0.14184,0.13796,0.0135833,0.0462256,0.0614392,0.0395101,0.0472435,4.20144,4.19594
0,C,100,1.2524,0.131552,1.37609,0.400205,0.770242,1.22444,0.901794,2.03422,0.000303166,0.143994,0.144109,0.0154455,0.0436134,0.0532504,0.0368706,0.0437667,4.22497,4.22179
0,C,1000,1.34298,0.00950074,1.37191,0.463023,0.870962,1.38224,0.880463,2.03422,0.000303166,0.144395,0.160008,0.0151765,0.0461122,0.0600097,0.0392999,0.0494769,4.21609,4.21244
0,D,100,1.10479,0.358274,1.40484,0.32012,0.618585,0.951076,0.976859,0.1872,0.000351959,0.430896,0.45991,0.0423108,0.039229,0.0426916,0.0337384,0.0319383,4.23906,4.23614
0,D,1000,1.13887,0.3032,1.40477,0.355913,0.672815,1.00499,0.976015,0.1872,0.000351959,0.424691,0.472837,0.0446506,0.0409638,0.0462843,0.0353577,0.0348271,4.23817,4.23508
0,E,100,1.20347,0.344984,1.39554,0.221072,0.550636,0.956546,0.89562,0.633126,0.000343292,0.477903,0.495094,0.0471855,0.0359063,0.0392755,0.0309882,0.0268725,4.22061,4.21937
0,E,1000,1.21121,0.334805,1.39554,0.224968,0.56065,0.966137,0.895454,0.633126,0.000343292,0.483106,0.509474,0.0531129,0.0362423,0.0399072,0.0313063,0.027388,4.22046,4.21916
0,F,100,1.26427,0.342876,1.38444,0.162013,0.560545,1.03504,0.903421,0.765099,0.000302083,0.620631,0.626466,0.0608687,0.0337041,0.0399508,0.0303675,0.0279118,4.24622,4.24559
0,F,1000,1.26869,0.335635,1.38444,0.164111,0.565505,1.04197,0.901141,0.765099,0.000302083,0.623858,0.624251,0.0582525,0.0338808,0.0402755,0.0305404,0.0281678,4.24628,4.24565
0,G,100,1.38779,0.306283,1.37688,0.15143,0.549425,1.15964,0.855708,2.65132,0.000255,0.756707,0.746306,0.0642629,0.0322892,0.0391328,0.0299952,0.0254191,4.20405,4.20334
0,G,1000,1.39557,0.309451,1.37636,0.137463,0.54396,1.20466,0.853411,2.65132,0.000255,0.715533,0.732557,0.0624034,0.0318405,0.0390482,0.0297235,0.0250432,4.20371,4.20301
0,H,100,1.4292,0.275575,1.3723,0.187563,0.582601,1.1174,0.858177,11.6272,0.000313333,0.761559,0.764571,0.0692412,0.034679,0.0423272,0.0313204,0.0288441,4.1922,4.19173
0,H,1000,1.43755,0.276032,1.37142,0.176795,0.57965,1.12625,0.855683,11.6272,0.000313333,0.775165,0.758106,0.0706081,0.0344811,0.0422877,0.031183,0.0287799,4.19094,4.19052
0,native,100,1.40496,0.322243,1.3759,0.113121,0.532336,1.30215,0.85458,2.65132,0.000261584,0.773011,0.797701,0.0672038,0.0309211,0.0389322,0.0291612,0.0242552,4.20441,4.20366
0,native,1000,1.40571,0.321502,1.37589,0.112117,0.532212,1.30659,0.853714,2.65132,0.000261584,0.81106,0.858029,0.0696102,0.0308961,0.0389549,0.0291503,0.0242446,4.20446,4.20371
2,Cheb,100,1.33902,0.363079,1.37613,0.267451,0.579802,0.890387,0.942882,0,0.00102004,0.0201155,,0.0201155,,,,,4.11709,4.11709
2,Cheb,1000,1.34734,0.363921,1.37531,0.256971,0.577293,0.898464,0.941214,0,0.00102004,0.0196354,,0.0196354,,,,,4.11475,4.11475
2,A,100,1.20103,0.219032,1.39196,0.483424,0.781531,1.29983,1.00056,0.15671,0.000401709,0.134894,0.131556,0.0161917,0.0410541,0.047706,0.0343769,0.0500786,4.15401,4.15604
2,A,1000,1.33718,0.0244415,1.3895,0.600401,0.978225,1.56696,1.00267,0.15671,0.000401709,0.153833,0.13318,0.0139133,0.045959,0.0602695,0.0384991,0.0624077,4.14811,4.14945
2,B,100,1.26001,0.162281,1.38139,0.393307,0.809162,1.23506,0.971443,0.558285,0.000352792,0.139466,0.13653,0.0142423,0.0412725,0.0498883,0.0349734,0.0527602,4.14003,4.14042
2,B,1000,1.37403,0.0169241,1.37841,0.46971,0.955446,1.4605,0.97237,0.558285,0.000352792,0.133675,0.136056,0.0155969,0.0447372,0.0597712,0.0380065,0.062067,4.12998,4.12956
2,C,100,1.25265,0.166307,1.38367,0.434163,0.784816,1.19446,0.951122,2.06607,0.000300667,0.133862,0.132161,0.014687,0.0407818,0.0475275,0.035302,0.049207,4.13867,4.13913
2,C,1000,1.36712,0.0170061,1.3807,0.512969,0.929087,1.42183,0.946093,2.06607,0.000300667,0.126204,0.13122,0.0149415,0.0443186,0.0570056,0.0383914,0.0579467,4.12607,4.12567
2,D,100,1.20962,0.256206,1.39497,0.4449,0.741422,1.13017,0.997629,0.167269,0.000317542,0.430516,0.413847,0.0421889,0.0393027,0.0447949,0.0330823,0.0461947,4.16394,4.16623
2,D,1000,1.27206,0.133952,1.39463,0.546324,0.859464,1.23361,0.993416,0.167269,0.000317542,0.463824,0.467294,0.0471099,0.0427647,0.0517169,0.0357504,0.053824,4.16333,4.16555
2,E,100,1.23943,0.386684,1.38151,0.157576,0.579612,1.59794,0.966297,0.648122,0.000326709,0.503456,0.628786,0.0502251,0.0324666,0.0340314,0.0280143,0.0363647,4.14488,4.14673
2,E,1000,1.2381,0.386838,1.38151,0.156699,0.578231,1.59651,0.965069,0.648122,0.000326709,0.496673,0.475589,0.0491038,0.0324238,0.033957,0.0279794,0.0362818,4.1448,4.14664
2,F,100,1.18937,0.375251,1.39459,0.219851,0.579254,1.12257,0.954505,0.821871,0.0003205,0.596072,0.592455,0.0620145,0.0339061,0.0327521,0.0284269,0.0368767,4.16458,4.16675
2,F,1000,1.19174,0.372759,1.39459,0.220729,0.581857,1.12682,0.954616,0.821871,0.0003205,0.61721,0.713567,0.063304,0.0339865,0.0328975,0.0284918,0.03704,4.16443,4.1666
2,G,100,1.26748,0.40375,1.39105,0.236896,0.559727,1.04941,0.963477,2.86966,0.000323416,0.788621,0.688719,0.0645037,0.0325577,0.0329961,0.0284264,0.0331855,4.13635,4.13882
2,G,1000,1.26951,0.403261,1.39105,0.233154,0.559268,1.05927,0.962529,2.86966,0.000323416,0.691162,0.675848,0.0655434,0.0325042,0.0330368,0.0284092,0.0331601,4.13582,4.13831
2,H,100,1.33078,0.365927,1.37717,0.284109,0.572929,1.16643,0.938855,12.3005,0.000358875,0.726258,0.770414,0.0677753,0.032384,0.0330121,0.0286499,0.0348787,4.12279,4.12469
2,H,1000,1.3391,0.365792,1.37632,0.275764,0.571587,1.19957,0.937379,12.3005,0.000358875,0.691999,0.71387,0.0688924,0.0322076,0.0331198,0.0285737,0.0349309,4.12114,4.12306
```
</details>

All 19 fits / 19,000 updates / 38 checkpoints and 32 approximation rows completed;
no failed or pending scientific case. J=3 polynomial/power indexing and initial
sampled discrepancy reproduce PR #4; A walks are bit-identical for both seeds.
Cheb/matrix_exp block relative errors are 7.55e-16 / 1.18e-15 at beta 1.0/1.9;
small/real-graph NLL, action and raw beta/signal/noise gradients agree around
1e-15 or better. Independent GPyTorch Cholesky conditioning checks all 38 states:
max NLL/NLPD/mean-variance error 2.22e-16, RMSE 1.78e-15. Optimizer counts/loss
prefixes, timing sums and all unrounded gates pass. A resume skips all 19 fits,
with unchanged fit rows and checkpoint/walk mtimes. Eight tests pass before
changes and afterward in both environments; known strict-CG warnings remain.

Post-run CSV string-key duplication and a replication-score figure lookup error
were fixed before reporting, without repeating a fit/sample; canonical CSV has
70 unique complete rows. Sparse CSR/invariant and upstream deprecation warnings
remain in logs. No experiment convergence, covariance or memory failure occurred.
Peak memory was not measured; single sequential timings have no replication
error bars. Only two joint data/walk seeds, no independent walk replicates,
GeneralGRF, temporal models, GPU work or scaling were tested.

Apple M4, 16 GiB, 10 logical CPUs, macOS 26.6.2; CPU float64, four Torch threads,
four sampler workers, effective base seeds 42/2 (0→42 upstream). Python 3.12.0;
Torch 2.11.0, GPyTorch 1.15.2, linear_operator .6.1, NumPy 1.26.4, SciPy 1.17.1,
Matplotlib 3.10.8, OSMnx 1.9.4, Shapely 2.0.7, pandas 2.2.3, NetworkX 3.3,
GeoPandas .14.4, pyproj 3.8.0, tables 3.11.1. Existing environment reused.

Pinned [traffic code 274b7ee](https://github.com/MatthewZhang473/Graph-Random-Features-for-Scalable-Gaussian-Processes/tree/274b7ee393ee516585552f053e9d4a9c6ad2222f)
and [GRF core 268fff9](https://github.com/MatthewZhang473/Fast-Graph-GP/tree/268fff9a3724cdd480dce4b2946068d8211b4f30)
are unchanged. Core declares MIT; local changes are the tiny beta adapter and
attributed checkpointable equivalent of `experiments.utils.train_model`.
The notebook uses J=3/halt=.5. The work order identifies J=10/halt=.1 in final
[paper Appendix C.5](https://openreview.net/pdf?id=89SQfLguNn), with C.4 being CG.
OpenReview access was blocked (verification/403). The accessible
[preprint](https://arxiv.org/abs/2509.03691) confirms those traffic settings under
older numbering C.4 and five seeds. Agreement with the paper's plotted figure
remains **unverified**; this is a controlled two-split subset.

Adjacency SHA256 `96026f51a092dac8461388374081628aa1e85349a33214dbcd51cb88ae156a64`
and L SHA256 `64661f4a768224ebcf799bfee713dce1402b36febec7d6236b94625c3548671e`
are unchanged. Separate split/target fingerprints are checked; mismatch stops
execution. No substitute roads or silent snapshot regeneration.

From the existing pinned PR #4 setup ([fresh installation commands](https://github.com/josephor1271/graph-gp-research/pull/4)),
explicitly prepare snapshots only if absent, then pilot, full/resume, tests and
figure generation:

```sh
mkdir -p results/traffic
results/traffic_env/bin/python - <<'PYDATA'
from experiment_grf_traffic import load_data
for seed in [0, 2]:
    load_data(seed)
PYDATA
caffeinate -i results/traffic_env/bin/python -u experiment_grf_frontier.py --pilot > results/traffic/frontier_pilot.log 2>&1
caffeinate -i results/traffic_env/bin/python -u experiment_grf_frontier.py > results/traffic/frontier_full.log 2>&1
results/traffic_env/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
results/traffic_env/bin/python experiment_grf_frontier.py --figure-only
git diff --check
```

The full command resumes without repeating finished fits; preserve earlier logs
when rerunning. One two-panel `results/traffic/grf_frontier.png` plots oracle
NLPD/RMSE against total cost; colors mark Cheb/J3/J10/native, markers mark splits,
hollow/filled marks 100/1000. No lines join splits. As specified, plots/data stay
ignored and local; numerical tables appear here and in the PR. Walk parameters,
raw tensors, both optimizer/model checkpoints, losses, environment freeze and
pilot/full/resume/audit/test logs remain under `results/traffic/`.

Manifest: added one script and this README evidence; deleted nothing. Core GP,
historical experiments, requirements, upstream sources and old toolbox unchanged.
No pending runs. Limitations: two joint seeds, MC-sensitive gates, unverified paper
figure, no peak-memory measurement. Leave the PR open and stop this experiment.

## Large synthetic graphs: actual sparse iterative GRF versus Chebyshev

All 24 matched 100-update runs completed through n=50,000 (12,500 train/test each).
GRF C is 1.29–2.09× faster in measured model totals; its noisy-test RMSE is
0.47–0.96% higher. No timing crossover, failed/incomplete run or resource limit hit.
At 50,000: Cheb 780.65 s / RMSE .604190; GRF C 373.37 s / RMSE .609002.
Means below use three timing repeats through 5,000 nodes, one thereafter; graph,
data, walks and probes are fixed across repeats, with bit-identical fitted scores.

| n | Train/test each | Method | Timing reps | Setup s | Train s | Mean prediction s | Model total s | Noisy-test RMSE |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 250 | Cheb | 3 | 0.355 | 12.301 | 0.0109 | 12.667 | 0.589892 |
| 1000 | 250 | GRF_C | 3 | 2.364 | 5.060 | 0.0099 | 7.435 | 0.595208 |
| 2500 | 625 | Cheb | 3 | 0.350 | 20.847 | 0.0263 | 21.223 | 0.603283 |
| 2500 | 625 | GRF_C | 3 | 5.651 | 10.823 | 0.0196 | 16.493 | 0.606118 |
| 5000 | 1250 | Cheb | 3 | 0.354 | 56.816 | 0.0590 | 57.229 | 0.577812 |
| 5000 | 1250 | GRF_C | 3 | 12.043 | 23.807 | 0.0407 | 35.891 | 0.582342 |
| 10000 | 2500 | Cheb | 1 | 0.355 | 136.970 | 0.1099 | 137.435 | 0.594429 |
| 10000 | 2500 | GRF_C | 1 | 28.512 | 55.459 | 0.0717 | 84.043 | 0.598794 |
| 25000 | 6250 | Cheb | 1 | 0.494 | 394.773 | 0.2730 | 395.540 | 0.595196 |
| 25000 | 6250 | GRF_C | 1 | 74.427 | 138.426 | 0.1968 | 213.050 | 0.600931 |
| 50000 | 12500 | Cheb | 1 | 0.479 | 779.663 | 0.5058 | 780.648 | 0.604190 |
| 50000 | 12500 | GRF_C | 1 | 128.179 | 244.901 | 0.2911 | 373.371 | 0.609002 |

Weighted jittered local grids use L=I-D^-1/2 W D^-1/2, bounded degree <=6.
Independent SciPy expm_multiply generates beta_true=1.5 heat draws plus noise
variance .1; 25% train/25% test, train-only normalization. Both learn bounded beta,
signal std and noise, AdamW .05/100 steps, CPU float64/four threads, CG=1e-4,
64 probes/24 Lanczos/no preconditioner/no Cholesky. GRF C is J3/.5/4096 walks,
four sampler workers; source pins/adapters unchanged. Common prediction is mean
only via ordinary ExactGP CG; NLPD/variance unavailable. Totals exclude graph/data,
audit and I/O costs, but include GRF sampling/lazy feature work where it occurs.
Training dominates. Empirical total exponents are 1.11/1.03 over these six sizes;
these are finite-range fits, not O(n) training proofs. Heat is correctly specified
by this generator; no universal real-data accuracy or large-graph variance claim.
Parent high-water RSS at metric completion is 1249.8/1068.8 MiB at n=50,000,
including process/data overhead; worker-inclusive sampled sums are separate.
Small dense references, guarded sparse paths and data/split checks pass; largest
Cheb mean residual 4.47e-6. Eight tests pass in both environments. Raw warnings,
24 rows, status/manifest, graphs/targets/checkpoints and two inspected figures
(runtime.png, rmse.png) stay in ignored results/large_synthetic; full tables in PR.

```sh
mkdir -p results/large_synthetic
caffeinate -i results/traffic_env/bin/python -u experiment_grf_large_synthetic.py --smoke > results/large_synthetic/smoke.log 2>&1
caffeinate -i results/traffic_env/bin/python -u experiment_grf_large_synthetic.py > results/large_synthetic/sweep.log 2>&1
results/traffic_env/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
results/traffic_env/bin/python experiment_grf_large_synthetic.py --figure-only
git diff --check
```
Recorded attempts are skipped on resume; preserve original logs with a new filename.
Manifest: one script plus this README; no deletions, core/dependency/upstream edits,
PR merges or pending runs. Earlier experiments and old toolbox remain unchanged.

## Independently tuned numerical and predictive frontiers

`experiment_speed_accuracy.py` follows the November 6 optimization work order,
using new development and fresh final graph/data/split seeds. Historical results
above remain historical. The generator accepts optional seeds/output directories;
its original defaults reproduce the recorded tiny-graph fingerprint. Core GP,
recurrence, upstream sources and numerical packages are unchanged. Degree is a
scoped experiment override of the existing coefficient API.

All six degrees 6/8/10/12/16/20 have positive polynomials on the audited spectral
and beta grids, including beta=2; this is not a formal PSD certificate. Degree 6
has maximum scalar/physical-beta derivative errors 6.92e-5/1.86e-4; degree 8
9.14e-7/3.34e-6. Ideal Bessel tails and actual quadrature roundoff are separate.
Own-kernel Cholesky audits use n=1000, two fixed physical parameter states and
three trace seeds. The exploratory screen requires absolute NLL/observation
error <=.02, physical-beta gradient error <=.01, true residual <=2e-4 and no
material gradient sign reversal. Cheb 8/8 and 16/16 fail; original GRF C 8/8
fails. Full finite-feature audits reject additional GRF configurations. Rejected
settings remain on the development frontier; thresholds were not relaxed.

The 1% development RMSE screen is exploratory, relative to each method's original
strict reference. GRF candidates are confirmed with two walk seeds; an original-C
inference-only control distinguishes solver savings from changing covariance.
Every fit uses AdamW .05, CPU float64/four threads, beta in (0,2), learnable signal
and noise, CG=1e-4, no preconditioning/Cholesky, and 100 actual updates. Model
totals include fresh walks/setup, training and mean-only prediction; common
graph/data generation, off-timer dense audits, checkpoint I/O and imports are
separate. NLPD and predictive variance are unavailable.

```sh
results/traffic_env/bin/python experiment_speed_accuracy.py --numerics
results/traffic_env/bin/python experiment_speed_accuracy.py --smoke
caffeinate -i results/traffic_env/bin/python -u experiment_speed_accuracy.py --degree-screen --solver-screen
caffeinate -i results/traffic_env/bin/python -u experiment_speed_accuracy.py --features
caffeinate -i results/traffic_env/bin/python -u experiment_speed_accuracy.py --select-confirm
results/traffic_env/bin/python experiment_speed_accuracy.py --freeze
caffeinate -i results/traffic_env/bin/python -u experiment_speed_accuracy.py --extended
caffeinate -i results/traffic_env/bin/python -u experiment_speed_accuracy.py --final
results/traffic_env/bin/python experiment_speed_accuracy.py --figures
results/traffic_env/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

Freeze once before final outcomes. Resume with `--final`, which skips completed
fits and preserves attempt logs; failed/incomplete cases remain retryable. Each
attempt has a fixed 1800-second/12-GiB sampled summed-RSS watchdog. Seconds use
`perf_counter`, which excludes macOS system sleep; UTC clock gaps are also saved
for final runs. The development J10/walk-202 confirmation crossed a laptop sleep
and power-source change: its 448.34 s timing is retained and flagged, alongside
270.70 s for walk 101, without attributing that difference solely to walk seeds.
Memory samples
can miss peaks and double-count shared pages; small-case parent high-water RSS
includes off-timer dense audits. Operator counters are non-equivalent work units.
Raw settings, checkpoints, warnings, seed fingerprints, four reproducible figures
and tables stay under ignored `results/speed_accuracy/`; full numerical tables
and the frozen manifest belong to the PR.
