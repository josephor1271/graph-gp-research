# Appendix C.3 mesh importance-sampling ablation

Question: does the authors' importance-corrected learned GRF outperform their
ablated learned GRF on one frozen 900-node diffusion field? This is a
paper-specified configuration with documented notebook defaults and numerical
adaptations, **not an exact Table 5 / Figure 5 reproduction**.

## Source reconciliation (frozen before the pilot)

The supplied `grf.pdf` is the 23-page preprint; SHA256
`8e453910edf00c8b16ed9b10885407951eee5ad41aeb7f1d53957e92f37771ee`.
Sources: [paper](https://arxiv.org/abs/2509.03691),
[pinned notebook](https://github.com/MatthewZhang473/Graph-Random-Features-for-Scalable-Gaussian-Processes/blob/274b7ee393ee516585552f053e9d4a9c6ad2222f/experiments/ablation/2d_mesh.ipynb),
[pinned core](https://github.com/MatthewZhang473/Fast-Graph-GP/tree/268fff9a3724cdd480dce4b2946068d8211b4f30).
Notebook cells below are **zero-indexed**. No notebook-protocol experiment is run.

|Setting|Paper source / value|Pinned notebook / code|Primary choice|
|---|---|---|---|
|Graph|C.3 p16, 30×30|cells 2–4, unweighted grid|900 nodes; upstream symmetric normalized Laplacian|
|Generating beta|C.3 p16, 10|cell 2, 40|10|
|Observed fraction|C.3 p16, 10%|cell 2, 20%|90 / 810 distinct train / test|
|Walk budget|C.3 p16, 10,000/node|cell 2, same|10,000 for each sampler|
|Cutoff|C.3 p16, walks exceeding 10 steps truncated|cell 2, 8|API argument 10: recorded powers 0…9, at most nine contributing hops; a final unrecorded hop consumes RNG|
|Observation noise|C.3 / §4.1 do **not** specify its magnitude in supplied PDF|cells 2–4, SD .1 (variance .01)|variance .1 as explicitly required by work order; do not attribute this value to C.3|
|Signal variance / seed / halt|Not established by C.3|cell 2: 10 / 1 / .01|These notebook defaults, fixed before scores|
|Generation jitter|Not specified|cell 3 adds 1e-6 I|None: exact heat covariance, Cholesky succeeds|
|Normalization / split|Not fully specified|cell 4: train-only mean, sample SD; NumPy choice|Same procedure, seed 1; no test-label normalization|
|Learning|C.3 p16: Adam .01, 1000 updates|cell 2: .05, 500; `experiments/utils.py:18` actually AdamW, default decay .01|Explicit Adam .01, decay 0, 1000 updates|
|Initialization|Not specified|core diffusion raw beta/signal 1/1, noise 0; GeneralGRF randn|Official defaults; reset Torch seed 1 for each model so both GRFs share coefficients|
|Inference|§3 describes CG / stochastic gradients; C.3 budgets missing|Notebook does not call `set_gp_defaults`; active installed defaults: Cholesky 800, CG 1, Lanczos 20, 10 probes|Explicit Cholesky threshold 1000 and exact log probability; both thresholds use Cholesky at 90 observations|
|Scoring|Table 5 p17; marginal / joint convention unspecified|`utils.py:59–65`: exact joint noisy NLPD; GRF diagonal latent NLPD from 64 samples|Common exact own-kernel posterior, marginal observation NLPD; add learned noise once|

The notebook's cell 14 says fixed uniform modulation, but cells 12/15 both
instantiate `GeneralGRFKernel`: ten unconstrained learned coefficients here.
The ablation actually removes importance correction (`grf_ablation_sampler.py:61`
versus core `sampler.py:95`): load multiplies by weight, versus
`degree * weight / (1-p_halt)`. Sampling is on **L**, including its diagonal,
as in cell 7, rather than the positive grid adjacency. Degree means CSR nonzero
row count (including that diagonal). Same four chunks and per-node RNG
`seed + chunk_index + start_node` pair underlying walks: both recursions consume
the same random calls, independent of load; tiny support and first-step load
checks confirm this. Identical seeds with different worker counts would not pair.
The unused `utils/config.py` helper would set Cholesky 0 / CG .01 / Lanczos 1 /
64 probes. It is **not invoked** by this notebook or package initialization.
The initial printed preflight table listed that helper; checking its callers
corrected the distinction without changing any executed experiment setting.

## Numerical gates and Chebyshev scope

The existing degree-20 / 128-quadrature polynomial has max spectral errors
3.27424e-9 at beta 10, 8.53236e-6 at 20, and .00128028 at 40. It becomes
negative at 20/40, including on the tiny graph's actual eigenvalues. The
official exact heat family has **unbounded positive beta**, while our shared
kernel defaults to (.05,4). Therefore the optional Chebyshev arm is **deferred**
under the work order's explicit option: no clipping, new constraint, monkeypatch,
degree hook or global change. There is no Chebyshev fidelity or speed claim.

Tiny 6×6 checks cover symmetric normalized spectrum [0,2], deterministic
data/split regeneration, eigendecomposition versus independent matrix exponential,
both actual sampler classes and paired walk loads, covariance, normalized NLL,
all raw model/noise gradients, positive noisy Cholesky, exact latent mean and
variance, and observation NLPD against independent NumPy conditioning. Gate
tolerances: covariance/NLL/gradients and posteriors atol 1e-9, rtol 1e-8;
generating covariance atol 2e-13, rtol 2e-12. No clipping or added jitter.

All reported fits use their **own fitted kernel**, not a heat projection of a
learned GRF. For normalized training targets, H=K_TT+vI, mean=K_ST H^-1 y,
and latent marginal variance=diag(K_SS-K_ST H^-1 K_TS). Observation NLPD is
the mean univariate Gaussian negative log density with variance latent+v;
raw-unit NLPD adds log(train SD). RMSE rescales by train SD. The saved 810
means/latent variances are normalized; multiply variances by train SD squared
for raw units. NLL is negative log marginal likelihood divided by 90, evaluated
after the last update. Neither native 64-sample GRF prediction nor joint
test-set NLPD is substituted for these common marginal scores.

## Run record

Started from main `fab89cb1c57a7d9aae15039e69f4bce10e43bf51` (PR #8 merged).
PR #9 remains open; this branch has no dependency on it. Pinned source checkouts
are unchanged. Float64 CPU, four Torch threads / four official sampler workers.
All output, walks, data, splits, latent/noisy targets, losses, parameters,
optimizer/model checkpoints, posterior marginal variances, hashes, timings and
warnings remain in ignored `results/paper_c3/`.

Smoke uses 6×6, 32 walks/node and five updates. Pilot uses frozen 30×30 data,
256 walks/node and 20 updates. These are engineering checks, **not Table 5
measurements**. Full uses exactly 10,000 walks/node and 1000 updates per arm.
Each invocation has a two-hour wall-time alarm; full allocation is chosen only
after pilot timing. RSS self/child high-water marks are measured separately,
not aggregate memory or per-arm incremental peaks. This is not a GPU timing replication.

## Full results (all three arms complete)

One frozen seed, 90 observed / 810 test nodes, 1000 Adam updates each. All
RMSE scores target **noisy observations**, not the latent generating field. The
printed Table 5 numbers below are reference values with unresolved scoring and
data-generation differences; neither NLPD column is asserted to reproduce them.

|Model|RMSE raw|RMSE normalized|Obs NLPD raw|Obs NLPD normalized|NLL/obs|Noise variance normalized|
|---|---:|---:|---:|---:|---:|---:|
|Exact heat|0.510440|0.750254|0.822386|1.207525|1.261705|0.00924131|
|Principled GRF|0.557867|0.819962|0.844776|1.229915|1.262103|0.00513749|
|Ablated GRF|0.606043|0.890773|0.899182|1.284320|1.328818|0.00967241|

Table 5 (paper reference only): exact diffusion RMSE/NLPD .262/.090;
principled GRF .339/.339; ad-hoc GRF .573/1.265.

**Qualitative ordering replicated; quantitative Table 5 replication not established.**
Principled minus ablated: RMSE -0.048177, marginal NLPD -0.054405.
The advantage is substantially smaller than the printed table. Exact heat is
best on both common test metrics. This one field cannot establish a universal
necessity or superiority of importance correction. Unspecified amplitude/seed,
work-order noise, hop convention, optimizer initialization and marginal scoring
prevent an exact numerical replication claim; no settings were retuned.

Exact fitted beta=6.337524788, signal SD=4.235828775
(normalized units). Starts: raw beta/signal/noise 1/1/0, physical
beta/signal SD/noise variance 1.313261688/1.313261688/.693247181. Both GRFs
have the same seed-1 initial modulation and raw noise 0; their starting kernels
are different. Modulations below are ordered by recorded power 0…9:

```text
initial: 0.661352172, 0.266924098, 0.0616772582, 0.621317329, -0.451905965, -0.166130235, -1.52276848, 0.381683902, -1.02760858, -0.563052799
principled final: -2.65614036, 2.94610204, 0.761548497, -0.881639008, -0.376699281, 0.13189943, 0.0218705343, 0.0174253824, -0.00104638823, -0.000486139724
ablated final: 2.0763192, -7.48228965, -1.09902803, 7.77566971, 6.95899342, 7.1743414, 5.30401181, 7.16994767, 7.42977527, 8.58734493
```

For exact / principled / ablated, fitted noise variance in raw units is
0.00427766/0.00237807/0.00447721, all far below generating variance .1.
Noise decomposition alone does not establish predictive calibration. Saved latent
variance min/mean/max and common observation NLPD retain the full uncertainty evidence:

|Model|Latent variance min / mean / max (normalized)|Initial NLL|Last 100-update loss change|Final raw-gradient norm|
|---|---|---:|---:|---:|
|Exact heat|0.036881 / 0.520023 / 1.273998|1.413761|-0.000256|0.00134393|
|Principled GRF|0.060208 / 0.588369 / 1.417473|5.495725|-0.007874|0.787436|
|Ablated GRF|0.195348 / 0.726420 / 1.019340|1.425794|-0.005819|0.00405646|

These are **fixed-budget results, not certified converged optima**. Principled
loss oscillates near the end (updates 800/900/1000: 1.26756/1.27003/1.26216);
ablated loss continues falling. Raw-gradient norms depend on parameterization
and must not be treated as a cross-family convergence ranking. The ablation
changes feature scaling as well as importance correction: on a constant-degree
nonzero pattern its expected powers differ by ((1-halt)/degree)^j, which free
modulation could absorb. Boundary degrees vary. Equal starting coefficients /
Adam steps therefore do not equalize initial covariances or optimization difficulty.

## Validation, cost and failure accounting

All tiny gates passed: maximum covariance error 1.34e-15 and raw-gradient
error 2.23e-16. Independent full 900-node audits use SciPy matrix exponential
for exact heat, explicit dense walk-feature products for GRFs, NumPy Gaussian
conditioning and analytic derivatives of NLL: G=(H^-1-alpha alpha.T)/(2m),
with dNLL/dtheta=trace(G dH/dtheta). They agree with saved scores/moments and
upstream autograd: max NLL error 6.67e-16, mean/variance error 1.22e-14,
gradient error 2.30e-14. All checkpoint hashes, nine finite loss histories,
raw-noise transforms and frozen pilot/full data validate. Resume skips all three
full fits without resampling or training.

|Beta|Cheb20 max spectral error|Derivative max error|Minimum on tiny eigenvalues|
|---:|---:|---:|---:|
|1.31326169|8.88178e-16|2.77556e-16|0.0723295|
|4|2.88658e-15|8.71352e-15|0.000335463|
|10|3.27424e-09|4.39299e-09|3.24604e-09|
|20|8.53236e-06|3.93112e-06|-2.94008e-06|
|40|0.00128028|0.000173947|-0.000562819|

|Model|Sampling + feature-save s|Model/optimizer setup s|Training s|Posterior s|Arm subtotal s|
|---|---:|---:|---:|---:|---:|
|Exact heat|0.000000|0.349913|332.719455|0.038818|333.108186|
|Principled GRF|44.812744|0.004320|61.950694|0.201396|106.969153|
|Ablated GRF|40.141173|0.001074|65.813708|0.218827|106.174782|

Exact heat has no walk sampling (the sub-microsecond timer above is bookkeeping).
Data generation/fingerprinting/save: .233731 s once, shared; arm subtotals exclude
validation and final checkpoint writes. Pilot projection was 423 s; measured full
arm subtotals sum to 546.252 s. Allocation: 7200 s alarm, 4 threads/workers,
Apple M4 / 16 GiB RAM. Parent process cumulative peak RSS .984528 GiB;
child high-water marks .036896/.095398/.098190 GiB, not summed concurrent usage.
No speedup, GPU or scaling conclusion is drawn. No timing baseline was inferred.

All primary gates/fits completed. The optional Chebyshev range audit fails
positivity; that arm is deferred, not reported complete. Sparse CSR / invariant UserWarnings remain in logs; no
experiment NumericalWarning. Initial smoke logging emitted a scalar-conversion
warning, fixed by detaching diagnostic values only; original log retained.
Early PDF-reader/path/dtype/syntax tooling errors and the source-helper correction
are recorded in `preflight_notes.txt`; none changed data or experiment settings.
Baseline eight tests pass; targeted eight pass in each existing environment.
Existing strict-CG test warnings remain visible. `git diff --check` passes.

## Provenance and reproduction

Train mean / sample SD: -0.025370012845143636 / 0.68035632220220943.
Full dataset fingerprint (L, train, test, normalized y, latent, noisy, center, SD):
`fca6e9f2ad4a9ccfcefab1809e97d605a7bc2ef07650c4d31f58d6e2dbd5af66`.
Latent / noisy SHA256:
`bcc184ba96acfe842e273179586ea4adb57b67a62fc7a120eba1276fe29e1c34` /
`36f7b07e583654cb01f96db40aac01f869796c57891a7cb001e5ce809799c843`.
Train / test IDs, tensor/file hashes and original training manifest remain in
`full_data.pt`, `artifact_hashes.json` and `training_manifest.json`.
Environment: macOS 26.6.2 arm64, Torch 2.11.0, GPyTorch 1.15.2, linear_operator
.6.1, NumPy 1.26.4, SciPy 1.17.1, NetworkX 3.3. No packages changed.

From the existing pinned traffic environment/checkouts (README setup; no PEMS
download is required for this synthetic study), keep prior logs and use new names:

```sh
mkdir -p results/paper_c3
results/traffic_env/bin/python -u experiment_grf_ablation_replication.py check > results/paper_c3/check_replay.log 2>&1
results/traffic_env/bin/python -u experiment_grf_ablation_replication.py smoke > results/paper_c3/smoke_replay.log 2>&1
caffeinate -i results/traffic_env/bin/python -u experiment_grf_ablation_replication.py pilot > results/paper_c3/pilot_replay.log 2>&1
caffeinate -i results/traffic_env/bin/python -u experiment_grf_ablation_replication.py full > results/paper_c3/full_replay.log 2>&1
results/traffic_env/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

`full` resumes only verified complete arms (matching configuration/data/checkpoint
hashes). To regenerate fits, archive `results/paper_c3/` first; do not overwrite
prior evidence. JSON rows and checkpoints retain complete raw parameters/losses;
`results.csv` has all nine smoke/pilot/full metrics, explicitly distinguished by
stage. `independent_audit_command.txt` preserves the exact additional local audit
command; `independent_audit.json`, `checks.json`, logs, walk matrices and
`provenance.json` preserve diagnostics and provenance. All are ignored local
paths, not remote downloadable artifacts. The tracked table suffices; no plot.

For planner review, the next **single** PR could validate the deferred Chebyshev
comparator at matched heat parameters with a predeclared supported beta/degree
regime. It has not begun; no ring, traffic, wind, BO, extra seed or tuning study.

Changed: one 200-line experiment script and this report (two paths). The report
exceeds the ordinary 200-total-line soft guideline to preserve required source,
numerical, result and reproduction evidence; Python meets the 200-line target.
Deleted: nothing. Kept: existing experiments, core kernels/models, defaults,
AGENTS, broad README, dependencies, pinned authors code and read-only toolbox.
One PR for review; do not merge. No pending primary runs.
