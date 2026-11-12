# Appendix C.3 mesh importance-sampling ablation

**Qualitative ordering replicated; quantitative Table 5 replication not established.**
On one frozen field, exact heat beats principled GRF, which beats ablated GRF.
The GRF advantage is smaller than the paper's; these are fixed-budget fits.

## Source reconciliation

Sources: [paper](https://arxiv.org/abs/2509.03691),
[pinned notebook](https://github.com/MatthewZhang473/Graph-Random-Features-for-Scalable-Gaussian-Processes/blob/274b7ee393ee516585552f053e9d4a9c6ad2222f/experiments/ablation/2d_mesh.ipynb),
[pinned core](https://github.com/MatthewZhang473/Fast-Graph-GP/tree/268fff9a3724cdd480dce4b2946068d8211b4f30).
The supplied 23-page `grf.pdf` has SHA256
`8e453910edf00c8b16ed9b10885407951eee5ad41aeb7f1d53957e92f37771ee`.
Notebook cells below are zero-indexed. No full notebook-protocol run was added.

|Setting|Paper source / value|Pinned notebook / code|Executed choice|
|---|---|---|---|
|Graph|C.3 p16, 30×30|cells 2–4, unweighted grid|900 nodes; upstream symmetric normalized L|
|Generating beta|C.3 p16, 10|cell 2, 40|10|
|Observed fraction|C.3 p16, 10%|cell 2, 20%|90 train / 810 test|
|Walk budget|C.3 p16, 10,000/node|cell 2, same|10,000 per sampler|
|Cutoff|C.3 p16, exceeding 10 steps truncated|cell 2, 8|API argument 10: powers 0…9, nine contributing hops; final unrecorded hop consumes RNG|
|Observation noise|C.3 / §4.1 magnitude unspecified|cells 2–4, SD .1 (variance .01)|variance .1 required by work order, not established by C.3|
|Signal variance / seed / halt|C.3 unspecified|cell 2: 10 / 1 / .01|These notebook defaults, frozen before scores|
|Generation jitter|Unspecified|cell 3, 1e-6 I|None; exact heat covariance passes Cholesky|
|Split / normalization|Not fully specified|cell 4, NumPy choice; train-only mean/sample SD|Same procedure, seed 1|
|Learning|C.3 p16, Adam .01, 1000 updates|cell 2, .05/500; `utils.py:18` uses AdamW with decay .01|Explicit Adam .01, decay 0, 1000 updates|
|Initialization|Unspecified|Diffusion raw beta/signal 1/1, noise 0; GeneralGRF randn|Official defaults; reset Torch seed 1 per model, pairing GRF coefficients|
|Inference|§3 uses CG/stochastic gradients; C.3 budgets absent|Active defaults: Cholesky 800, CG 1, Lanczos 20, 10 probes|Explicit Cholesky 1000, exact log probability; both thresholds use Cholesky at 90 observations|
|Scoring|Table 5 p17; marginal/joint convention unspecified|`utils.py:59–65`: exact joint noisy NLPD; GRF diagonal latent NLPD from 64 samples|Own-kernel exact posterior; common marginal observation NLPD|

Cell 14 incorrectly describes fixed uniform modulation: cells 12/15 both use
`GeneralGRFKernel`, with ten learned coefficients here. Sampling is on **L**,
including its diagonal; degree means CSR nonzero row count. The load update is
`degree * weight / (1-p_halt)` (`sampler.py:95`), versus `weight` in the ablation
(`grf_ablation_sampler.py:61`). Both actual sampler classes consume paired draws
with four identical chunks and per-node seed `1 + chunk_index + start_node`;
tiny support/first-step checks confirm this. Changing worker count breaks pairing.
The notebook never calls `set_gp_defaults` (0/.01/1/64); the initial preflight
report confused this unused helper with active settings, corrected without changing runs.

## Numerical gates and Chebyshev scope

The tracked script implements tiny 6×6 checks: deterministic data/split, symmetric
normalized spectrum [0,2], eigendecomposition versus matrix exponential, actual
sampler/load behavior, covariance/NLL/raw gradients, noisy Cholesky, and posterior
mean/latent variance/observation NLPD versus NumPy conditioning. Maximum covariance
and gradient errors: 1.34e-15 / 2.23e-16. Tolerances: atol 1e-9, rtol 1e-8;
generating covariance atol 2e-13, rtol 2e-12. No clipping or added jitter.

**The full independent 900-node audit was a separate local check; it is not fully
implemented in the tracked experiment script.** It used SciPy matrix exponential,
dense walk-feature products, NumPy conditioning and analytic NLL derivatives.
Reported maximum NLL / posterior-moment / raw-gradient errors:
6.67e-16 / 1.22e-14 / 2.30e-14. Its exact command and outputs remain ignored in
`results/paper_c3/independent_audit_command.txt`, `independent_audit.log` and
`independent_audit.json`; the script's `check` command does not reproduce this full audit.

Chebyshev is **deferred**, with no fitted comparator or fidelity/speed claim.
Its shared kernel defaults to beta (.05,4); official exact beta is unbounded positive.
The tracked degree-20 / 128-quadrature range check gives:

|Beta|Max spectral error|Derivative max error|Minimum on tiny eigenvalues|
|---:|---:|---:|---:|
|10|3.27424e-09|4.39299e-09|3.24604e-09|
|20|8.53236e-06|3.93112e-06|-2.94008e-06|
|40|0.00128028|0.000173947|-0.000562819|

## Full results and limitations

All three arms completed 1000 updates. Scores target **noisy observations**.
NLL/obs uses each fitted kernel after the final update. Observation NLPD adds
learned noise variance once to latent marginal variance; raw NLPD adds log(train SD).
RMSE rescales by train SD; saved normalized posterior variances rescale by SD squared.

|Model|RMSE raw|RMSE normalized|Obs NLPD raw|Obs NLPD normalized|NLL/obs|Noise variance normalized|
|---|---:|---:|---:|---:|---:|---:|
|Exact heat|0.510440|0.750254|0.822386|1.207525|1.261705|0.00924131|
|Principled GRF|0.557867|0.819962|0.844776|1.229915|1.262103|0.00513749|
|Ablated GRF|0.606043|0.890773|0.899182|1.284320|1.328818|0.00967241|

Table 5 reference RMSE/NLPD: exact .262/.090; principled .339/.339; ad-hoc .573/1.265.
Principled minus ablated: RMSE -0.048177, NLPD -0.054405. One seed, unspecified
paper settings, noise choice, hop convention, initialization and scoring differences
prevent exact numerical replication or universal importance-correction claims.
No settings were retuned. Exact fitted beta/signal SD: 6.337524788/4.235828775.
Learned raw noise variances are below generating .1; this alone does not establish predictive calibration.

Convergence is uncertified: principled loss oscillates at updates 800/900/1000
(1.26756/1.27003/1.26216); ablated loss continues falling. Identical initial GRF
coefficients do not match initial covariances or optimization difficulty: at constant
CSR degree, ablated expected powers differ by ((1-halt)/degree)^j, absorbable by
free modulation; mesh boundary degrees vary. Full parameter/loss/variance diagnostics
remain in the ignored checkpoints and audit rather than duplicated here.

|Model|Sampling + feature-save s|Model/optimizer setup s|Training s|Posterior s|Arm subtotal s|
|---|---:|---:|---:|---:|---:|
|Exact heat|0.000000|0.349913|332.719455|0.038818|333.108186|
|Principled GRF|44.812744|0.004320|61.950694|0.201396|106.969153|
|Ablated GRF|40.141173|0.001074|65.813708|0.218827|106.174782|

Exact has no sampling; data generation/fingerprinting/save costs .233731 s once.
Subtotals exclude validation/final checkpoint writes; no speedup/GPU/scaling claim.
Pilot projected 423 s; measured arm subtotals sum to 546.252 s. Allocation: 7200 s
alarm, Apple M4 / 16 GiB, float64 CPU, four Torch threads/four sampler workers.
Cumulative parent RSS .984528 GiB; child peaks .036896/.095398/.098190 GiB are
separate high-water marks, not concurrent totals. All primary gates/fits passed;
Chebyshev failed positivity. Sparse UserWarnings and initial diagnostic/preflight
issues are preserved in logs/`preflight_notes.txt`; no experiment NumericalWarning.
Eight tests pass in both environments; existing strict-CG test warnings remain visible.

## Provenance and reproduction

Base main `fab89cb1c57a7d9aae15039e69f4bce10e43bf51`; independent of PR #9.
Train mean/sample SD: -0.025370012845143636 / 0.68035632220220943. Dataset SHA256:
`fca6e9f2ad4a9ccfcefab1809e97d605a7bc2ef07650c4d31f58d6e2dbd5af66`.
Environment: macOS 26.6.2 arm64; Torch 2.11.0, GPyTorch 1.15.2, linear_operator .6.1,
NumPy 1.26.4, SciPy 1.17.1, NetworkX 3.3. Use the existing pinned traffic setup;
no PEMS download is needed. Smoke: 6×6/32 walks/5 updates; pilot: 30×30/256/20.
Both are engineering checks, not Table 5 measurements. Full: 30×30/10,000/1000.

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

Verified `full` resume skips three completed arms using config/data/checkpoint hashes.
Archive prior outputs before fresh fits. Ignored `results/paper_c3/` retains all nine
stage-labeled fits (`results.csv`), splits/targets (`full_data.pt`), hashes, original
training manifest, checkpoints/walks, warnings and provenance; these are local artifacts.
For planner review only: the deferred Chebyshev comparator needs a supported beta/degree
regime. No follow-up began. Scope remains one 200-line script and this report. Cleanup
changes only this report, adding/deleting no files or dependencies; core and experiments
unchanged. No pending primary runs. Leave PR #10 unmerged.
