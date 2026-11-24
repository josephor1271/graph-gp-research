# Post-hoc traffic cutoff-3 diagnostic

**Consistent with deterministic truncation:** the exact cutoff-3 control improves
both RMSE and marginal observation NLPD over heat on all three already-used
snapshots, by approximately the magnitude of GRF C's advantage. It is slightly
better than the two-seed GRF C mean on both metrics each date; sampling is not
needed to reproduce these advantages within this frozen setup. Individual walk
seeds can still improve or worsen the deterministic result. This is not causal
proof or statistical generalization from three related, previously inspected dates.

Spatial regression at each stored Monday 17:30 timestamp, not forecasting:
1016 vertices / 1173 edges; the same 250/75 sensor split and train-only scaling
as PR #6. All 12 models have 1000 AdamW .05 updates, float64, four CPU threads,
independent learned beta/signal/noise and beta in (0,2). Deltas below are relative
to heat on the same date; lower is better. Every raw beta/signal/noise start is
1/1/0 (physical 1.462117/1.313262/.693247); RNG is reset to zero before fitting.
Signal/noise are standardized-target
standard deviation/variance. NLPD adds observation noise exactly once; mph NLPD
is normalized NLPD + log(train_std), also saved in raw results.

| Date (2017) | Model / walk seed | RMSE mph | Obs NLPD norm | ΔRMSE | ΔNLPD | NLL/obs | Beta | Signal std | Noise variance |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|01-09|Heat (Cheb20)|15.738451|1.421688|+0.000000|+0.000000|1.356774|1.995557|1.456977|0.237140|
|01-09|Deterministic Q2²|15.596541|1.345409|-0.141910|-0.076280|1.369465|1.369140|1.356058|0.008214|
|01-09|GRF C / 101|15.701407|1.359017|-0.037045|-0.062671|1.372251|1.337333|1.360686|0.009830|
|01-09|GRF C / 202|15.514496|1.341682|-0.223955|-0.080006|1.371790|1.352181|1.362302|0.008494|
|02-06|Heat (Cheb20)|16.065616|1.377650|+0.000000|+0.000000|1.352924|1.995959|1.436681|0.246440|
|02-06|Deterministic Q2²|15.810487|1.307753|-0.255129|-0.069897|1.363099|1.459741|1.341068|0.007147|
|02-06|GRF C / 101|15.898595|1.317218|-0.167021|-0.060432|1.368829|1.405125|1.353156|0.009038|
|02-06|GRF C / 202|15.787873|1.309319|-0.277744|-0.068331|1.364173|1.439409|1.347149|0.007214|
|03-06|Heat (Cheb20)|16.034552|1.476915|+0.000000|+0.000000|1.361634|1.995438|1.400853|0.287205|
|03-06|Deterministic Q2²|15.494595|1.385215|-0.539957|-0.091701|1.366569|1.447588|1.346685|0.007617|
|03-06|GRF C / 101|15.554065|1.397026|-0.480487|-0.079889|1.369795|1.419926|1.353369|0.008939|
|03-06|GRF C / 202|15.503558|1.384569|-0.530994|-0.092346|1.371446|1.381773|1.360297|0.008704|

Upstream modulation is sigma*(-beta/2)^k/k!, including k=0. Supplying exact
I,L,L² to the existing bounded DiffusionGRFKernel gives Phi=sigma*Q2 and
K=Phi*Phi.T, where Q2=I-beta*L/2+beta²*L²/8. No new kernel/GP class is used.
The six tiny/traffic gates at beta .4/1/1.9 compare indexed covariances, normalized
NLL and all three raw-parameter gradients to an independent dense expression.
Max covariance/NLL/gradient errors are 6.67e-16 / 0 / 1.12e-16; tolerances are
1e-10/1e-9 (NLL 1e-11/1e-10). Full latent eigenvalues are positive (minimum
.431164); noisy Cholesky succeeds, without clipping or added jitter. Full gate
and fixed-parameter kernel discrepancies are in validation.json and the PR.
At traffic beta=1.9, truncation/heat difference is .770209; finite GRF/truncation
is .140824/.135935, and finite GRF/heat .815112/.800812 (seeds 101/202).
These covariance differences are not predictive accuracy scores.

Heat beta approaches the shared upper bound (1.9954–1.9960); the deterministic
control is interior (1.3691–1.4597). Heat has lower training NLL despite worse test
scores. Separate fitted parameters, finite features and the common bound may
explain residual differences. The next single experiment would be a separately
validated wider-beta heat control on these same frozen snapshots; it is not run here.

New deterministic training seconds (Jan/Feb/Mar): 15.242, 14.203, 14.134.
Nine PR #6 baselines are reused only after checking data/checkpoint hashes,
1000-loss histories, matched initial loss and recomputed own-kernel NLL, scores
and physical parameters (within 1e-11). Their setup/train/walk costs are historical;
new posterior evaluation times are separately measured. GRF walk costs 2.137291/
1.964202 s remain recorded (shared across dates); no speed/scaling claim is made.
All timings, raw initial/final parameters, losses, 75 posterior means/latent
variances per fit and provenance remain in ignored results/traffic/truncation_diagnostic/.

Started from live main 2dbf18d850ebb07618e43f3ff2cf7870740ea0cf; PR #8 was open
and unmerged. This experiment does not depend on it. Source pins: Fast-Graph-GP
268fff9a3724cdd480dce4b2946068d8211b4f30; experiments
274b7ee393ee516585552f053e9d4a9c6ad2222f. Frozen data SHA256:
```text
802e5e523b5db6ff9d13da3388c566de7d0212b09b23497a3d0e50a0385ad71e
```

Existing traffic environment, pinned checkouts and cached PEMS data are prerequisites.
Keep earlier logs; completed verified cases are skipped on resume.

```sh
mkdir -p results/traffic/truncation_diagnostic
results/traffic_env/bin/python -u experiment_grf_truncation_diagnostic.py --check > results/traffic/truncation_diagnostic/check_replay.log 2>&1
caffeinate -i results/traffic_env/bin/python -u experiment_grf_truncation_diagnostic.py --pilot > results/traffic/truncation_diagnostic/pilot_replay.log 2>&1
caffeinate -i results/traffic_env/bin/python -u experiment_grf_truncation_diagnostic.py > results/traffic/truncation_diagnostic/full_replay.log 2>&1
results/traffic_env/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
```

Baseline eight tests pass; targeted eight pass in each environment, including
existing strict-CG test warnings. Six numerical gates and independent NumPy
marginal-NLPD/Jacobian checks pass; all 12 fits and resume hashes validate.
No failed equivalence gate, optimizer failure or incomplete fit. An auxiliary
integrity scan initially matched a manifest as a fit; its log was preserved,
file discovery corrected, and all checks passed. Sparse CSR/invariant and
upstream deprecation UserWarnings remain visible; no experiment NumericalWarning.

Changed: one 193-line experiment script and this note (two paths, <=200 Python
source lines). Deleted: nothing. Kept: existing experiments, broad README,
AGENTS, core kernels/model, upstream, packages and read-only toolbox. No pending
runs. Open one PR for review, do not merge, and stop.
