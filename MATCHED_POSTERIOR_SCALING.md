# Matched posterior uncertainty: qualification blocked

**No common P* was selected and no 5K–50K uncertainty prediction was run.**
The first tiny GeneralGRF case failed the strict native-sampler parity comparison.
This leaves the scaling question unanswered; it establishes neither a variance
accuracy failure at scale nor a performance advantage for either method.

## Frozen protocol and implementation

Base main: `31fd991282f8f026931edb71da7a3bbc8d1e985e` (merged PR18).
CPU float64, four Torch threads, Apple M4/16 GiB, macOS26.6.2; existing traffic
environment: Torch2.11.0, GPyTorch1.15.2, linear_operator0.6.1, NumPy1.26.4,
SciPy1.17.1. No software upgrades or training updates.
Pinned clean authors' trees: Fast-Graph-GP
`268fff9a3724cdd480dce4b2946068d8211b4f30`; experiments
`274b7ee393ee516585552f053e9d4a9c6ad2222f`.

The 297-line driver reuses PR18's polynomial root, recurrence, sparse guard,
scores and dense reference, PR15–17 model/checkpoint helpers, and library CG.
The existing qualification entry point is Poly-only and has no matched General
adapter or residual-bracket audit. One study driver adds that authorized scope
without changing the historical experiment.
The user explicitly approved a 300-line cap before execution; two tracked files
are added. Kernels, inference defaults, training and dependencies are unchanged.

General uses its actual lazy feature operator B=Phi and K=BB^T, including the
transpose; Poly uses sqrt(outputscale) q(L). Both prepare root/features and lazy
A/C once per invocation, solve the analytic mean, and stream width32 centered
paths `u=(Bz)[test]-C solve(A,(Bz)[train]+sqrt(nu)e)` into Chan/Welford moments.
Observation variance adds nu once. CG is 1e-4/max1000/no preconditioner/no SLQ.
Dense Cholesky and actual-walk references are outside candidate timers/guards.
Large-path execution and timing code is present but **not qualified or exercised**.

Provenance froze all 48 PR17 Poly/General fitted endpoints and PR16 development
identities, complete/100-update states, checkpoint/input hashes, original walk
metadata, parameters, data fingerprints and selected audit IDs before scoring.
Historical input hashes were checked as subsets, without altering old directories.
No missing checkpoint or large walk was reconstructed. The permitted new tiny
W1024 cache uses seed1671036, J3, halt.5, four walk workers under the new output
directory; the historical tiny cache had W4096 and was incompatible.

## Actual Stage A result

Only n36/r9/H/initial General, seed521, width32 was attempted. Identical z/e
perturbations were passed to the actual pinned `pathwise_conditioning` with zero
y and its batch-major convention. The adapter-versus-native path assertion
failed at atol1e-10/rtol1e-9: **158/288 elements differed**, maximum absolute
error **7.154941639e-9**, maximum relative error **7.538366823e-7**.
No threshold, seed, solver setting or implementation was changed and no failed
case was rerun. The cause of the discrepancy has not been established; the small
absolute difference alone does not show materially inaccurate posterior variance.

| Check completed before failure | Measured result |
|---|---:|
| Actual-walk BB^T vs official full covariance, max abs | 0 |
| Full deterministic path-map covariance vs dense Schur complement, max abs | 2.776e-16 |
| Analytic mean vs dense, max abs / relative L2 | 1.881e-6 / 6.645e-6 |
| True mean RHS relative residual | 5.488e-6 |
| Selected-column true residual, max / mean / p95 | 8.038e-6 / 3.075e-6 / 7.343e-6 |
| Maximum bracket relative width | 8.221e-13 |
| Zero RHS / nonzero zero-RHS residual | 0 / 0 |

The residual bracket `upper=d-c^T x-x^T r`,
`lower=upper-||r||²/nu` was positive, within .002 relative width, and contained
the independent dense diagonal to the prescribed roundoff tolerance at all three
tiny selected positions [8,6,3], vertices [23,20,26]. Three-column local products
avoid a legitimate tiny basis colliding with the full-shape guard. This validates
only this tiny bracket, not the remaining numerical qualification or large audit.
The feature/root action assertion passed before native parity failed. No completed
streaming-variance, same-draw variance or MC-prefix result was emitted.

| Qualification / campaign | Completed / attempted / scheduled |
|---|---:|
| Stage A case invocations | 0 / 1 / 12 |
| New General n1000 MC combinations, each of P512 and P1024 | 0 / 0 / 24 |
| New Poly tiny/n1000 regression invocations | 0 / 0 / 2 |
| Final predictions, 48 fits × three streams | 0 / 0 / 144 |

Eleven qualification invocations were explicitly marked unattempted. P* remains
unset. Prior PR18 supports both budgets (24/24 each); it cannot substitute for
the unperformed General qualification. No final schedule was activated.

## Metrics, timing and limitations

The only scores are the tiny **exact reference**, not a completed sampled model:
raw/normalized RMSE .575605/1.143530, raw/normalized marginal observation NLPD
.893438/1.579891, descriptive 95% observation coverage 1.0 on nine test nodes.
Raw variance is normalized variance × scale²; raw NLPD adds log(scale).
These scores do not establish calibration or between-method predictive quality.

Instrumented partial phase times: preparation1.297ms, analytic mean16.202ms,
selected-column audit50.857ms. Whole attempt2.490s; sampled process-group peak
RSS426.75MiB includes imports, dense references and guards, may miss peaks, and
is not candidate-only memory. No complete prediction time, reconstructed total,
amortized total, paired ratio or scaling figure exists. None is inferred.
The 1800s/case, 4h total and 12GiB sampled RSS caps were not reached.
The launcher stopped on failed JSON status despite process exit0.
Four PyTorch sparse/storage UserWarnings are retained; no NumericalWarning was
recorded in this attempt. No jitter, clipping or variance repair was used.

## Evidence and reproduction

Ignored `results/matched_posterior_scaling/` contains the work order, manifest
(source/package/pin/input/checkpoint hashes and qualification order), raw failure,
warnings, reference NPZ, bracket NPZ/JSON, resource record, status CSVs, summary,
evidence hashes and local metadata/watchdog scripts adapted from PR18. No new
scientific calculations are hidden in those local launcher/table scripts.
`unattempted_final.csv` lists all 144 planned outcomes with no invented P*.

With the retained local checkpoint/data/environment prerequisites, inspect evidence
without rerunning a failed case:

```sh
cat results/matched_posterior_scaling/summary.json
cat results/matched_posterior_scaling/qualification_status.csv
cat results/matched_posterior_scaling/qual_n36_r9_0_General_initial_s521.log
results/traffic_env/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

The original preparation/execution commands were
`results/traffic_env/bin/python results/matched_posterior_scaling/freeze_local.py`
and `caffeinate -i results/traffic_env/bin/python -u results/matched_posterior_scaling/launch_local.py qualification`.
They intentionally refuse to overwrite existing attempts. A single-case invocation
is `results/traffic_env/bin/python experiment_matched_posterior.py '["qual",36,9,0,"General","initial",521,32]'`;
rerunning the failed case or proceeding requires planner authorization. Full
reproduction needs ignored checkpoints/walks; a fresh clone alone is insufficient.

Baseline and final existing suites: **9/9 in each environment**, with the same
three existing strict-CG NumericalWarnings per suite; py_compile and diff checks
pass. All **3,570** prior artifact hashes and the complete inventoried filename
set are unchanged. Added: one297-line driver and this report; deleted: nothing;
kept: all prior code, training, kernels, checkpoints and results. Open issue:
native parity must be resolved/reviewed before any further qualification or scaling.
