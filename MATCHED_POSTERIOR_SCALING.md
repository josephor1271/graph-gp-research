# Matched posterior uncertainty: no common MC budget qualifies

**The amended numerical qualification passes; the frozen Monte Carlo qualification
does not.** General passes18/24 combinations at512 paths and23/24 at1024.
No common P* was selected. Per the original stopping rule, all144 planned
5K–50K predictions and their designated large guarded audits remain unattempted.
No scaling, speed, reconstructed-total or memory advantage is claimed.

## Protocol and preserved evidence

The benchmark method is **official GeneralGRF features with a batched float64
posterior inference adapter**, compared with our existing Poly root. It is not
unmodified native prediction. Both use their own fitted covariance K=BB^T,
analytic-form iterative mean and centered paths with conditioning noise once;
unbiased streaming variance adds observation noise once for scoring.
CPU float64/four threads, width32, CG1e-4/max1000/no preconditioner/n_tridiag0;
no training, kernel, solver, seed or dependency changes. No SLQ for prediction.
PR16 n1000/r10,11 weighted normalized graphs use train/test250 each and training-only
normalization: H covariance exp(−1.5L), M .5exp(−1.5L)+.5(L/2)², raw noise variance.1.

Main base31fd991 contains merged PR18; this continues PR19 from8a08096.
After reviewing the two preserved failures, the planner authorized strict
comparisons against float64 algebra for the adapter and independently verified
float32 noise-diagonal conversion for native General. Both retain original
atol1e-10/rtol1e-9. The adapter noise is unchanged. The residual-aware path bound
includes measured operator/RHS differences, known noise perturbation and float64
roundoff; residual<=2e-4 and same-draw variance L2<=.002 gates are unchanged.
This is an explicit post-observation protocol amendment, not a relabeled pass.

The original7.15494e-9 path mismatch and1.84986e-8 native noise-diagonal shift,
all old thresholds, manifests, logs, diagnostics and source snapshots remain.
Git commits fd6febe/8a08096 retain their complete reports and one-time diagnostic
code. All **3,653 earlier result files and the inventoried filename set** are
unchanged. New evidence is isolated in ignored
`results/matched_posterior_scaling/matched/`. The active driver shrank432→358
lines by removing the one-time trace investigation and saved-array explanation.

## Numerical qualification

Completed the minimal original n36/r9/H/initial General seed521 regression, then
all11 remaining original invocations: tiny stress General, tiny Poly, one fitted
n1000 Poly regression and eight prescribed General states on r10/11. General uses
seeds521/522/523 and nested512/1024 prefixes; prior Poly421/422/423 support was
verified without rerunning PR18. All **53 state/stream/prefix numerical checks**
pass; no NumericalWarning, invalid variance, jitter/clipping or zero sampling RHS.

The amended tiny initial strict algebra discrepancy is<=1.111e-15. Actual native
CG residual7.624e-6 and identical-draw variance error1.510e-7 pass. Native noise
.6932471990585327 matches the explicitly checked float32 conversion of adapter
noise.6932471805599453; the entire conditioning-matrix difference is that scalar
shift times identity. The tiny stress regression also passes; no unexpected
operator difference was accepted. Independent actual-walk covariance/full-map
checks and streaming-vs-torch.var checks retain strict tolerances.

| Maximum across completed checks | Measured | Gate |
|---|---:|---:|
| Sampling true per-RHS relative residual | 1.41524e-4 | 2e-4 |
| Mean true residual | 8.98123e-5 | 2e-4 |
| Mean max abs / relative L2 discrepancy | 1.81653e-4 / 8.06705e-5 | max abs .002 |
| Variance L2 vs identical-draw Cholesky | 6.12939e-6 | .002 |
| Selected-column relative bracket width | 8.00406e-6 | .002 |

All12 selected-column audits executed under guards, with positive lower bounds
and independent dense containment to the original roundoff tolerance. Their
maximum true reference residual was8.63342e-4: the oracle's prescribed acceptance
uses finite residuals and verified bracket width, rather than assuming requested
CG tolerance is a per-column guarantee. This does not relax the mean/path gates.
These are tiny/n1000 audits, not evidence that large guarded audits executed.

## Frozen MC decision

Each row below covers two development graphs × three streams, with their own
analytic finite-kernel posterior as reference. H/M share data; streams/prefixes
are not independent datasets. Targets: latent L2<=10%, observation L2<=5%.
Times are measured preparation+mean+paths+streaming moments; no projections.

| General state | P | Latent error % range | Observation error % range | Joint passes | Prediction ms range |
|---|---:|---:|---:|---:|---:|
|initial H|512|5.93–6.81|3.00–3.36|6/6|508.7–622.2|
|initial H|1024|3.98–4.59|2.01–2.30|6/6|987.2–1115.8|
|stress H|512|5.71–7.09|5.24–6.48|0/6|843.1–1006.6|
|stress H|1024|4.07–5.59|3.73–5.11|5/6|1694.7–1895.7|
|fitted H|512|6.00–6.91|2.73–4.17|6/6|494.9–648.6|
|fitted H|1024|4.06–4.73|1.88–3.07|6/6|1000.7–1198.7|
|fitted M|512|5.55–6.88|0.67–1.43|6/6|503.4–628.7|
|fitted M|1024|3.93–4.78|0.47–0.99|6/6|1009.2–1145.6|

Latent gates pass24/24 at both budgets; observation/joint counts are18/24 and
23/24. Prior Poly supports24/24 at each budget and its new regression passes,
but the common rule requires **all24 General combinations**, including stress.
We did not discard the stress control because fitted states passed.

The sole1024 miss is **r10/H/stress/seed522**: latent5.594369%, observation
**5.111535%**. Its true sampling residual1.33018e-4 and same-draw variance
error2.75251e-6 pass. Exact Cholesky conditioning of those identical perturbations
still gives observation error **5.111562%**. This is finite-sample MC error,
not an inference defect cured by tighter CG. Stress uses prescribed modulation
[1.5,−3.225,1.35], noise variance.05; no state or seed was retuned.
Aggregate error is not pointwise: worst1024 latent/observation node errors are
18.42%/15.09%, and worst per-state p95 errors9.82%/8.12%.

## Predictive scores, costs and limitations

General1024 sampled scores below range over the same six combinations per state.
RMSE uses the analytic-form iterative mean, not sample mean; NLPD is marginal
**observation** NLPD. Exact stress coverage is only72.4–73.2% at nominal95%, so
numerical fidelity does not establish calibration of this deliberately fixed state.

| State | Raw RMSE | Raw NLPD | 95% observation coverage |
|---|---:|---:|---:|
|initial H|.561338–.598339|.868882–.948074|.980–.984|
|stress H|.626936–.681073|1.537898–1.625768|.716–.736|
|fitted H|.554718–.585408|.829337–.895974|.944–.964|
|fitted M|.575216–.599359|.869220–.908872|.972–.976|

All normalized/raw scores, observation SDs/interval widths, exact references,
per-node variances and residuals remain in CSV/JSON/NPZ. Raw variance multiplies
by scale²; raw NLPD adds log(scale), with training-only normalization.
Qualification attempt wall totals71.454s, max9.112s; sampled process-group peak
515.70MiB includes imports, dense references and guards. Candidate phase timers
exclude references/residual audits/scoring/I/O, but include guard overhead and
library warmup. These development timings/RSS do not support a fair large-scale
speed or memory claim. No1800s/case,4h total or12GiB cap was hit.
Only two previously seen development datasets and fixed states were checked;
no large uncertainty, statistical significance, population calibration or universal
sample-budget guarantee follows. All144 final outcomes are explicitly unattempted
(36 each at5K/10K/25K/50K); no final schedule was activated or budget increased.

## Reproduction and verification

Retained local evidence includes `amendment.md`, frozen `manifest.json`,
`selection.json`, `rows.csv` (all53 checks), `summary.json`, per-case arrays/logs,
resource records, `unattempted_final.csv` and preservation hashes. Metadata-only
local launch/table scripts contain no additional inference or solver methods.
Exact execution used the retained watchdog with phases `regression` then
`qualification`; it checks failed JSON status, not only process exit code:

```sh
caffeinate -i results/traffic_env/bin/python -u results/matched_posterior_scaling/matched/launch_local.py regression
caffeinate -i results/traffic_env/bin/python -u results/matched_posterior_scaling/matched/launch_local.py qualification
cat results/matched_posterior_scaling/matched/selection.json
results/traffic_env/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

Commands refuse to overwrite attempts; replay requires an isolated copy with the
frozen prerequisites and clean new output. A fresh clone lacks ignored inputs.
Unchanged Apple M4/16GiB, CPU float64/four threads; Torch2.11.0, GPyTorch1.15.2,
linear_operator0.6.1, NumPy1.26.4, SciPy1.17.1. Clean pinned upstream commits:
Fast-Graph-GP268fff9a3724cdd480dce4b2946068d8211b4f30 and authors' experiments
274b7ee393ee516585552f053e9d4a9c6ad2222f. All60 frozen identities/packages verify.
Baseline/final tests:9/9 in both environments, with the same three existing
strict-CG NumericalWarnings per suite; py_compile/diff checks pass.
Changed: existing driver/report only; deleted: one-time diagnostic code;
kept: kernels, training, dependencies and all earlier evidence. Open issue:
no budget satisfies the frozen all-case rule; further research needs planner review.
