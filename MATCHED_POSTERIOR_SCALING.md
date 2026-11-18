# Matched posterior uncertainty through 50K vertices

**Completed all144 cold predictions on48 existing fits.** Both methods pass all72
of their audited latent/observation variance targets. Poly's median prediction
phase sum is lower in all24 paired fits (TP/TG=.385–.768); the advantage narrows
at50K. Predictive differences are small and sometimes mixed; no significance,
statistical equivalence, universal scaling law or real-traffic claim follows.

## Frozen protocol and qualification

The planner-authorized menu extension was frozen before execution:2048, then4096
only if needed. At2048, **Poly24/24 and General24/24** prescribed n1000
state/stream combinations pass both latent L2<=10% and observation L2<=5% targets.
Worst latent/observation errors: Poly3.389%/2.535%, General3.507%/3.205%.
All48 new numerical checks pass (same-draw Cholesky variance L2<=.002 and true
mean/path residual<=2e-4). **4096 was not evaluated.** Prior failed General
qualifications at512(18/24) and1024(23/24), native diagnostics, thresholds,
manifests and source snapshots remain intact; no stress state was dropped.

New development qualification uses both methods, r10/11, initial H/stress H/
fitted H/fitted M and seeds521/522/523. Existing outputs lack RNG/running-moment
checkpoints, so the unchanged seeded streams replay earlier perturbations and
continue to2048; only the new endpoint is evaluated. No native investigation,
training, kernel change, solver sweep or new dependency. The active driver remains
**358 lines: four existing lines changed** for output/budget names and streams.

The final schedule was frozen immediately after selection: n5K/10K/25K/50K,
r0/1/2,H/M,Poly/General W1024,three fresh processes per endpoint with seeds521/522/523,
2048 paths,width32,CPU float64,four Torch threads,CG1e-4/max1000/no preconditioner.
Actual feature/root and lazy A/C preparation is once per invocation for both arms.
The General arm is **official GeneralGRF features with a batched float64 posterior
inference adapter**, not unmodified native prediction. Adapter noise stays float64.
Poly reuses its quadratic Chebyshev root (degree-four covariance); General uses
the actual finite, potentially asymmetric Phi, K=Phi Phi^T. No SLQ in prediction.

PR17's unchanged weighted normalized road-like graphs have train/test quarters,
training-only normalization, H covariance exp(−1.5L), M .5exp(−1.5L)+.5(L/2)^2,
raw noise variance.1. All48 checkpoints, data/feature fingerprints, original walk
caches (W1024,J3,halt.5), input hashes and fitted parameters verify. No refitting
or new walks. Model/cache state resets per process; disk caches are not flushed.
Arm order rotates by(size_index+r+regime+stream_index)%2 in increasing size.

## Prediction and reconstructed costs

Each fitted endpoint's primary time is the median of its three cold stream phase
sums. Table centers are medians across r0/1/2; brackets span those three endpoint
medians. Paired TP/TG is computed per r, then summarized, not divided group centers.
H/M share graph/noise/split; MC streams are not independent data replicates.
All individual timings, phase costs and ranges are retained in the local CSVs.

|N/regime|Poly prediction s [range]|General prediction s [range]|Paired TP/TG [range]|Standalone s P/G|Amortized General s|
|---|---:|---:|---:|---:|---:|
|5K/H|3.755[3.139,3.867]|9.747[6.571,9.750]|.397[.385,.478]|8.917/24.417|22.049|
|5K/M|4.299[4.177,4.476]|8.985[8.781,9.012]|.478[.464,.510]|9.499/23.745|21.313|
|10K/H|4.962[4.959,5.414]|9.748[9.707,9.871]|.511[.509,.549]|16.725/41.023|35.951|
|10K/M|5.005[5.001,5.046]|9.664[9.599,9.686]|.521[.518,.521]|16.661/41.205|35.899|
|25K/H|13.722[13.716,25.897]|25.869[25.013,48.590]|.533[.530,.548]|45.417/109.741|95.890|
|25K/M|13.741[13.520,28.644]|25.985[24.746,56.415]|.529[.508,.546]|45.507/109.556|95.705|
|50K/H|50.111[47.786,50.216]|65.248[65.233,65.728]|.764[.733,.768]|110.156/198.793|169.015|
|50K/M|47.997[47.992,48.196]|65.770[65.115,66.113]|.730[.729,.737]|107.824/198.938|169.134|

All24 pairs meet the audited accuracy targets in all three streams. Their
standalone cost ratios span.364–.558; amortized ratios.403–.656. Standalone=
PR17 setup+original fresh walk+training+new prediction phases, **replacing** old
mean-only time. This sums historical and new measurements on the same hardware/
software, not a freshly timed end-to-end execution. Each General standalone fit
is charged its original walk fully; amortized excludes that walk. Poly has zero
walk cost, so its amortized total equals standalone. Shared graph/data excluded.

Prediction phases measure feature/root/block preparation, analytic mean,
random/root/CG/correction batches and streaming moments. Residual audits,
selected-column references, loading/hashing, guards, scoring and I/O are excluded.
Initial generator/running-moment-vector setup and Python orchestration are also
outside these phase timers; inclusive worker wall is retained separately. Audit
products interleave batches and can warm caches. Thus phase sums are not complete
API wall latency. Raw stream timings span2.916–58.547s Poly and6.565–108.432s
General; substantial timing variation is retained, including the25K outliers.
No slope is fitted or speed inferred for unmeasured graphs/settings.

## Variance fidelity and predictive scores

All48 selected-column audits actually executed under guards:8 mandatory fresh
r0/H mean+32-path smokes plus40 other endpoint audits. Smoke rejects an existing
audit cache. Each endpoint's reference is reused only for its three predictions.
The fixed256 held-out positions (seed891000+n+100000r) are shared across methods/
regimes and frozen before outcomes. Local basis products, not full covariance or
Poly diagonal/exact-entry fallback, supply the actual finite-kernel reference.

Positive residual brackets use upper=d−c^T x−x^T r and lower=upper−||r||²/nu.
All selected widths pass; maximum relative width **1.31609e-10**, maximum true
reference-column residual9.99221e-6. Aggregate comparisons use the prescribed
conservative triangle bounds against these intervals, on audited nodes only;
float arithmetic is not a formal certified enclosure or whole-test guarantee.

|Maximum across72 final streams per arm|Poly|General|Target|
|---|---:|---:|---:|
|Latent relative L2 bound|3.4124%|3.4845%|10%|
|Observation relative L2 bound|2.1620%|1.9586%|5%|
|Worst latent node / per-fit p95 error|14.07%/6.85%|13.68%/7.23%|no pointwise gate|
|Worst observation node / per-fit p95 error|7.29%/4.53%|7.54%/3.82%|no pointwise gate|

Maximum true mean/path residuals are8.94885e-6/1.04103e-5; means match saved PR17
means bit-for-bit (these saved references are iterative, not dense large solves).
No NumericalWarning, invalid variance, jitter/clipping or numerical failure.
Gaussian scoring uses the analytic-form iterative mean and sample latent variance
plus observation noise once; no finite-sample mean for RMSE. Below: means across
three independent graph/data replicates of endpoint medians over streams, P/G.
NLPD is **marginal observation** NLPD; variances scale by training scale² and
NLPD_raw=NLPD_normalized+log(scale). Normalized scores and mean95% widths
(3.919928×mean SD), all raw vectors and stream values remain in CSV/NPZ.

|N/regime|Raw RMSE P/G|Raw NLPD P/G|95% coverage % P/G|Mean observation SD P/G|
|---|---:|---:|---:|---:|
|5K/H|.615702/.615558|.932777/.937338|94.37/94.45|.6052/.6093|
|5K/M|.644206/.644363|.980097/.981975|94.29/94.40|.6322/.6333|
|10K/H|.608023/.609091|.921519/.924676|94.47/94.35|.6014/.6049|
|10K/M|.622117/.622832|.944503/.945826|94.97/95.04|.6249/.6249|
|25K/H|.608565/.609663|.921373/.925308|94.91/94.84|.6071/.6104|
|25K/M|.634939/.635190|.964681/.965289|94.93/94.88|.6344/.6345|
|50K/H|.604998/.605613|.915597/.918347|94.89/94.95|.6034/.6078|
|50K/M|.634946/.634999|.964736/.964967|94.90/94.90|.6323/.6323|

Poly's mean NLPD is lower in every size/regime group, but some individual large-M
pairs favor General. General has slightly lower mean5K/H RMSE. Paired raw RMSE
P−G ranges−.001802 to+.001015; NLPD differences−.006721 to+.000258; coverage
differences−.36 to+.40 percentage points. These descriptive differences do not
establish significance or noninferiority. Own-kernel numerical fidelity does not
make the two posterior variances equal or establish population calibration,
hyperparameter uncertainty, joint predictive density or transfer to real graphs.

## Memory, resources, preservation and reproduction

One-second sampled process-group peak RSS ranges across18 primary workers/arm/size:

|N|Poly MiB|General MiB|
|---|---:|---:|
|5K|396.5–510.9|456.7–518.5|
|10K|492.9–549.2|497.8–560.5|
|25K|626.0–767.4|655.1–813.7|
|50K|633.9–659.5|685.7–825.2|

These fresh-process measurements include imports, checkpoint/data loading/hash
verification, inference, scoring and output; dense references run separately.
Sampling may miss peaks/double-count shared pages; nonmonotonic values do not
identify exact tensor memory or an asymptotic memory law. Final worker wall sums
4581.370s (max121.018s); primary phase sums3545.722s; reference audits152.492s.
All192 final jobs completed; no failed/skipped final case or resource violation
(1800s/case,4h total,12GiB sampled RSS). Safe MC misses would have continued;
actual final misses are **0/144**. One local preparer import-path error was fixed
before any case ran; its logs remain, and no experiment case was retried.

All **3,805 prior result files and their inventoried filename set** are unchanged.
Qualification and final manifests, source/package/input hashes, old failures,
selection,192 outcomes,144 prediction rows,48 endpoint medians,24 paired ratios,
48 audit rows, warnings, resources and vectors remain local/ignored under
`results/matched_posterior_scaling/extended/`. Two figures (`runtime.png`,
`uncertainty_cost.png`) reproduce from the measured CSVs; shaded runtime ranges
are data-replicate ranges, not confidence intervals. Numerical logic remains in
the tracked driver; local table/plot/watchdog scripts only handle presentation,
metadata and resources. Reproduction needs the retained ignored prerequisites;
fresh clones alone lack checkpoints/walks. Commands refuse to overwrite attempts.

```sh
results/traffic_env/bin/python results/matched_posterior_scaling/extended/freeze_local.py
caffeinate -i results/traffic_env/bin/python -u results/matched_posterior_scaling/extended/launch_local.py qualification2048
results/traffic_env/bin/python results/matched_posterior_scaling/extended/freeze_final_local.py
caffeinate -i results/traffic_env/bin/python -u results/matched_posterior_scaling/extended/launch_local.py final
results/traffic_env/bin/python results/matched_posterior_scaling/extended/tables_local.py
results/traffic_env/bin/python results/matched_posterior_scaling/extended/plots_local.py
```

Same Apple M4/16GiB/macOS26.6.2; Torch2.11.0,GPyTorch1.15.2,linear_operator0.6.1,
NumPy1.26.4,SciPy1.17.1. Clean Fast-Graph-GP pin268fff9a3724cdd480dce4b2946068d8211b4f30;
authors' experiments274b7ee393ee516585552f053e9d4a9c6ad2222f. Base dceed47,
main31fd991; all original identities and newly required Poly states verify.
Existing unittest suites pass9/9 before/after in both environments (same three
pre-existing strict-CG NumericalWarnings per suite); py_compile/diff checks pass.
Changed: four driver lines and this report; deleted:nothing; kept: kernels,
training, inference settings, dependencies and all earlier evidence. Stop for
planner review; no further budget, dataset, solver or scaling experiment.
