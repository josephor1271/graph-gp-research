# Learned-kernel scaling: numerical qualification stopped the study

Neither prescribed common inference budget qualifies. Heat passes, but Poly and official GeneralGRF W4096 fail
raw-gradient accuracy at both 32/16 and 64/24. The work order requires stopping here. No n2500 development
fits,100/200-update diagnostic, n512 sparse-path smoke, final configuration selection or final scaling fit was run.
All 72 planned final fits are explicitly not attempted; no large-N runtime/accuracy, paired speedup, memory or
empirical-slope evidence exists.

Completed: weighted sparse n36 preflight and all 96 n1000/m250 numerical trials, using r9 only. Both regimes share
graph/split/z/noise as specified. H: f=expm_multiply(-.75L,z); M: (f_H+Lz2/2)/sqrt(2), giving covariance
.5exp(-1.5L)+.5(L/2)^2. Both reuse original independent variance.1 observation noise. Training-only normalization,
finite values and disjoint quarter-train/quarter-test IDs are verified. Graph generator and its defaults are
unchanged.

Seeds for graph/data/split/z2/walk/trace are base+n+100000r, bases 731000/741000/751000/761000/771000/781000. New
development directories separate n36/n1000 and r9; no PR8 final data reused. Final sizes 5K/10K/25K/50K with r0/1/2
and deterministic arm rotations remain a 72-row unattempted schedule, not a selected/frozen final run.

Reuse the learned-traffic model_for, existing recurrence/LinearOperator/ExactGP, official sparse sampler/kernel, PR8
controls and dense-test helpers. Heat uses scoped Cheb6; Poly is degree2 filter/squared-degree4 with q(0)=1; General
uses J3/.5 and W1024/4096, sampled once per budget/graph and shared across regimes. Heat has 3 trainables;
Poly/General have 4 including noise. No architecture, dependency or shared-source changes.

Initial beta=2sigmoid(1)=1.462117157, signal SD=softplus(1)=1.313261688, noise variance=.693247181. Poly
a1=-.196611933241482, a2=.133611661347131; General=sigma*[1,-beta/2,beta²/8]. Model RNG0. Second state: noise=.05,
SD=1.5; Poly a1=-.35/a2=.45; General modulation=1.5*[1,-2.15,.9]; Heat beta=1.8. This is fixed-state inference
qualification, not optimizer convergence or predictive-model selection.

CPU float64/four Torch threads/four sampler workers; Cholesky0/eager0, CG/eval-CG1e-4, maxCG1000, no preconditioner,
stochastic log probability. Ordinary ExactGP conditional mean, skip_posterior_variances=True; no pathwise MC.
Budgets 32/16 and 64/24 are probes/Lanczos. Each setting has 12 checks: 2 regimes × 2 states × trace seeds 71/72/73.
Those are numerical trials on one dev dataset, not independent statistical replicates.

Frozen gates: |NLL error|<=.02 per observation; every raw-gradient absolute error<=.02; no sign reversal when
|reference gradient|>.02; true observed-system residual<=2e-4; normalized mean max error<=.002. References use the
exact same finite kernel on a selected 500-vertex block with Cholesky. Residual is measured from the actual GP
mean_cache alpha, not inferred from requested CG tolerance.

|Arm / W|Probes/Lanczos|Pass /12|Max abs NLL error/obs|Max abs raw gradient error|Max true residual|Max mean error|
|---|---|---:|---:|---:|---:|---:|
|Heat|32/16|12/12|0.002801|0.013781|9.03252e-05|0.000216975|
|Heat|64/24|12/12|0.003254|0.010044|9.03252e-05|0.000216975|
|Poly|32/16|4/12|0.008235|0.043312|7.48253e-05|0.000183617|
|Poly|64/24|8/12|0.006042|0.027322|7.48253e-05|0.000183617|
|General/1024|32/16|10/12|0.012094|0.023346|7.69857e-05|0.000183129|
|General/1024|64/24|10/12|0.007407|0.029510|7.69857e-05|0.000183129|
|General/4096|32/16|8/12|0.007449|0.023147|5.84693e-05|0.000106736|
|General/4096|64/24|8/12|0.005063|0.024101|5.84693e-05|0.000106736|

All 96 trials produced finite results: 72 pass/24 reject. Every rejection is a raw-gradient absolute-error violation
only; all NLL, residual, mean and sign gates pass. No common budget is selected. W1024 independently fails both
budgets too, but its rejection alone would not block the study: the required Poly and W4096 configurations also
fail.

At 32/16 Poly passes 2/3 initial-state traces per regime and 0/3 second-state traces. At 64/24 it passes 2/3 in each
regime/state. General failures occur only in the second state: W1024 passes 2/3 there, W4096 passes 1/3 there, in
each regime at each budget. Full 96 rows and 360 labeled signed gradient components are retained locally; failing
traces are not averaged away.

|64/24 example|Regime/state/trace|Raw component|Reference|Estimate|Abs error|
|---|---|---|---:|---:|---:|
|Poly|H/second/71|covar_module.base_kernel.a2[0]|-11.436663|-11.463985|0.027322|
|General/4096|M/second/71|covar_module.raw_modulation_function[0]|2.067362|2.043262|0.024101|
|General/1024|H/second/72|covar_module.raw_modulation_function[2]|0.747184|0.776694|0.029510|

These are absolute raw-coordinate engineering bounds. For example, Poly’s .027322 error is about .24% of its large
reference gradient, yet fails the frozen .02 bound. The evidence does not establish failed learning, a kernel error
or inability to scale. Small NLL/mean error does not certify the required gradient fidelity. Larger budgets are not
assumed to improve every realized trace error; no replacement budgets, seeds, states or relaxed thresholds were
tried.

Tiny checks verify symmetry/spectrum and both same-vector draws against independent dense algebra, full 36-vertex
Poly covariance/likelihood/raw gradients, and selected-row finite General covariance/gradients/posterior/NLPD.
Existing helper keys 303/404 alias the same prescribed tiny draw (actual seed 1671036); they are not independent
walk replicates. Initial Poly/finite-General discrepancy=.141965 relative Frobenius; intended coefficients match,
finite kernels differ. Heat 6 covariance max error 8.25671e-6 at initialization passes its PR8 envelope.

After the stop, a separate local oracle audit formed only 250 selected filter/feature rows for 12 learned-model
kernel/regime/state cells. Independent matrix-power Poly and actual-CSR General expressions reproduce the Cholesky
reference NLL/gradients within 8.89e-16/3.56e-15. This validates the failing comparisons without new stochastic
trials; it is in reference_audit_command.txt, not fully implemented in the tracked driver.

Nine existing tests pass before/after in both environments; existing strict-CG test warnings remain. Sparse
invariant/CSR and upstream deprecation UserWarnings are retained; no experiment NumericalWarning, nonfinite result
or unexpected runtime exception. Qualification deliberately exits nonzero at the final eligibility assertion. The
budget failure is the prescribed stopping condition, not a software-budget or hardware-limit result.

Run from repo root with the pinned traffic environment and the supplied work order at
~/Downloads/CODEX_LEARNED_KERNEL_SCALING_STUDY.md (its hash is checked). Preserve original logs.
Source/environment/work-order changes are rejected; saved data are checked on replay. The driver intentionally
implements only tiny/qualify after this stop; no final worker/scheduler or plotting framework was added.

```sh
results/traffic_env/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
results/traffic_env/bin/python -u experiment_learned_kernel_scaling.py tiny
caffeinate -i results/traffic_env/bin/python -u experiment_learned_kernel_scaling.py qualify  # expected exit 1
git diff --check
```

Base main `6554aaef34704d2a4d49a5012e432de1bd15ce1b` contains merged PR #14. Pins:
core268fff9a3724cdd480dce4b2946068d8211b4f30 / experiments274b7ee393ee516585552f053e9d4a9c6ad2222f; tracked upstream
sources remain clean. Full source/package/work-order hashes, sparse graphs, z/z2/truth/noisy targets, splits,
normalizations, walk caches, oracle audits, all qualification rows and unattempted final schedule stay ignored in
results/learned_kernel_scaling/.

Added: one 181-line preflight driver and this report. Deleted nothing; all existing generators, kernels,
GP/inference helpers, tests, prior experiments, dependencies, upstream, README/AGENTS and read-only toolbox kept. No
final data or performance plots generated, no timing/accuracy projected, no settings chosen from final tests. Open
one focused PR for review, leave unmerged, stop.
