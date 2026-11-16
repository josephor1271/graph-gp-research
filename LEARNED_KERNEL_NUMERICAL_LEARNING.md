# Same-kernel numerical learning qualification

Does CG/SLQ learn comparable models to 100-update Cholesky training of the **same finite kernel**?
**Selected:32 probes/16 Lanczos, CG/evalCG1e-4/maxCG1000, no preconditioner.**
All arms pass both gates at all three budgets; W1024 is eligible for the next
walk-budget comparison. This supports comparable fixed-budget learning on these data;
Cholesky is faster at this small size. W4096 remains the required General arm.
PR15's 72/96 old-rule passes and 24 rejections remain unchanged; this study uses
the newly authorized scale-aware rule on different data, not a retrospective pass.

## Frozen design

Base main `986389eeaab0fad1319daf87687bba4f532c5fb8` (merged PR15). Two independent
n1000 replicates r10/11, 250 training/250 validation vertices, train-only normalization,
raw noise variance .1. Weighted road-like graph, L=I−D^(-1/2)AD^(-1/2).
Latent covariances H:exp(−1.5L), M:.5exp(−1.5L)+.5(L/2)^2; shared noise/splits.
The helper's directory name `final/` is historical: **all these data are development**.
Graph/data/split/z2/walk seeds respectively equal
`731000/741000/751000/761000/771000 + 1000 + 100000*r`.
No r9, final r0/1/2, external test sensors, or parameter search.

Heat is degree6/beta∈(0,2); Poly squares its degree2 filter through the existing operator;
General uses J3/halt.5/W1024 or4096. Cached walks are shared across regimes/states/budgets
and controls for each (r,W). W1024/4096 are distinct finite kernels; Cholesky retains
the actual Cheb6/squared polynomial/sampled-feature covariance in each arm.
CPU float64, 4 Torch threads/workers, model RNG0, AdamW .05/default decay .01,
100 updates from identical raw starts; no warm starts or endpoint optimization.
Inference menu 32/16,64/24,128/24 (probes/Lanczos), CG/evalCG1e-4/maxCG1000,
eager0/no preconditioning. Trace seeds81/82/83 diagnose fixed states; learning
seeds91/92 are set once per fit, with new probes each step; exact fits consume no RNG.
The two numerical states are the original start and unchanged `second_state`:
noise.05, Heat(beta1.8,sigma1.5), Poly(sigma1.5,a1−.35,a2.45), General[1.5,−3.225,1.35].

## Numerical gate

Every trial requires finite values, no NumericalWarning, |NLL error|≤.02/obs,
true alpha residual≤2e-4, normalized mean max error≤.002, and no gradient sign
reversal when |g|>.02. Within each dataset/regime/state/arm/budget/component,
the three-trace signed mean error must be ≤.02+.05|g|; **each** error≤.05+.10|g|.
Smean/Swide are maxima of error/bound (≤1 passes); no pooling components or datasets.
Each row has8 fixed cells/24 draws; “old” counts draws passing the old .02 absolute
gradient bound. All288 draws were finite and safe.

| Arm | Probes/Lanczos | Cells pass | max abs ΔNLL | max abs Δg | Smean | Swide | old /24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Heat |32/16|8/8|.008821|.011710|.2175|.1405|24|
| Heat |64/24|8/8|.002783|.003071|.0201|.0368|24|
| Heat |128/24|8/8|.003145|.005101|.0517|.0612|24|
| Poly |32/16|8/8|.006390|.101579|.5058|.3882|8|
| Poly |64/24|8/8|.003557|.041751|.0641|.1084|16|
| Poly |128/24|8/8|.004358|.045309|.1240|.1550|20|
| General1024 |32/16|8/8|.008672|.058176|.8914|.8869|8|
| General1024 |64/24|8/8|.010086|.026720|.4318|.5191|18|
| General1024 |128/24|8/8|.007367|.016800|.2230|.3011|24|
| General4096 |32/16|8/8|.007570|.047192|.4643|.4553|18|
| General4096 |64/24|8/8|.005543|.024214|.0774|.1397|20|
| General4096 |128/24|8/8|.003063|.024760|.0953|.1375|22|

Worst residual9.93e-5/mean error2.63e-4. All32 state/dataset/arm exact-MLL checks
agree with selected-block Cholesky within NLL8.88e-16/raw-gradient3.55e-15;
RNG unchanged. Prior independent tiny-kernel checks remain in PR15.
Only selected250/500 blocks are densified; no full n1000 kernel/identity.
Eleven individual draws miss the tighter **mean** bound (General1024:9 at32,
1 at64; General4096:1 at32), but their three-trace means and wider individual
guards pass. All sign checks pass. Three traces do not establish unbiasedness.

## Direct learning comparison

Completed16 exact+96 iterative fits, 11,200 updates; no failed/blocked fit or NumericalWarning.
Qualification per fit:100 finite updates, safe residual/mean,
exact endpoint NLL≤paired exact control+.02 and actual iterative RMSE≤1.02×control.
Exact controls are fixed-budget optimizations, not certified optima.
Each row combines r10/11×two trace seeds (4 fits), retaining each fit's paired control.
Columns show **mean/max** signed exact-own-kernel NLL and actual raw-RMSE percentage gaps,
and mean training seconds **exact/iterative**.

| Arm | Regime | Probes | Pass | ΔNLL/obs | ΔRMSE % | Train s E/I |
|---|---|---:|---:|---:|---:|---:|
| Heat | H |32|4/4|+.000014/+.000029|+.001/+.002|1.179/2.585|
| Heat | H |64|4/4|+.000031/+.000044|+.001/+.003|1.179/4.561|
| Heat | H |128|4/4|+.000018/+.000039|+.001/+.003|1.179/6.838|
| Heat | M |32|4/4|+.000017/+.000036|−.002/+.008|1.265/2.580|
| Heat | M |64|4/4|+.000032/+.000050|+.001/+.013|1.265/4.527|
| Heat | M |128|4/4|+.000012/+.000026|+.003/+.014|1.265/6.831|
| Poly | H |32|4/4|+.000531/+.001609|−.016/+.022|.952/1.928|
| Poly | H |64|4/4|+.000144/+.000346|+.052/+.105|.952/3.332|
| Poly | H |128|4/4|+.000060/+.000122|−.013/−.002|.952/5.171|
| Poly | M |32|4/4|+.000034/+.000067|+.011/+.021|.922/1.847|
| Poly | M |64|4/4|+.000052/+.000138|+.025/+.041|.922/3.275|
| Poly | M |128|4/4|+.000024/+.000065|+.017/+.034|.922/5.081|
| General1024 | H |32|4/4|+.000374/+.001317|+.012/+.037|1.525/3.758|
| General1024 | H |64|4/4|+.000079/+.000206|+.006/+.013|1.525/5.724|
| General1024 | H |128|4/4|+.000074/+.000167|−.002/+.001|1.525/8.496|
| General1024 | M |32|4/4|+.000011/+.000128|+.020/+.034|1.528/3.683|
| General1024 | M |64|4/4|+.000034/+.000129|+.022/+.043|1.528/5.555|
| General1024 | M |128|4/4|+.000017/+.000047|−.000/+.043|1.528/8.341|
| General4096 | H |32|4/4|+.000258/+.000510|+.016/+.046|1.493/3.683|
| General4096 | H |64|4/4|+.000179/+.000456|+.012/+.026|1.493/6.034|
| General4096 | H |128|4/4|+.000039/+.000057|+.002/+.011|1.493/8.581|
| General4096 | M |32|4/4|+.000334/+.000849|−.060/+.047|1.569/3.678|
| General4096 | M |64|4/4|+.000164/+.000372|−.010/+.042|1.569/5.641|
| General4096 | M |128|4/4|+.000046/+.000062|−.017/+.034|1.569/8.332|

Exact control endpoints (r10/r11) supply the absolute scale of those paired gaps:

| Arm | Regime | NLL/obs r10/r11 | Raw RMSE r10/r11 |
|---|---|---:|---:|
| Heat |H|1.392743/1.390536|.584035/.552939|
| Heat |M|1.414978/1.416930|.599656/.575364|
| Poly |H|1.393167/1.390139|.584401/.553842|
| Poly |M|1.407939/1.415433|.605722/.571278|
| General1024 |H|1.393404/1.396659|.585189/.554705|
| General1024 |M|1.413979/1.414184|.599225/.575164|
| General4096 |H|1.392043/1.394091|.584260/.553888|
| General4096 |M|1.409597/1.412119|.602209/.569936|

Worst fitted residual9.05e-6/mean discrepancy1.33e-5; actual/own-exact RMSE differs
by at most1.36e-7: learning dominates prediction-solve error. Largest two-trace spread
within a dataset:.001512 NLL/obs/.000950 raw RMSE. Some gaps favor iterative training;
larger budgets do not improve every endpoint. No setting is rejected;32/16 is chosen
by declared budget order, not timing rank.
Training totals501.111481s:20.863992 exact (16 fits),480.247489 iterative (96).
Timers exclude construction/audits/I/O, with comparable logging. Cold walks paid once
per(r,W):r10 .525/1.868s,r11 .480/1.863s (W1024/4096), separate from training.

All initial physical noise variances=.693247181; Heat/Poly sigma=1.313261688,
Heat beta=1.462117157; Poly(a1,a2)=(−.196611933,.133611661);
General modulation=(1.313261688,−.960071223,.350934152).
Across endpoints: Heat beta1.644–1.812/sigma.810–1.655/noise.319–.826;
Poly a1−.502–−.116/a2.097–.524/sigma1.014–1.651/noise.503–.835;
General1024 modulation ranges [.869,1.525]/[−1.430,−.953]/[.092,.465],noise.333–.870;
General4096 [.980,1.568]/[−1.540,−1.098]/[.200,.568],noise.444–.694.
Max absolute exact endpoint raw gradient: Heat.00407, Poly.10686,
General1024 .06429, General4096 .04604: **not stationarity certificates**.
Local `physical_responses.json` retains every initial/final parameter and response
at lambda0,.5,1,1.5,2. General's intended polynomial response is not its actual
finite-feature spectral covariance. Neither two datasets nor repeated traces
establish significance, uncertainty accuracy or generalization to larger graphs.

## Artifacts, failures and reproduction

Ignored `results/learned_kernel_numerical_learning/` retains protocol/data/walk hashes,
the112-fit schedule, signed gradients, all losses/raw parameters/endpoint gradients,
warnings and model/RNG checkpoints with hashes. PR15 remains separate.
One pretraining launch failed provenance equality because regenerated elapsed times
changed. Only timing metadata was excluded; original source/manifest/log/migration
hashes remain local. No numerical row, seed or scientific setting was replaced.
An auxiliary summary export rejected a NumPy integer; conversion to Python int
fixed serialization only. Its failed log is retained.
Non-numerical PyTorch sparse/TypedStorage deprecation warnings are preserved.
The **separate local** `endpoint_audit_command.txt` independently constructs selected
Heat6/power-polynomial/actual-CSR feature rows at all112 endpoints; MVN likelihood,
raw gradients/exact RMSE agree within4.44e-16/1.67e-16/1.11e-16.
This full audit/physical-response export is **not fully implemented in the tracked
driver**; its built-in checks compare the selected-block oracle and ordinary MLL.

```sh
PY=results/traffic_env/bin/python
ORDER=/path/to/CODEX_LEARNED_KERNEL_NUMERICAL_LEARNING_STUDY.md
$PY experiment_learned_kernel_numerical_learning.py numerical "$ORDER"
$PY experiment_learned_kernel_numerical_learning.py learn "$ORDER"
$PY experiment_learned_kernel_numerical_learning.py select "$ORDER"
$PY -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

Use the existing pinned upstream checkouts/environment (traffic reproduction
instructions in `README.md`). Core pin
`268fff9a3724cdd480dce4b2946068d8211b4f30`, experiments pin
`274b7ee393ee516585552f053e9d4a9c6ad2222f`; both tracked trees verified clean.
Apple M4/16GiB, macOS26.6.2, Torch2.11.0, GPyTorch1.15.2,
linear_operator0.6.1, NumPy1.26.4, SciPy1.17.1. Dataset fingerprints:
r10 `fef4c78afc4f4687d10fa7bee46cef5f7ea81caa6e2c9c174828cce836b6bbf5`;
r11 `02490951e22670009a5e18f6150c9bf73d70ac39ff4ddd18382f631a1589cd68`.
Full source/package/work-order/input hashes are in the protocol; resume verifies
provenance/checkpoints and preserves completed/failed rows.
Protocol SHA256:`c876d946cb09f4d06a061a2e7bd594f16d30a9ef993d80d76855205106ab9998`;
selection SHA256:`b781f585c498a2721d9c06165657c9bf4d742f53d8e56459fcfed0c7dde974fc`.

Baseline and final tests:9/9 in each environment, with existing strict-CG test
warnings retained. `git diff --check` passes. All112 JSON rows agree with embedded
checkpoint records. Resume verifies128 cases without retraining or changing their
rows/checkpoints. All30 PR15 artifact hashes/file set are unchanged.
Changed: one 220-line driver and this report. Deleted:none. Kept: all prior
code, kernels, optimizer, inference machinery, dependencies and PR15 evidence.
No large scaling was run. Review before any further experiment.
