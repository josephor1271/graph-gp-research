# Does a fourth-degree spectral filter earn its training cost?

**Poly4 passed the predeclared screen against Poly2.** Mean relative RMSE gain was 5.26%, with four wins in five dates and a median training-time ratio of 1.63. HeatWide had the lowest RMSE on three dates and reached its beta ceiling on all five.

## Frozen comparison

Five Monday 17:30 PEMS-BAY snapshots use the PR #13 weighted normalized Laplacian (1,016 vertices),
ordered 325 sensors, and identical 250/75 spatial split. The filtered indices were 3, 7, 11, 15 and 19.
Each date has its own training-only population mean and SD. The held-out HDF columns were opened for
scoring only after recording all attempts and choosing each heat fit by successful finite training NLL.
All polynomial parameters were fit from scratch. A checkpoint trained on a different date would be an invalid control.

The normalized filter is `q = 1 + sum_j a_j (T_j(lambda-1) - (-1)^j)`. Both arms use
the full-graph squared filter times signal variance, followed by sensor selection. Poly4 starts with
the existing Poly2 coefficients and zero extra coefficients. The same ExactGP, ScaleKernel, sparse
recurrence, likelihood and four-thread CPU float64 settings apply throughout. Both arms took 1,000
AdamW updates (lr .05, decay .01, default betas/epsilon, seed 0), alternating order by date.
Cholesky cutoff 800, eager cutoff 0 and all fast computations disabled give exact observed-block
training. The common guard rejects full-width graph products. Poly2 has four parameters, Poly4 six.

HeatWide uses one eigendecomposition and exact selected-row spectral blocks. Its three training-only
L-BFGS-B starts use beta 1.462117, 8 and 32, with matching physical amplitude and noise starts.
Log-space bounds are beta [1e-4,64], signal variance [1e-6,1e4], noise variance [1e-4,100].
Settings were maxiter 200, ftol 1e-12, gtol 1e-6 and maxls 30. This stronger search gives an accuracy
baseline. Its dense eigensystem and different optimizer preclude a deployment speed comparison.

## Numerical checks

On a weighted six-vertex graph, the independent spectral polynomial uses explicit powers of
lambda−1. It covers Poly2, nested Poly4 and a sign-changing Poly4. Max absolute errors were
9.86e-16 for covariance, 6.67e-15 for duplicate-ID/multiple-RHS products, 9.65e-16 for transpose,
2.23e-16 for normalized NLL, 8.33e-16 for raw gradients and 1.12e-16 for posterior means.
The zero-mode normalization, symmetry and PSD checks passed (minimum eigenvalue threshold −1e-10).
At beta 1.462117, 8, 32 and 64, matrix exponential vs spectral heat errors were at most
1.73e-15 for covariance, 4.45e-16 for NLL and 7.46e-16 for log-parameter gradients.
At the initial beta, the historical degree-20 heat agreed within 4.72e-16, 2.23e-16 and
5.56e-17 for covariance, NLL and physical gradients. Value tolerances (atol, rtol) were
(1e-10, 1e-9), and gradient tolerances were (1e-9, 1e-8).

## Results

|Date|Poly2 RMSE|Poly4 RMSE|HeatWide RMSE|Poly2 − Poly4 (mph)|Relative gain|NLL/obs, Poly2 / Poly4 / heat|
|---|---|---|---|---|---|---|
|01-23|11.4644|11.0811|10.8166|+0.3833|+3.34%|1.309623 / 1.203656 / 1.198310|
|02-20|8.6854|8.8211|7.2791|-0.1357|-1.56%|1.320439 / 1.294374 / 1.328224|
|03-20|13.5746|12.3081|13.2018|+1.2665|+9.33%|1.333764 / 1.283044 / 1.266329|
|04-17|12.6834|11.6511|11.9221|+1.0323|+8.14%|1.324421 / 1.264536 / 1.238444|
|05-15|13.0229|12.1071|11.4019|+0.9158|+7.03%|1.312933 / 1.269159 / 1.246508|

|Date|Setup s, 2 / 4|Training s, 2 / 4|Prediction s, 2 / 4|Training ratio 4/2|Total ratio 4/2|
|---|---|---|---|---|---|
|01-23|0.0010 / 0.0011|8.330 / 14.549|0.0052 / 0.0088|1.747|1.746|
|02-20|0.0007 / 0.0008|8.371 / 13.585|0.0045 / 0.0082|1.623|1.623|
|03-20|0.0009 / 0.0008|8.291 / 13.685|0.0047 / 0.0087|1.651|1.651|
|04-17|0.0007 / 0.0007|8.295 / 13.537|0.0049 / 0.0084|1.632|1.632|
|05-15|0.0007 / 0.0007|8.317 / 13.436|0.0047 / 0.0080|1.615|1.615|

|Date|Heat NLL/obs, starts 1.462 / 8 / 32|Selected start|Selected beta / signal variance / noise|Bounds reached (all starts)|
|---|---|---|---|---|
|01-23|1.198309684 / 1.198309684 / 1.198309684|heat32|64 / 12.862 / 0.29501|Beta upper at all three|
|02-20|1.406738812 / 1.328223731 / 1.328223731|heat32|64 / 8.0494 / 0.5256|Noise lower at first start, beta upper at other two|
|03-20|1.266329437 / 1.266329437 / 1.266329437|heat32|64 / 10.576 / 0.39704|Beta upper at all three|
|04-17|1.238444145 / 1.238444145 / 1.238444145|heat0|64 / 11.344 / 0.35446|Beta upper at all three|
|05-15|1.246508083 / 1.246508083 / 1.246508083|heat32|64 / 11.059 / 0.36732|Beta upper at all three|

All 10 polynomial fits and 15 heat optimizations completed. Heat starts shown as heat0 in the table
used beta 1.462117. Each heat attempt met the projected-gradient stopping condition (≤1e-6).
Near-tied starts are displayed rounded, and selection used full-precision NLL. February's first
start reached noise 1e-4 and beta 0.9713. All other attempts reached beta 64. No signal bound was hit.
These results leave the sufficient heat-time range unresolved. Further heat work awaits review.

Poly4's relative gains had median +7.03% and range −1.56% to +9.33%. All ten polynomial fits,
mean gain ≥2%, at least four wins and median paired training ratio ≤2 were required. The result
supports advancing Poly4 to a later sparse scalability comparison, including a matched official
higher-cutoff GRF after planner review. HeatWide beat Poly2 on all five dates (mean difference
0.962 mph) and Poly4 on three (0.269 mph average advantage). Lower training NLL for Poly4 than
Poly2 on every date did not ensure better held-out RMSE in February.

Standardized RMSE for Poly2 / Poly4 / HeatWide in date order was .6581/.6361/.6209,
.8281/.8411/.6940, .7139/.6473/.6943, .6984/.6415/.6565 and .7317/.6803/.6407.
Raw final gradient norms ranged .000136–.000237 for Poly2 and .00107–.261 for Poly4.
The fixed endpoint does not certify optimization convergence.

Training timers include AdamW setup and all 1,000 updates, excluding the final gradient audit and
checkpoint I/O. Prediction times use the same cold selected 325-sensor covariance and posterior helper
(including marginal variance calculation). Setup + training + prediction ratios exclude shared input
loading and the heat eigendecomposition. The graph has 3,360 Laplacian nonzeros. The covariance
recurrence uses 4 or 8 sparse products per action, costing O(2d × nnz(L) × RHS width); coefficient
construction costs O(d²). Training uses a 250-sensor Cholesky. These wall times do not estimate
large-graph solver costs. Five related snapshots share a graph and spatial split. The screen is
descriptive evidence for spatial interpolation; independent-graph generalization remains open.
PR #17 remains the mean-only scaling benchmark.

Both environments passed all nine existing tests before and after; each retained the three known
strict-CG warnings. The completed experiment emitted no warnings. Reloading sparse checkpoints for
the post-run audit emitted Torch's sparse-invariant-check UserWarning. All 15 RMSEs independently
recomputed from the saved means, all 10 checkpoint hashes matched, and all 546 pre-existing traffic
result files and frozen inputs passed their final hash checks. The completed worker ran about 115 s
with a sampled process-group peak of 622 MiB. No limit or scientific fit failed.

## Reproduction and evidence

Base `075cc6dace46ab33c78ab25e603ef678eeb1ac0e` includes merged PR #19.
Start with the PR #13 data and manifest under `results/traffic/learned_polynomial_fresh/`,
and the pinned local upstream checkouts. Fast-Graph-GP was `268fff9a3724cdd480dce4b2946068d8211b4f30`
and experiments was `274b7ee393ee516585552f053e9d4a9c6ad2222f`, both clean.
The pinned archive is `data/traffic/PEMS.zip` in the latter checkout.
PR #13 data SHA-256 starts `d14c1e5a45a09498` and the ordered sensor-map digest starts `ad0b4737ddcbcee8`.

On the Apple M4, macOS 26.6.2, we used Torch 2.11.0, GPyTorch 1.15.2, linear_operator 0.6.1,
SciPy 1.17.1 and NumPy 1.26.4. Torch used four intra-op and ten inter-op threads, OpenMP four,
with OMP/MKL thread environment variables unset. From the repository root, use a fresh output
directory. The driver refuses a second schedule in an existing attempts file.

```sh
mkdir -p results/spectral_expressivity
results/traffic_env/bin/python -u experiment_spectral_expressivity.py
.venv/bin/python -m unittest discover -s tests -v
results/traffic_env/bin/python -m unittest discover -s tests -v
results/traffic_env/bin/python -m py_compile experiment_spectral_expressivity.py spectral_checks.py
git diff --check
```

On the experiment Mac the exact bounded launch was `bash results/spectral_expressivity/watch.sh`.
It ran the driver under caffeinate, sampled process-group RSS once per second and enforced the
8 GiB and 60-minute ceilings. The driver set a 30-minute alarm per fit.
The ignored output directory holds the work order, frozen source/input hashes, split and training
targets, 10 checkpoints with 1,000 losses each, attempt records, 75 posterior means per scored fit,
scores.csv, test logs and the resource log.

Two preflight events preceded all traffic fits. First the independent heat gradient check reused a
released autograd graph. Giving each objective its own noise tensor allowed the same comparison to
run. Second, opening the local HDF in pandas' default writable mode during inspection changed byte
20 of its header. The provenance gate stopped. With planner approval the frozen schedule used the
byte-identical original recovered from the pinned zip (SHA-256
`eba3ef97ab496a2fca35e69f2901dca961a16c146106ae8bd70b3375ed63df9e`). Both logs and the
inspected copy were preserved locally. The experiment opens the store explicitly read-only.

Three new tracked files comprise the experiment (166 Python lines), independent checks (74), and
this report. Existing production code, historical experiments, dependencies and tests are unchanged.
