# Where does the PEMS-BAY heat likelihood prefer to stop?

**Three dates favor a finite diffusion time materially above 64. Increasing the range worsened held-out RMSE on all five dates.**
At the frozen threshold of 0.001 NLL per training sensor, February, March and April have interior preferred scales:
both refinements agree, the profiled likelihood deteriorates toward infinity, and signal and noise bounds are inactive at the winners.
January and May have no material further gain over 64. On every date the optimized infinite-time reference loses materially
to the best finite fit. None of these profiles supports interpreting the winning heat fit as a near-nullspace model.

## Graph and likelihood

The 1,016-vertex symmetric normalized Laplacian has one connected component, no isolated vertices and one zero mode.
Its leading eigenvalue in ascending order was −1.239e−16 and its largest was 1.999576.
The norm of the nullspace residual was 3.19e−15; the basis passed the orthonormality check.
Only the verified structural zero was replaced by exactly zero. For this connected weighted graph the null vector is
proportional to the square root of weighted degree. The infinite-time unit kernel projects onto that vector (rank one).
At finite β its operator-norm distance to the projector is exp(−β × gap), where the measured gap is 0.000126091900594.
The grid extended to 32/gap = 253783.15, giving a unit-kernel tail of 1.27e−14. Infinity used the projector directly.

We reused the PR #20 five Monday snapshots, fixed 250/75 spatial split, training-only standardization and zero mean.
For each finite β and infinity we cached the selected unit covariance, then optimized log signal variance s and log
noise variance ν separately per date. The objective was exact observed-block Cholesky NLL divided by 250.
Starts were the date’s training-selected PR #20 variances and the shared physical initialization.
Bounds were s ∈ [1e−6,1e4] and ν ∈ [1e−4,100]. L-BFGS-B used autograd, maxiter 200, ftol 1e−12, gtol 1e−6 and maxls 30.
Two log-β/s/ν refinements started at the best finite grid point and the prior anchor. The grid fit remained eligible.
Selections used the lowest successful training NLL, preferring earlier candidates on exact ties. These are local minima.

## Training selections and predictive consequences

Positive anchor − finite and limit − finite NLL differences favor the finite selection.
RMSE is in mph except for the standardized column. Held-out labels were read after saving all selections.
|Date|Best finite β|s / ν|Anchor NLL|Anchor − finite NLL|Limit − finite NLL|Refined β from grid / anchor|
|---|---|---|---|---|---|---|
|01-23|65.136|12.9900 / 0.2963|1.1983097|+0.0000098|+0.2206386|65.136 / 65.136|
|02-20|577.281|38.2423 / 0.6273|1.3282237|+0.0219383|+0.1126531|577.283 / 577.281|
|03-20|209.442|24.0525 / 0.4587|1.2663294|+0.0130001|+0.1656093|209.443 / 209.442|
|04-17|128.077|18.0213 / 0.3903|1.2384441|+0.0081287|+0.1886231|128.077 / 128.077|
|05-15|70.507|11.6348 / 0.3747|1.2465081|+0.0001755|+0.1726059|70.507 / 70.507|

|Date|β64|Finite|Limit|Poly2|Poly4|Finite − β64|Limit − finite|Finite RMSE / training SD|
|---|---|---|---|---|---|---|---|---|
|01-23|10.8166|10.8330|18.0150|11.4644|11.0811|+0.0164|+7.1820|0.62187|
|02-20|7.2791|7.4914|9.0031|8.6854|8.8211|+0.2123|+1.5117|0.71429|
|03-20|13.2018|13.8811|19.7845|13.5746|12.3081|+0.6793|+5.9034|0.72999|
|04-17|11.9221|12.3113|19.1174|12.6834|11.6511|+0.3892|+6.8061|0.67789|
|05-15|11.4019|11.4208|19.4893|13.0229|12.1071|+0.0189|+8.0685|0.64173|

## Full fixed-time profiles

Cells give training NLL per sensor / fitted signal variance / fitted noise variance, rounded after selection.

|β|01-23|02-20|03-20|04-17|05-15|
|---|---|---|---|---|---|
|0.0001|1.41894 / 0.8502 / 0.1499|1.41894 / 0.7651 / 0.235|1.41894 / 0.8117 / 0.1883|1.41894 / 0.8272 / 0.1729|1.41894 / 0.8222 / 0.1778|
|1.462117|1.39592 / 1.611 / 0.3486|1.41146 / 2.498 / 0.07672|1.39687 / 1.598 / 0.3552|1.39793 / 1.575 / 0.3652|1.39809 / 1.574 / 0.3657|
|2|1.38673 / 1.814 / 0.371|1.41598 / 1.215 / 0.6022|1.38803 / 1.807 / 0.3753|1.38949 / 1.774 / 0.3873|1.38888 / 1.783 / 0.3836|
|8|1.30961 / 3.754 / 0.3039|1.39686 / 2.517 / 0.5708|1.32556 / 3.804 / 0.3203|1.32414 / 3.651 / 0.3327|1.31975 / 3.671 / 0.3249|
|32|1.21518 / 8.495 / 0.2677|1.34994 / 5.697 / 0.5154|1.27823 / 7.783 / 0.3508|1.25553 / 7.982 / 0.3204|1.25680 / 7.797 / 0.3267|
|64|1.19831 / 12.86 / 0.295|1.32822 / 8.049 / 0.5256|1.26633 / 10.58 / 0.397|1.23844 / 11.34 / 0.3545|1.24651 / 11.06 / 0.3673|
|128|1.20472 / 17.4 / 0.3686|1.31426 / 12.54 / 0.5433|1.25681 / 15.75 / 0.4379|1.23032 / 18.01 / 0.3902|1.25093 / 15.84 / 0.4303|
|256|1.20579 / 29.53 / 0.4156|1.31050 / 19.16 / 0.589|1.25427 / 30.1 / 0.4666|1.24066 / 32.88 / 0.4415|1.26297 / 27.81 / 0.4856|
|512|1.22502 / 77.88 / 0.4533|1.30646 / 33.66 / 0.6231|1.28012 / 84.9 / 0.5083|1.27053 / 72.71 / 0.5106|1.29414 / 63.1 / 0.5524|
|1024|1.28716 / 373.7 / 0.5149|1.31261 / 82.54 / 0.6479|1.34468 / 174.9 / 0.6434|1.32266 / 171.1 / 0.6142|1.34369 / 95.15 / 0.6842|
|2048|1.34899 / 119.2 / 0.7449|1.33906 / 164.6 / 0.7147|1.38352 / 98.4 / 0.8105|1.36517 / 122.6 / 0.7696|1.37253 / 100.5 / 0.7909|
|4096|1.37585 / 724.8 / 0.7696|1.37041 / 849.5 / 0.7545|1.40792 / 40.08 / 0.9251|1.39376 / 643.7 / 0.804|1.40115 / 632.2 / 0.8174|
|7930.724|1.39910 / 185.1 / 0.8994|1.38847 / 258.1 / 0.8724|1.41440 / 69.55 / 0.9486|1.40925 / 124.9 / 0.927|1.41303 / 73.78 / 0.9448|
|31722.89|1.41894 / 0.0001617 / 1|1.41894 / 4.169e-06 / 1|1.41894 / 0.000349 / 1|1.41894 / 9.33e-06 / 1|1.41894 / 1.464e-05 / 1|
|126891.6|1.41894 / 1e-06 / 1|1.41894 / 7.877e-05 / 1|1.41894 / 0.000215 / 1|1.41894 / 5.13e-05 / 1|1.41894 / 1e-06 / 1|
|253783.2|1.41894 / 1e-06 / 1|1.41894 / 7.877e-05 / 1|1.41894 / 0.000215 / 1|1.41894 / 5.13e-05 / 1|1.41894 / 1e-06 / 1|
|inf|1.41894 / 1e-06 / 1|1.41894 / 7.877e-05 / 1|1.41894 / 0.000215 / 1|1.41894 / 5.13e-05 / 1|1.41894 / 1e-06 / 1|

At infinity the signal variances were tiny and the noise variances were near one in standardized units. The fitted
means were consequently close to the date-specific training averages. The lower signal bound was reached in six
attempts, on Jan 23 and May 15 at 16/gap, 32/gap and infinity, with no other bound hits. All 170 fixed-time attempts
and 10 refinements met the projected-gradient stopping criterion (largest reported norm 9.89e−7). No pair of starts
differed by more than 0.001 NLL/obs. No optimization failed. Reprofiling at 64 changed NLL by at most 6.89e−15.

Mean RMSE was 10.9243 at the historical anchor, 11.1875 at the best training-selected finite fit, and 17.0819 at
infinity. The finite fit beat historical Poly2 on four dates and Poly4 on three. Mean Poly2 and Poly4 RMSEs were
11.8861 and 11.1937 mph. The 0.0062 mph mean difference between finite heat and Poly4 offers little basis for
ranking them across graphs. These are five related, previously scored snapshots on one graph. The wider, more
thoroughly optimized heat reference clarifies the accuracy comparison. Its dense eigensystem supplies no
deployment speed evidence. No predictive scores influenced the grid, selection, or stopping rule.

## Verification and reproduction

The training targets, centers and scales reproduced bitwise from the read-only HDF. Full-precision successful prior
objectives selected the same five anchors as PR #20. Original-parameter anchor NLL agreed within 6.67e−16 and
posterior means within 2.89e−15; all raw RMSEs passed absolute tolerance 1e−8. On a tiny weighted graph, independent
matrix exponentials (64 and 256), the degree-derived projector, dense solves and log determinants gave max errors
9.44e−15 in covariance, 3.67e−15 in NLL/obs and 1.73e−15 in log-variance gradients. Value tolerances were
1e−10 absolute / 1e−9 relative and gradient tolerances 1e−9 / 1e−8. Idempotence, symmetry, null residual and
gap-predicted tail checks passed.

Base `565ae48aac57cf93a743b8d38774d93343011934`. Upstream trees verified clean at the PR #20 pins.
The existing traffic environment on Apple M4 used Torch 2.11.0, GPyTorch 1.15.2, SciPy 1.17.1, NumPy 1.26.4,
CPU float64 and four intra-op / ten inter-op threads. Optimizations took 4.97 s, scoring 0.047 s, whole worker
about 12 s and sampled peak group RSS 537.3 MiB (1 s sampling). These are resource accounting figures.
The run emitted zero warnings. Both environments passed 9/9 tests before and after, each with three existing
strict-CG warnings. All 586 historical files and the frozen inputs, including the HDF, passed post-run checks.

From the repository root with PR #20 local inputs available, use a fresh output directory.

```sh
out=results/heat_baseline_profile
mkdir "$out"
cp ~/Downloads/CODEX_STEP_3B_HEAT_BASELINE_PROFILE.md "$out/"
shasum -a 256 experiment_heat_profile.py "$out"/*.md results/spectral_expressivity/{training.pt,attempts.json,scores.csv,freeze.sha} results/traffic/learned_polynomial_fresh/{manifest.json,data.npz} results/traffic_upstream/experiments/data/traffic/PEMS/pems-bay.h5 > "$out/freeze.sha"
sed 's|results/spectral_expressivity/|results/heat_baseline_profile/|g;s|experiment_spectral_expressivity.py|experiment_heat_profile.py|;s|>= 3600|>= 1200|' results/spectral_expressivity/watch.sh > "$out/watch.sh"
bash "$out/watch.sh"
.venv/bin/python -m unittest discover -s tests -v
results/traffic_env/bin/python -m unittest discover -s tests -v
results/traffic_env/bin/python -m py_compile experiment_heat_profile.py
git diff --check
```

The measured run additionally fingerprinted the prior inventory, package list, launcher and reused source helpers.
Local evidence includes the pre-optimization manifest, full-precision attempt records with gradient/status fields,
selected parameters, normalized means and scores, and the finite-profile plot with dotted infinite-time references.
The two additions are the 198-line experiment and this report. Existing kernels, experiments and results were preserved.
