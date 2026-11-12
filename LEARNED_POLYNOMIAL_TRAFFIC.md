# Learned polynomial traffic GP

Poly has lower RMSE, observation NLPD and training NLL than heat, C and GeneralGRF on all three dates at 1000
updates. These previously inspected PEMS-BAY snapshots use the original 250/75 split: a post-hoc comparison.

## Model, data and checks

`q(lambda)=1+a1*(T1(lambda-1)+1)+a2*(T2(lambda-1)-1); K=sigma_f^2*q(L)^2`. q(0)=1 fixes normalization; ScaleKernel
learns variance. `Ti*Tj=(T(i+j)+T|i-j|)/2` produces five degree0…4 coefficients for existing
HeatOperator/recurrence. Selection follows squaring; no new solver, LinearOperator or GP model.
Fixed start: beta0=2*sigmoid(1)=1.462117157; a1=-.196611933241482, a2=.133611661347131, signal
SD=softplus(1)=1.313261688, raw scale1, raw noise0 (variance .693247181). GeneralGRF uses
sigma*[1,-beta0/2,beta0^2/8] on the exact cached C walks, J3/.5/W4096, seeds101/202. Its initial finite
covariance differs from Q2^2 by .104500/.101551 relative Frobenius on the sensor block; intended coefficients
match, finite kernels differ. Heat degree20; CPU float64/four threads; AdamW .05/default decay .01, 1000 updates,
Cholesky800. One Jan9 Poly 20-update pilot passed; full fits restart at the fixed initialization. Same weighted
symmetric normalized L: 1016 nodes/1173 edges, split seed20261104, train-only scaling and Jan9/Feb6/Mar6 2017
17:30 snapshots. Data SHA256: `802e5e523b5db6ff9d13da3388c566de7d0212b09b23497a3d0e50a0385ad71e`. Base main
`8c7cede6f79a4dd3924c09b75c470c4bf6ccea71`; open #9/#11 are not dependencies. Pins: [core
268fff9](https://github.com/MatthewZhang473/Fast-Graph-GP/tree/268fff9a3724cdd480dce4b2946068d8211b4f30),
[experiments
274b7ee](https://github.com/MatthewZhang473/Graph-Random-Features-for-Scalable-Gaussian-Processes/tree/274b7ee393ee516585552f053e9d4a9c6ad2222f).

Nine tests pass in each environment (baseline eight): initial Q2/q(0), symmetry/PSD, positive noisy Cholesky,
negative/nontrivial filters, actions/repeated/rectangular IDs, diagonals/slicing, four raw gradients, NLL,
posterior moments, observation NLPD and backprop. Separate local selected-row audits cover paths through all
traffic vertices using only 6/325 filter rows, never full Q/graph identity/kernel. Max new-fit
NLL/moment/gradient errors: 4.44e-16/1.12e-15/6.95e-16 (independent NumPy/analytic gradients for fitted models).
The full audits are separate local checks, not implemented in the tracked experiment; commands are
`selected_audit_command.txt` and `results_audit_command.txt` under output. Full-graph operator
densification is guarded. Existing prepare's off-timer spectrum check remains; the new kernel/train/deployment
path uses no eigensystem. No clipping or added jitter; no failed gate/fit or experiment NumericalWarning.
Sparse/deprecation warnings and the existing strict-CG unit-test warnings are preserved.

## Per-date evidence

RMSE: mph; NLPD: standardized marginal observation density, adding learned noise once. NLL: own-kernel
likelihood per training observation. Exact 325-sensor oracle scoring is off-timer. Time tuple:
setup/train/deployment/oracle seconds. Heat/C setup/train/deploy are **historical PR6**, with all nine
checkpoints, starts, scores and 1000-loss histories verified; oracle times are new. All Poly/General times are
new, single measurements.

|Date|Model|RMSE|Obs NLPD|NLL/obs|Setup / train / deploy / oracle s|
|---|---|---:|---:|---:|---|
|2017-01-09|Heat|15.738451|1.421688|1.356774|0.220478 / 31.413013 / 0.076833 / 0.024844|
|2017-01-09|C/101|15.701407|1.359017|1.372251|0.000279 / 15.153830 / 0.173518 / 0.015975|
|2017-01-09|C/202|15.514496|1.341682|1.371790|0.000410 / 13.953406 / 0.143632 / 0.014912|
|2017-01-09|Poly|14.022977|1.223779|1.289893|0.000800 / 10.056612 / 0.019301 / 0.005970|
|2017-01-09|General/101|15.154192|1.503597|1.312642|0.000252 / 15.730053 / 0.285937 / 0.015770|
|2017-01-09|General/202|14.186767|1.408962|1.328126|0.000239 / 16.177482 / 0.404582 / 0.020636|
|2017-02-06|Heat|16.065616|1.377650|1.352924|0.000986 / 39.864338 / 0.080486 / 0.028848|
|2017-02-06|C/101|15.898595|1.317218|1.368829|0.000177 / 17.606334 / 0.155108 / 0.018649|
|2017-02-06|C/202|15.787873|1.309319|1.364173|0.000201 / 17.465764 / 0.154857 / 0.020706|
|2017-02-06|Poly|14.612846|1.226193|1.275660|0.001136 / 10.442261 / 0.018950 / 0.004397|
|2017-02-06|General/101|15.261952|1.368845|1.311518|0.000223 / 16.457983 / 0.283433 / 0.016330|
|2017-02-06|General/202|14.665232|1.305394|1.310707|0.000267 / 16.129281 / 0.267819 / 0.016830|
|2017-03-06|Heat|16.034552|1.476915|1.361634|0.001619 / 36.732598 / 0.079339 / 0.022355|
|2017-03-06|C/101|15.554065|1.397026|1.369795|0.000182 / 17.294806 / 0.154832 / 0.016015|
|2017-03-06|C/202|15.503558|1.384569|1.371446|0.000202 / 17.360520 / 0.158640 / 0.015243|
|2017-03-06|Poly|14.782277|1.366000|1.319641|0.000655 / 10.267184 / 0.020681 / 0.005361|
|2017-03-06|General/101|15.254501|1.457070|1.342093|0.000217 / 15.958517 / 0.193211 / 0.017283|
|2017-03-06|General/202|14.921524|1.420338|1.347247|0.000258 / 16.080338 / 0.194446 / 0.017015|

Poly minus heat (RMSE/NLPD), Jan/Feb/Mar: -1.715474/-0.197910; -1.452770/-0.151457; -1.252275/-0.110915. General
improves C's RMSE in all six pairs, but worsens NLPD in five (exception Feb/202). Poly's lower training NLL
means achieved fit quality is not matched. These fixed-budget fits do not certify global optima or isolate
determinism from finite features, model family, parameterization and AdamW coordinate-dependent decay.
Heat beta stays near its frozen (0,2) bound. New families have four trainables including noise; heat/C have three.

Shapes below: Poly=(a1,a2); General=f0,f1,f2 includes amplitude, without separate sigma. Signal SD/noise
variance are standardized-target units; full raw states/losses stay local.

|Date/model|Shape|Signal SD|Noise variance|
|---|---|---:|---:|
|2017-01-09/Poly|-0.313516, 0.518879|1.935584|0.007586|
|2017-01-09/General/101|1.688063, -3.577457, 1.250217|—|0.005190|
|2017-01-09/General/202|1.810081, -3.623634, 1.236520|—|0.011167|
|2017-02-06/Poly|-0.297478, 0.525810|1.873439|0.017251|
|2017-02-06/General/101|1.662292, -3.363402, 1.136486|—|0.022401|
|2017-02-06/General/202|1.735612, -3.281982, 1.104660|—|0.090876|
|2017-03-06/Poly|-0.297006, 0.538382|1.809371|0.099352|
|2017-03-06/General/101|1.628571, -2.652191, 0.816764|—|0.200840|
|2017-03-06/General/202|1.669468, -2.726801, 0.853076|—|0.217999|

Normalized responses have interior notches; at lambda2 Poly=.139105/.164060/.164827 versus heat≈.0185. The local
spectral plot shows deterministic curves only; finite General is not an exact polynomial function of L. The plot
did not select parameters. Poly deploys ordinary GP mean/variance; General deploys 512 pathwise samples,
seed100, CG1e-4, distinct from oracle. Historical walk costs 2.137291/1.964202 s are shared across dates and
C/General. No speedup or fresh paired timing conclusion.

## Reproduction and scope

Existing pinned traffic environment/data/checkpoints; preserve logs. Completed rows resume by data/checkpoint
hashes. Missing old checkpoints are never silently refitted.

```sh
results/traffic_env/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
results/traffic_env/bin/python -u experiment_learned_polynomial_traffic.py check
caffeinate -i results/traffic_env/bin/python -u experiment_learned_polynomial_traffic.py pilot
caffeinate -i results/traffic_env/bin/python -u experiment_learned_polynomial_traffic.py full
git diff --check
```

Output is ignored under `results/traffic/learned_polynomial/`: per-run rows, checkpoints, losses/parameters,
posterior moments, source/data/package hashes, audit commands and plot inputs. Nine new full fits, one pilot,
nine verified reused baselines; none unfinished. Added: kernel49/tests86/experiment195 Python lines plus this
note (four paths); the 200-line soft-limit overage supplies the specified kernel and gates. No existing
recurrence/operator/model/experiment/default/dependency or upstream edits, deletions, broader degrees/datasets
or further work. Leave PR unmerged for planner review.
