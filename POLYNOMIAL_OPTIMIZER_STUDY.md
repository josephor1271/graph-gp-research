# Learned polynomial / GeneralGRF optimizer development study

The frozen rule retains A independently for Poly and GeneralGRF. Poly C’s minimum validation NLPD is only .000609 below A;
General A is the minimum. All candidates lie within .01 of their method’s minimum, and A has the lowest median training
seconds. These four policies provide no validation-based reason to replace A.

Post-hoc development tuning: Jan16/Feb13/Mar13 2017, PR13 indices 2/6/10. First 200 of the ordered 250 training sensors fit;
remaining 50 validate. Undo saved normalization, reconstruct raw training values, then normalize both subsets using only
200. Reconstruction, ID alignment/disjointness, finite targets and graph/map hashes pass. Outer 75 IDs serve separation
checks; neither outer-label array yt/raw is opened. Input files are hashed.

Unchanged PR12 model_for; fresh RNG0 per fit. Poly degree2/squared-degree4; official GeneralGRF J3/.5/W4096 with cached
303/404 walks, never resampled. CPU float64/four threads, observed Cholesky 800; no CG/SLQ or training eigendecomposition.
AdamW defaults/weight decay .01; D changes param_groups LR while preserving optimizer state.

|Policy|Learning rate|Updates|
|---|---|---:|
|A|.05|1000|
|B|.01|1000|
|C|.05|2000|
|D|.05 first 1000, then .005 next 1000|2000|

All 36 fits /54,000 updates completed; zero failed/dropped/pending. Separate 20-update pilots passed: Poly .167504s,
General/303 .269181s. Full fits restarted. A’s separate 1000-update replay is excluded from selection/cost aggregates.

Primary: internal marginal observation NLPD in units normalized on the inner 200, exact own-kernel posterior, noise once.
Average General seeds per date, then three dates. Complete candidates only. Within <.01 of minimum NLPD, select cheapest
median individual fit training time (3 Poly/6 General); RMSE is secondary. Only prescribed final checkpoints are scored.

Date-averaged metrics; time cell = mean setup / median training / mean validation seconds. C/D have double A/B’s updates.
Every candidate is retained; raw per-date/seed evidence stays ignored.

|Method|Policy|Done|Val NLPD|RMSE mph|Train NLL/obs|Setup / train / val s|
|---|---|---:|---:|---:|---:|---|
|Poly|A|3/3|1.183013|11.778420|1.338302|0.064376 / 7.209504 / 0.004563|
|Poly|B|3/3|1.184694|11.793333|1.338789|0.000845 / 7.271616 / 0.003973|
|Poly|C|3/3|1.182404|11.774962|1.338252|0.000804 / 15.313519 / 0.003816|
|Poly|D|3/3|1.182890|11.777010|1.338283|0.001105 / 15.170046 / 0.003723|
|General|A|6/6|1.244258|12.030514|1.350634|0.000391 / 11.926009 / 0.011609|
|General|B|6/6|1.251220|12.333714|1.358077|0.000250 / 12.076015 / 0.011288|
|General|C|6/6|1.245032|12.028623|1.350552|0.000253 / 26.820674 / 0.011538|
|General|D|6/6|1.245127|12.030404|1.350602|0.000254 / 24.991944 / 0.011135|

Baseline and selected policy are identical: all NLPD/RMSE/NLL/time deltas are zero. Poly C takes median 15.313519s versus A
7.209504s for its .000609 NLPD gain. General B worsens NLPD by .006962 and RMSE by .303200mph. A/B timing margins are only
.062112s/.150006s (Poly/General), sensitive to ordinary Mac variation; no timing repeats added.

No convergence certificate: all final windows of 100 losses have negative net changes, but upward steps remain. General/404/Feb13/A:
+.001361 (range.001589); General/404/Mar13/C: +.000542; Poly/Jan16/C: +.000337. Maximum final raw gradient norms: Poly C
.004287, General B .001567. D smooths maximum late ranges to 7.25e-6/5.10e-6 without a qualifying validation gain. Raw norms
depend on coordinates/AdamW decay.

Normalized noise variance still moves under B: General/404/Mar13 .113307–.135508 over the last 100 updates;
General/303/Mar13 .061885–.079172. Low NLL is not a convergence/calibration certificate. Longer training slightly improves
aggregate NLL without comparable NLPD benefit.

Nine tests pass before/after in both environments. Poly covariance/NLL/raw-gradient dense-reference
errors:8.88e-16/2.22e-16/2.22e-16. Four General gates (both seeds, initial/noninitial) compare six selected finite-feature
rows, covariance, likelihood, all raw gradients and posterior/NLPD to independent dense algebra; logged errors <=2.22e-16.
No clipping or added jitter.

Initial prefit check hit unsupported sparse-CSR GP posterior-diagonal access. Log/manifest retained; the gate was corrected
to validate the unchanged production selected-block posterior helper. Only driver hash changed; inputs/policies/shared
source/tolerances stayed fixed. An initial test-log directory race was corrected before tests ran.

A replay has identical initial/final parameters and 1000 losses. Nine C/D groups match A’s first 1000 losses; D counters
reach 2000 with declared LR history. Independent local score errors:3.55e-15 RMSE/4.44e-16 NLPD. Altered
script/config/data/walk/checkpoint guards reject changes; resume skips 36 without modifying results. Extra trace/score/guard
audits are local commands. Sparse UserWarnings/old strict-CG test warnings retained; no experiment NumericalWarning.

Full training sum 599.988660s excludes setup/validation/final-gradient audits/checkpoint I/O; finite-gradient/noise logging
is included. Walk construction is historical PR13 work. Three related development dates, one graph, 200 fitted labels: this
does not retest the original 250-observation March instability or assess outer 75 performance. No external-test, large-size
or speedup claim; no 5K–50K run.

Run from repo root with the pinned traffic environment and PR13 caches; archive outputs for fresh fits.

```sh
results/traffic_env/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
results/traffic_env/bin/python -u experiment_polynomial_optimizer.py check
caffeinate -i results/traffic_env/bin/python -u experiment_polynomial_optimizer.py pilot
caffeinate -i results/traffic_env/bin/python -u experiment_polynomial_optimizer.py full
caffeinate -i results/traffic_env/bin/python -u experiment_polynomial_optimizer.py replay
git diff --check
```

Base main `1ed006feddc233cddadc10c87245b025028c52d8` includes merged PR #13. Source pins:
`268fff9a3724cdd480dce4b2946068d8211b4f30` (core), `274b7ee393ee516585552f053e9d4a9c6ad2222f` (experiments).
Input/package/source hashes are frozen in the local manifest.

Study manifest SHA256: `0cf96987eed8fc09997fbffa5e446efbdf523d56d8f0fcacfefbf733766645b9`.
Driver SHA256: `4c52e0533b357e102c953203575570588c1e91eb741a80e6f5a36414ec3d7599`.

Added only the 240-line experiment and this report. The 20-line overage supplies finite-feature gates, resume guards and
replay. Deleted nothing; kernels/operators/recurrence/GP, tests, earlier experiments, environments/dependencies, README,
upstream and read-only toolbox kept. Evidence/commands stay ignored under results/traffic/polynomial_optimizer/. Leave PR
unmerged for planner review.
