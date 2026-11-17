# Frozen learned-kernel scaling continuation

Main `f779b471b5b1b94197a9fd5e79072eff23f821e3` contains merged PR16. Its reviewed selection SHA256 is `b781f585c498a2721d9c06165657c9bf4d742f53d8e56459fcfed0c7dde974fc`:32 probes/16 Lanczos, CG/evalCG1e-4/maxCG1000, eager/Cholesky/preconditioner0. PR15/16 studies are referenced, not rerun.
Unchanged weighted road-like normalized L; H covariance exp(−1.5L), M .5exp(−1.5L)+.5(L/2)^2; raw noise variance .1, train/test quarters, train-only normalization. Seeds:base+n+100000r with graph/data/split/z2/walk/trace bases731000/741000/751000/761000/771000/781000. Regimes share graph/noise/splits; only r is independent.
Heat6/Poly squared quadratic/official General J3 halt.5 use existing starts (3/4/4 parameters), RNG0, CPU float64, four Torch threads/workers, AdamW .05/default decay .01. Final fits always100 updates, probes seeded once. Scores are noisy-test raw RMSE; large NLL is estimated, no uncertainty/NLPD calculation.

## Development and freeze

n2500/r9 numerical endpoint checks passed for all8 original100-update arms/regimes and all6 selected200-update replays: max |estimated−exact NLL|=.000657/obs, mean discrepancy7.78e-6, residual7.16e-6; no NumericalWarning. Exact references use each finite kernel on selected1250 blocks off timer; no absolute-gradient gate was reintroduced.
W1024 wins the prescribed ≤1% rule independently in H/M:relative RMSE versus4096 +.1244%/−.6141%. Its selection preceded final inputs. The200-update diagnostic uses fresh identical-seed replays with bit-identical first100 losses; unchanged final100 budget. Last loss is the pre-update stochastic training value; exact NLL is audited at the endpoint.

| Regime/arm | RMSE100→200 | Exact NLL100→200 | Last loss100→200 | Training s100→200 |
|---|---:|---:|---:|---:|
|H/Heat|.584718→.584623|1.404253→1.404191|1.404292→1.403348|8.002→7.764|
|H/Poly|.585469→.585426|1.404280→1.404008|1.403837→1.403492|4.997→5.524|
|H/General1024|.587005→.586853|1.402377→1.402275|1.401513→1.401901|8.305→12.368|
|H/General4096|.586275→—|1.405016→—|1.404296→—|12.702→—|
|M/Heat|.632915→.633377|1.419284→1.419084|1.419293→1.419020|7.154→7.730|
|M/Poly|.636743→.636850|1.416721→1.416620|1.416006→1.417381|4.886→5.625|
|M/General1024|.633971→.634222|1.416045→1.415987|1.415451→1.416537|6.187→12.362|
|M/General4096|.637888→—|1.415740→—|1.415325→—|6.295→—|

All selected methods worsened M validation RMSE at200 despite lower exact NLL. Nonmonotonic replay timings are retained; no timing repeats or inference/optimizer changes were added. Six n512/r9 guarded two-update/mean smoke checks passed (max residual4.53e-6), including full-joint forbidden shapes and Cholesky rejection.
Before any final data,72 cases were frozen in size/r/regime order, rotating [Heat,Poly,General] by(size_index+r+regime)%3. Manifest SHA256:`8c2c6feccc682046465c264aadcfca061c01a43077967c15e6e7be1defed5e99`; driver SHA256:`7bfba056f005f5bed8890756e79820082f8ee4eb4c7e7d988e8205b77b04b4f5`.

## Final comparison

All72 fits completed100 finite updates with no NumericalWarning, failed fit, skipped case or limit hit. Triples below follow **Heat / Poly / General1024**; SD is sample SD across r0/1/2. Negative paired RMSE deltas favor Poly; time ratios<1 favor Poly. Every paired denominator is3, with no pooling of regimes or sizes.
Standalone total=actual model/optimizer/MLL setup+original fresh walks+training+cold mean; original walk cost is charged in full to each General standalone fit, including the second regime. Residual checks/audits, shared graph/data, imports and I/O are excluded; separate attempt wall includes them. All per-case phases and metrics are in local `final_rows.csv`/JSON.

H (heat control):

|N|Done Heat/Poly/General|Raw RMSE mean±SD|Train s mean±SD|Standalone total s mean±SD|Poly−General RMSE mean[range];pairs|Poly−Heat RMSE mean[range];pairs|Median TP/TG;TP/TH|
|---:|---|---|---|---|---|---|---|
|5000|3/3;3/3;3/3|.614184±.000714 / .615702±.000583 / .615558±.001065|6.98±.18 / 4.91±.03 / 11.97±.05|7.26±.20 / 5.18±.03 / 14.75±.05|+.000144[−.000553,+.001015];3|+.001518[+.000185,+.002352];3|.350;.705|
|10000|3/3;3/3;3/3|.607888±.007868 / .608023±.007857 / .609091±.007377|16.31±.05 / 11.49±.02 / 25.83±.01|16.60±.05 / 11.77±.02 / 31.36±.22|−.001068[−.001802,−.000582];3|+.000135[+.000043,+.000191];3|.377;.709|
|25000|3/3;3/3;3/3|.608295±.002079 / .608565±.002117 / .609663±.002559|44.48±.01 / 31.06±.36 / 68.84±.71|44.85±.02 / 31.38±.35 / 83.71±.32|−.001098[−.001399,−.000589];3|+.000270[+.000186,+.000432];3|.375;.699|
|50000|3/3;3/3;3/3|.605020±.002867 / .604998±.002703 / .605613±.002645|87.49±.38 / 59.90±.49 / 103.12±.36|87.91±.39 / 60.27±.51 / 133.65±.35|−.000615[−.000885,−.000200];3|−.000022[−.000169,+.000174];3|.451;.686|

M (multiscale):

|N|Done Heat/Poly/General|Raw RMSE mean±SD|Train s mean±SD|Standalone total s mean±SD|Poly−General RMSE mean[range];pairs|Poly−Heat RMSE mean[range];pairs|Median TP/TG;TP/TH|
|---:|---|---|---|---|---|---|---|
|5000|3/3;3/3;3/3|.645309±.007470 / .644206±.005814 / .644363±.005065|6.81±.15 / 4.94±.04 / 12.05±.10|7.09±.15 / 5.21±.04 / 14.84±.08|−.000157[−.000967,+.000648];3|−.001103[−.002782,+.000660];3|.351;.738|
|10000|3/3;3/3;3/3|.624590±.011302 / .622117±.011206 / .622832±.010796|16.37±.09 / 11.38±.06 / 25.98±.21|16.66±.09 / 11.66±.06 / 31.50±.20|−.000715[−.001309,+.000041];3|−.002473[−.002884,−.001889];3|.368;.701|
|25000|3/3;3/3;3/3|.636796±.006080 / .634939±.006645 / .635190±.006800|45.06±.25 / 31.22±.64 / 69.14±.32|45.42±.26 / 31.54±.65 / 84.00±.49|−.000251[−.000555,+.000348];3|−.001857[−.002812,−.001356];3|.380;.700|
|50000|3/3;3/3;3/3|.637018±.003383 / .634946±.003146 / .634999±.003373|87.75±.21 / 59.28±.44 / 103.56±.66|88.19±.24 / 59.65±.46 / 133.97±.65|−.000052[−.000312,+.000119];3|−.002072[−.002299,−.001797];3|.445;.678|

Poly is faster in every matched final pair. On M its mean RMSE improves on Heat at each size; its much smaller advantage over General has mixed individual signs. On H, Heat has lower mean RMSE through25K; the50K Poly/Heat difference is tiny and mixed. These are measured fixed-budget tradeoffs, not significance/equivalence, convergence, SOTA or asymptotic claims; only three datasets/size and one synthetic graph family, no real-graph transfer or large-N uncertainty evidence.
Attempt wall3233.635s, standalone sum3175.249s; max case wall137.682s, UTC/perf gap.0283s. Limits:1800s/case,4h allocation,12GiB sampled group RSS; none hit. Peak941.531MiB is the2-second sampled process-group RSS sum on an Apple M4/16GiB Mac, may miss peaks/double-count shared pages, and is not exact model memory. All144 fit warnings are preserved UserWarnings; no NumericalWarning. Max fitted residual9.31e-6.
First successful50K fit/method (r0/H) passed full-joint SparseGuard mean replay at atol.002 normalized/rtol1e-8; residuals Heat6.26e-6/Poly4.42e-6/General6.85e-6. Two initial replay attempts failed before prediction because Heat/Poly's recorded directory predated an unrelated walk-cache file. Their failed JSON/logs remain; temporarily parking only that unused cache reproduced the recorded file set, then the identical cache was restored/hash-verified. No source, fit, mean, metric or tolerance changed. This strict audit-directory requirement is a reproduction limitation; large guarded replays test prediction, not every training operation.

## Reproduction, evidence and limits

Use the existing pinned traffic environment/checkouts in README:core268fff9a3724cdd480dce4b2946068d8211b4f30 / experiments274b7ee393ee516585552f053e9d4a9c6ad2222f, tracked trees clean. New evidence is ignored in `results/learned_kernel_scaling_final/`:all configurations/provenance, per-case hashes/losses/parameters/means/checkpoints, endpoint audits, attempt logs/RSS and descriptive tables. Preserve/archive existing output before a fresh repetition; existing records verify/skip. Prior results remain separate.
```sh
PY=results/traffic_env/bin/python
OUT=results/learned_kernel_scaling_final
mkdir -p "$OUT"
cp /path/to/CODEX_LEARNED_KERNEL_SCALING_CONTINUATION.md "$OUT/work_order.md"
$PY experiment_learned_kernel_scaling_final.py fit 2500 9 0 Heat 0 100
# Repeat fit for H/M:Heat0,Poly0,General1024,General4096; then choose.
$PY experiment_learned_kernel_scaling_final.py choose
# Selected H/M arms:fit2500/r9/200 (fresh replay),then fit512/r9/2;then freeze.
$PY experiment_learned_kernel_scaling_final.py freeze
# Serial fit commands follow manifest.schedule; audit first successful50K per method.
# Preserve the recorded input-file set for audits (park later unused caches).
```
Actual local commands: `caffeinate -i bash "$OUT/development_commands.sh"`, then `caffeinate -i bash "$OUT/launch_final.sh"`; `audit_commands.sh` and `replay_recorded_input_files.sh` retain the initial audit failures and exact file-set replay commands. Only launches/watchdogs/descriptive tables are outside the driver. Resume verified all72 cases without retraining; source/input/checkpoint hashes are checked, failures never silently retried.
Baseline/final:9/9 existing tests in both environments (`$PY -m unittest discover -s tests -v`; `.venv/bin/python -m unittest discover -s tests -v`), existing strict-CG test warnings retained; `git diff --check` passes. All324 PR15/16 artifact hashes/file set and frozen sources are unchanged. Added one179-line driver/this report; deleted nothing; existing implementation/dependencies kept. No pending fits, merge or further experiment.
