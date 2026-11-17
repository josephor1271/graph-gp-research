# Poly posterior variance: development qualification

**512 paths is the first candidate budget:** both aggregate targets pass in24/24 prescribed case/stream combinations;128 passes0/24 and1024 passes24/24. All72 numerical prefix checks pass. This qualifies a small fixed-parameter development candidate, not calibration, generalization or large-N uncertainty.
Main `3989d58b8fe081446d4fe89b39bced47261d0dfc` contains merged PR17. No training was run. Initial/stress states use H targets; fitted H/M states load PR16's100-update Poly32-probe/seed91 checkpoints. Both r10/11 graphs/splits/targets and loaded raw parameters/fingerprints/checkpoint hashes match. These are previously seen development data, not fresh test datasets; inherited `final/` directory names do not change that.

## Method and numerical evidence

The unchanged degree2 filter has coefficients `[1+a1-a2,a1,a2]`; apply `B=sqrt(outputscale)q(L)` by the existing recurrence. For each32-column Gaussian batch, centered paths are `u=(Bz)[test]−C CG(A,(Bz)[train]+sqrt(noise)e)`, with A=Ktrain+noise*I and C=Ktest,train as existing lazy operators. Never use the degree4 covariance as a root or square an observed-only filter. Unbiased latent sample variance uses P−1; observation variance adds fitted noise once. Scores use the analytic-form CG mean, not the finite sample mean.
CPU float64/four Torch threads, explicit library CG1e-4/max_iter1000/no preconditioner, seeds421/422/423 with nested128/512/1024 prefixes. Gaussian z/e are identical for sparse and Cholesky paths. SparseGuard forbids full graph/observed/cross/joint/posterior blocks and Cholesky; scoped patches reject HeatOperator exact-entry/diagonal fallback. No clipping, added jitter, kernel or solver changes.
On weighted n36/r9 initial/stress states, independent monomial B, BBᵀ and the full linear-map covariance (including training-noise gain*gainᵀ) pass atol1e-10/rtol1e-9. Worst action/covariance/map discrepancies:8.88e-16/2.22e-16/2.12e-16. One32-path same-draw comparison per state also passes numerical checks. References are outside guards/timers.
Across eight n1000 states×three streams×three prefixes: max true per-RHS residual1.3792e-4≤2e-4; analytic mean max error1.4569e-4≤.002; iterative versus same-draw Cholesky latent-variance relative L2≤1.505e-5≤.002. Mean-cache residual max9.435e-5. Requested CG tolerance is not a per-column bound. No zero RHS occurred (zero-RHS count/absolute-residual fields are retained); no NumericalWarning or deterministic/resource failure.

## Monte Carlo error and cost frontier

Each state group below contains two graphs×three MC streams; prefixes share paths. Pass counts are **latent≤10% / observation≤5% / both**, each out of6. Error ranges are relative L2 percentages; sampling milliseconds include Gaussian generation, sparse root, CG/update, path storage and that prefix's variance computation.

|State|P|Pass L/O/both|Latent L2 % range|Observation L2 % range|Sampling ms range|
|---|---:|---|---:|---:|---:|
|initial|128|0/0/0|11.75–12.97|5.65–6.22|45.2–47.7|
|stress|128|0/0/0|11.25–13.49|8.42–10.10|57.6–59.7|
|fitH|128|0/3/0|11.77–13.27|4.72–5.61|44.0–72.3|
|fitM|128|0/6/0|11.70–13.48|1.89–2.97|46.4–51.6|
|initial|512|6/6/6|6.28–6.59|3.02–3.17|178.1–187.5|
|stress|512|6/6/6|5.89–6.48|4.41–4.85|231.5–240.7|
|fitH|512|6/6/6|5.90–6.77|2.20–3.12|176.6–287.8|
|fitM|512|6/6/6|5.97–6.56|0.98–1.46|184.7–205.7|
|initial|1024|6/6/6|4.33–4.75|2.08–2.28|355.3–373.2|
|stress|1024|6/6/6|4.15–4.87|3.10–3.64|462.0–474.1|
|fitH|1024|6/6/6|4.13–4.84|1.54–2.23|353.2–480.0|
|fitM|1024|6/6/6|4.52–4.87|0.68–1.13|366.5–428.8|

Across24 combinations per budget, latent/observation/joint counts are128:0/9/0,512:24/24/24,1024:24/24/24. No configurations/seeds were added after scoring. Stress observation error at512 reaches4.85%, close to the5% target; this is not a guarantee for new states.
Ideal iid Gaussian per-node variance relative RMS is sqrt(2/(P−1))=12.55%,6.26%,4.42%. Expected squared relative L2 is2/(P−1), without assuming independent graph nodes. Measured latent ranges follow that sampling scale; this is a descriptive comparison, not an equality or hypothesis test. The numerical same-draw error is far smaller.
Aggregate passing does **not** give pointwise bounds: worst latent node/max per-state p95 relative errors are50.73%/27.64% at128,25.27%/13.45% at512,18.45%/10.28% at1024. At512 observation counterparts are18.73%/9.01%. Per-node relative denominators are the positive analytic variances, with no floor (minimum latent=.096987 normalized).
Mean-only cost is282.6–292.0ms, once per state. Standalone measured mean+variance sums are327.8–356.9ms at128,460.4–572.4ms at512,637.8–764.5ms at1024. Incremental sampling costs128→512 are131.8–215.5ms and512→1024 are176.5–233.5ms. Larger prefixes sum their batches plus their own moment computation, excluding earlier-prefix moment/score overhead. A scoped timer hook stops before the helper's true-residual audit; dense references, residual audits and I/O are excluded. Guards/mock instrumentation and library warmup remain included. Different MC seeds are not timing repeats; no GeneralGRF/optimized production-runtime comparison is claimed.

## Predictive scores are descriptive

Exact own-kernel reference scores below; units in slash-separated columns are **normalized/raw**. Candidate RMSE uses the same analytic-form mean for every prefix. Normalized NLPD uses latent estimate+noise once; raw NLPD adds log(train scale), and raw variances multiply by scale². All raw/normalized estimates/references are retained.

|Graph/state|RMSE norm/raw|Marginal observation NLPD norm/raw|95% observation coverage|
|---|---:|---:|---:|
|10/initial|.928384/.595744|1.385798/.942164|.976|
|10/stress|1.011827/.649290|2.778606/2.334971|.652|
|10/fitH|.910188/.584068|1.331552/.887918|.964|
|10/fitM|.951506/.605850|1.370650/.919236|.972|
|11/initial|.962650/.560747|1.410391/.869971|.972|
|11/stress|1.056530/.615432|2.948858/2.408438|.604|
|11/fitH|.950810/.553851|1.368360/.827940|.944|
|11/fitM|.913142/.571285|1.334094/.865090|.976|

Worst candidate-versus-reference NLPD change (normalized or raw) is.05102/.03029/.01932 at128/512/1024; coverage changes≤.016/.012/.008. **Exact stress coverage is only60–65%**, despite nominal95%; the frozen low-noise/misspecified state is not calibrated. Accurate numerical reproduction cannot correct model misspecification. Coverage uses correlated nodes and previously seen development states; neither it nor three MC streams establishes calibration or accounts for hyperparameter uncertainty.

## Evidence, reproduction and limits

All ten states completed:39.330s summed attempt wall, max4.318s/state; no60-minute/10-minute/8GiB cap hit. Peak one-second sampled process-group RSS517.031MiB includes dense references, imports and software, can miss peaks/double-count shared pages, and is not pure candidate memory. The two test-path buffers occupy3.906MiB at n1000/P1024; live candidate RHS width stays32. Qualification references are dense only at36/1000 nodes; no large-scale claim.
Manifest SHA256:`09c433ca321d37f479dead6a8c39bdd5cc8588a02b86cf262b1afaf973e3ad71`. It freezes source/software/pinned upstreams, graph/state/seed schedule and consumed PR16 input/JSON/PT hashes before scoring. Pins:core268fff9a3724cdd480dce4b2946068d8211b4f30 / experiments274b7ee393ee516585552f053e9d4a9c6ad2222f, tracked trees clean. All766 previous artifact hashes/file set remain unchanged. Warnings are retained; only existing non-numerical sparse warnings occurred in these cases, while strict-CG warnings in the existing tests remain visible.
New ignored `results/poly_posterior_qualification/` retains exact arrays/scores, z/e batches and residual norms, matched paths, all72 state/seed/prefix JSON rows, `rows.csv`, timings/RSS, raw parameter/checkpoint provenance and commands. Preserve/archive existing output for a fresh repetition; the driver refuses silent retries. No scientific logic is hidden in local launch/table commands.

```sh
OUT=results/poly_posterior_qualification; PY=results/traffic_env/bin/python
mkdir -p "$OUT"
cp /path/to/CODEX_POLY_POSTERIOR_UNCERTAINTY_QUALIFICATION.md "$OUT/work_order.md"
$PY experiment_poly_posterior.py freeze
# Per-state entry point used by launcher: $PY experiment_poly_posterior.py 36 9 initial
caffeinate -i bash "$OUT/launch.sh"          # actual serial watchdog launch, starting before tiny
$PY -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

Baseline/final existing suites:9/9 in each environment. Diff checks pass. Added one198-line experiment and this report; deleted nothing; kernels/recurrence/GP/training/dependencies/old evidence kept. No training, GeneralGRF benchmark or5K–50K uncertainty run.512 is a candidate for later planner review only; stop here and leave the PR unmerged.
