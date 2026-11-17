# Matched posterior uncertainty: native noise precision mismatch

**Stopped again before qualification/scaling.** The authorized diagnostic explains
PR19's original native-versus-adapter path discrepancy: upstream's implicit
float32 identity rounds the conditioning noise. Both implementations meet the
residual and same-draw variance gates, but **strict operator equivalence fails**.
No P* was selected; no 5K–50K predictions or designated large guarded audits ran.
This leaves the scaling question unanswered and establishes no speed advantage.

## Original failure and authorized amendment

Base main `31fd991282f8f026931edb71da7a3bbc8d1e985e` contains merged PR18.
The original PR19 commit `fd6febe3211ac3cc4c45729937ce0b4a4ebe34b8` remains in
history. Its complete failed attempt, manifest, source, thresholds, logs,
reference/bracket arrays and results are preserved: all **41 original study files**
are byte-identical. All **3,570 earlier artifacts** and their filename set also
remain unchanged. New evidence is isolated under the ignored
`results/matched_posterior_scaling/continuation/` directory.

Original n36/r9/H/initial General, seed521/32 paths: native path parity failed
at atol1e-10/rtol1e-9, with max absolute **7.154941639e-9**, max relative
**7.538366823e-7**, 158/288 mismatched elements. Original mean max error1.881e-6,
mean true residual5.488e-6; exact covariance/map discrepancies0/2.776e-16.
The three selected variance brackets contained dense references, max relative
width8.221e-13, max true reference residual8.038e-6. Original attempt2.490s,
sampled inclusive peak426.75MiB. Nothing in that record was relabeled successful.

After reviewing that failure, the planner authorized one focused diagnostic and
conditional continuation, replacing near-machine-precision approximate-path
parity with a residual-aware check **only if deterministic equivalence passes**.
Strict algebra atol1e-10/rtol1e-9, true residual<=2e-4, same-draw Cholesky
variance relative L2<=.002 and all MC gates remain unchanged. The planner explicitly
approved up to450 Python lines; the existing driver is now432 lines. No new
tracked files, dependencies, classes, solver settings or upstream edits.

## Focused diagnostic: actual native CG captured

Used the original graph, walks, state, seed521 and identical z/e perturbations.
Scoped instrumentation captures the actual pinned `pathwise_conditioning` CG
operator, signed RHS, solution and product trace without editing upstream.
Native batch-major draws/signs were reconciled with the adapter's centered paths.
Independent reference uses a dense weighted sum of the actual saved walk matrices,
K=BB^T, dense Cholesky and the full posterior sampling-map covariance.

| Diagnostic against independent dense reference | Adapter | Native |
|---|---:|---:|
| Conditioning A, max abs discrepancy | 2.220e-16 | **1.850e-8 (FAIL)** |
| Cross-covariance C, max abs discrepancy | 2.776e-17 | 2.776e-17 |
| Positive conditioning RHS, max abs discrepancy | 8.882e-16 | 8.882e-16 |
| Actual true per-RHS residual, maximum | 7.624118e-6 | 7.624118e-6 |
| Same-draw latent variance relative L2 | 1.509226e-7 | 1.510366e-7 |
| Same-draw empirical full covariance relative error | 5.585905e-7 | 5.589514e-7 |
| Posterior path vs common Cholesky, max abs | 2.516335e-6 | 2.517655e-6 |
| CG iterations | 11 | 11 |

Full/selected prior products agree with dense algebra within1.111e-15;
A.matmul versus A._matmul agrees exactly for both operators. Both RHS strides are
[32,1]. A focused replay with contiguous batch-major draws produces exactly the
same native paths. sqrt(nu)^2-nu is zero here. No sign, transpose, layout or feature
orientation error was found. Both use actual B=Phi with BB^T, never B².
Exact conditioning-map covariances agree with the independent Schur complement
within3.331e-16. Empirical covariance/variance comparisons use identical32 draws;
they do not establish MC accuracy against analytic variance at P32.

**Identified cause:** installed `IdentityLinearOperator.__init__` defaults to
literal `dtype=torch.float` despite a float64 Torch default. Upstream constructs
`noise_std.pow(2) * IdentityLinearOperator(m, device=device)` without dtype;
its constant diagonal is float32. The adapter's explicit noise diagonal is float64.
The fitted noise is .6932471805599453; its float32 value is .6932471990585327,
a shift **delta=1.84985874396304e-8**. Captured native A minus adapter A equals
delta I to1.111e-16. This fails the original strict algebra tolerance; it is a
real finite-precision operator difference, not merely different CG stopping.

Analysis of saved arrays, without rerunning sampling or CG, separates the effects:
exact solves of the two captured systems differ in paths by7.154866366e-9;
the differential iterative-error contribution is at most2.114e-13. Perturbing
only the independent dense noise diagonal by delta predicts the observed native
path difference to **2.115e-13**. Thus noise casting explains essentially all of
the original discrepancy. Each solve's own absolute path error is about2.516e-6;
those similar errors largely cancel in the native-versus-adapter comparison.

Effective CG settings are unchanged: tolerance1e-4/max1000/no preconditioner,
n_tridiag0, eps1e-10, per-column stop_updating_after1e-10, terminate-by-size off.
The library checks the **mean** normalized residual after at least11 iterations
and has small-denominator safeguards; actual maximum residuals are audited
separately. No solver sweep or safeguard change was performed.

## Residual-aware equivalence and stopping decision

For common independent A0,b0 and each implementation i, the captured true residual
ri=bi-Ai(xi) gives

```
||xi-x*|| <= (||ri|| + ||bi-b0|| + ||Ai(xi)-A0 xi|| + roundoff)/lambda_lower
```

Propagate through ||C0||_F, adding measured prior-product and Ci(xi)-C0 xi
differences. Comparing with computed Cholesky paths also includes its true
residual. Sum individual bounds for native-versus-adapter paths. The code uses
IEEE float64 gamma terms for Gram assembly and norm/product/subtraction rounding;
lambda_lower subtracts the Gram/noise assembly error bound from nu. Every path
satisfies its propagated bound, and both unchanged residual/variance gates pass.
Per-column path bounds span2.016e-6–2.780e-5; observed norm errors are<=4.150e-6.
This is a floating-point diagnostic, not a formal certified enclosure.

However, deterministic operator equivalence fails **before** those approximate
path checks can authorize continuation. Per planner step6, stop for review.
One diagnostic attempt failed; the remaining11 qualification invocations and
all144 final predictions are explicitly unattempted. General's24 combinations
at each of P512/P1024 were not evaluated. Prior Poly24/24 support at both budgets
cannot replace General qualification. No final schedule was activated, no model
was retrained, no failed diagnostic was rerun, and no large cache was generated.

## Costs, provenance and reproduction

Authorized diagnostic:3.035s wall, sampled process-group peak359.56MiB, including
imports, dense references, guards and instrumentation; not candidate-only memory.
Its bracket audit41.254ms, preparation1.244ms, mean17.004ms are partial instrumented
costs. No complete prediction timing, reconstructed/amortized total, paired ratio
or scaling figure exists. No resource cap was reached (1800s/case,4h,12GiB).
Four sparse/storage UserWarnings and no NumericalWarning were recorded in each
attempt. A saved-array analysis helper initially hit variable shadowing; fixed
and logged without rerunning sampling or CG. No jitter/clipping/variance repair.

CPU float64/four threads, Apple M4/16GiB, macOS26.6.2; Torch2.11.0,
GPyTorch1.15.2, linear_operator0.6.1, NumPy1.26.4, SciPy1.17.1, unchanged.
Clean pins: Fast-Graph-GP `268fff9a3724cdd480dce4b2946068d8211b4f30`;
authors' experiments `274b7ee393ee516585552f053e9d4a9c6ad2222f`.
The frozen48 PR17 endpoints, PR16 identities, input/checkpoint hashes and original
walk metadata remain intact. The only new walks are the original permitted tiny
cache:W1024/J3/halt.5/seed1671036/four workers; it was reused for this diagnostic.

The continuation contains its pre-run amendment, manifest/source snapshot, native
operator/RHS/solution/path/trace arrays, explanation JSON, CSV metrics, all
unattempted statuses, warnings, resource logs, tests and preservation hashes.
`native_audit` and `explain_native` keep numerical logic in the tracked driver;
local launcher/table scripts only handle metadata, resources and presentation.
Source snapshots distinguish the diagnostic execution from later saved-array
analysis; the original manifest was never overwritten. With retained local data:

```sh
cat results/matched_posterior_scaling/continuation/native_explanation.json
cat results/matched_posterior_scaling/continuation/diagnostic_metrics.csv
cat results/matched_posterior_scaling/continuation/summary.json
results/traffic_env/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

Execution used `caffeinate -i results/traffic_env/bin/python -u results/matched_posterior_scaling/continuation/launch_local.py diagnostic`.
Saved-array analysis used `study.explain_native()` under torch.no_grad(), float64,
four threads. Both refuse to overwrite evidence; further execution requires
planner review. A fresh clone lacks ignored checkpoints, walks and manifests.
Baseline/final suites:9/9 in both environments, with the same three existing
strict-CG NumericalWarnings per suite; py_compile and diff checks pass.
Changed: existing experiment/report only; deleted:nothing; kept:all prior source,
training, kernels and evidence. Open issue: planner decision on native identity
noise precision is required before any qualification or scaling continuation.
