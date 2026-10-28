# Readable graph GP research

Answer one scientific question at a time. Read relevant code and callers first;
keep the mathematics visible in ordinary Python. Reuse the single ExactGP model.
Do not add frameworks, factories, workers, generic configuration, reporting,
packaging, or speculative features. Explain and obtain approval for new dependencies
or architecture. The initial work order authorizes dense, recurrence, and sparse
integration commits; stop after the working comparison demo.

Use L = (D - A) / maximum_degree for unweighted undirected graphs, spectrum [0, 2].
State dtype, heat-time range, polynomial degree, and observation noise variance.
Check numerical values and gradients against an independent dense reference.
Separate polynomial error from iterative/stochastic inference error. Keep warnings
visible; never clamp a covariance to manufacture correctness or claim unmeasured
speedups. Keep generated outputs and environments local and ignored.

Subsequent changes should usually fit 200 added lines and five files, with one
coherent purpose per tested commit. Explain larger changes before implementation.
Review the final diff for unused abstractions and incidental edits. Report actual
commands, results, failures, caveats, changed/deleted/kept files, and open issues.
The old graph-gp-toolbox repository is a read-only archive.
