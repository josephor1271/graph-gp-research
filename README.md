# A readable graph heat GP

Question: can a sparse Chebyshev heat covariance replace a dense heat kernel in
the same ordinary GPyTorch ExactGP, with checked values and heat-time gradients?

This is a small research example, not a toolbox or a performance study.
For an unweighted undirected grid, use `L = (D - A) / maximum_degree` with
spectrum in `[0, 2]`. This scalar scaling is different from symmetric degree
normalization. The covariance is `exp(-tau L)`, amplitude one, zero mean, and
Gaussian observation **variance** 0.04. All calculations use CPU float64;
trainable heat time is restricted to `(0.05, 4)`.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python demo.py
```

`dense_gp.py` contains the ordinary ExactGP and a small-graph eigendecomposition
reference. `demo.py` constructs a 6 × 6 grid, draws an exact synthetic GP at
tau=1.4 with seed 12, fits heat time from 24 observations, and predicts the 12
held-out vertices. It writes one figure and its inputs under ignored `results/`.
Fitting one draw is a usage demonstration, not a parameter-recovery guarantee.

Initial stages: (1) dense GP, (2) differentiable Chebyshev recurrence,
(3) sparse GPyTorch integration. Each stage is tested and committed separately.

Stage 1 validation (Python 3.12.0, pinned dependencies): both independent dense
tests passed; `demo.py` reduced negative MLL from 0.83903702 to 0.60669030 in
40 updates, fitted tau=2.85602691, and obtained latent held-out RMSE 0.56974583.
No tests failed. The first run emitted a Torch sparse-invariant warning; graph
construction now explicitly enables invariant checks.
