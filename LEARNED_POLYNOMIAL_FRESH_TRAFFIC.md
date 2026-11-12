# Frozen fresh-date PEMS-BAY validation

On five frozen new dates and a new split, unchanged Poly improves RMSE over Heat and the C seed mean on all five dates,
and observation NLPD on four. Against GeneralGRF's seed mean, RMSE wins on three dates and NLPD wins on three.
The earlier three-date dominance does not fully repeat.

Frozen before pilot/scoring: filtered Monday17:30 indices [2,6,10,14,18], Jan16/Feb13/Mar13/Apr10/May8 2017. Same
1016-node/1173-edge weighted symmetric normalized Laplacian and ordered 325-sensor map; HDF selected by sensor ID. New
250/75 split uses default_rng(20261112).permutation(325), shared by all dates/models; disjoint IDs and difference from
PR6 verified. Only training labels determine each date’s mean/std. No replacement dates, test-based selection or tuning.

Reuse PR12 model_for unchanged: Heat Cheb20; Poly q(0)=1, degree2 filter/squared-degree4 covariance; C/General J3,
halt.5, W4096, new seeds 303/404. Each date retrains from scratch. Initial beta=1.462117157, signal SD=1.313261688,
noise variance=.693247181; Poly a1=-.196611933241482, a2=.133611661347131; General=sigma*[1,-beta/2,beta²/8]. Heat/C raw
beta/signal/noise=1/1/0. CPU float64, four Torch threads/four sampler workers; fresh RNG0 per fit, official AdamW
.05/default decay .01, exactly 1000 updates, Cholesky800. Poly/General have four parameters including noise; Heat/C have
three.

Nine tests pass in both environments; existing strict-CG test warnings remain. Reused six-vertex traffic dense checks
give max covariance/NLL/raw-gradient errors 8.88e-16/2.22e-16/2.22e-16. The blind Jan16 Poly20 pilot passed (finite
losses/gradients, .492358s training, parent peak693.33MiB), without test scores; full Poly restarted. All 30 full fits
and scoring completed; zero failed, replaced or pending cases. Logs confirm all fits preceded all scoring. Full-graph
operator densification is guarded; existing off-timer graph-spectrum validation remains.

Each cell below is RMSE mph / marginal observation NLPD in train-standardized units; lower is better. All arms use exact
own-kernel 325-sensor posterior scoring, with noise once. GRF uses its finite learned kernel. NLL/obs, complete
parameters, all 30000 losses, initial states and timings stay in ignored CSV/JSON/checkpoints.

|Date (2017)|Heat|Poly|C303|C404|General303|General404|
|---|---:|---:|---:|---:|---:|---:|
|01-16|9.108460 / 1.351700|8.085515 / 1.421203|9.513626 / 1.361805|9.058724 / 1.296387|8.525101 / 1.227548|8.333663 / 1.393589|
|02-13|15.666226 / 1.339404|12.768223 / 1.154768|15.626753 / 1.355444|15.719136 / 1.367631|13.372310 / 1.241375|13.380557 / 1.253015|
|03-13|15.949414 / 1.339923|14.845971 / 1.270654|15.861848 / 1.338412|15.788820 / 1.339497|14.782025 / 1.262421|14.634219 / 1.275875|
|04-10|16.358627 / 1.383221|13.313248 / 1.188547|16.521102 / 1.411939|16.621268 / 1.420333|14.106397 / 1.312137|14.025621 / 1.294846|
|05-08|16.130451 / 1.373850|14.644124 / 1.281332|16.116958 / 1.375139|16.266074 / 1.389697|14.499777 / 1.285660|14.719261 / 1.291181|

Paired differences below are Poly minus comparator; entries are RMSE / NLPD. C/General comparisons average the two per-
seed scores. The five dates are the comparison units; ties: zero.

|Date/summary|vs Heat|vs C seed mean|vs General seed mean|
|---|---:|---:|---:|
|01-16|-1.022945 / +0.069503|-1.200660 / +0.092107|-0.343867 / +0.110634|
|02-13|-2.898003 / -0.184636|-2.904721 / -0.206769|-0.608210 / -0.092427|
|03-13|-1.103443 / -0.069269|-0.979363 / -0.068300|+0.137849 / +0.001507|
|04-10|-3.045379 / -0.194674|-3.257937 / -0.227589|-0.752761 / -0.114945|
|05-08|-1.486326 / -0.092519|-1.547391 / -0.101086|+0.034605 / -0.007089|
|mean|-1.911219 / -0.094319|-1.978015 / -0.102327|-0.306477 / -0.020464|
|median|-1.486326 / -0.092519|-1.547391 / -0.101086|-0.343867 / -0.007089|
|range|[-3.045379, -1.022945] / [-0.194674, +0.069503]|[-3.257937, -0.979363] / [-0.227589, +0.092107]|[-0.752761, +0.137849] / [-0.114945, +0.110634]|

RMSE/NLPD win counts: Heat 5/5 and 4/5; C mean 5/5 and 4/5; General mean 3/5 and 3/5. Jan16 Poly NLPD is worse than
every comparator despite better RMSE. Against General mean, March13 loses both metrics; May8 loses RMSE by .034605mph
but improves NLPD by .007089.

No convergence certificate: March13 Poly raw NLL gradient norm=.105450 and last100 recorded-loss change=+.000138136. Its
final NLL/obs=1.349381 versus last recorded optimizer loss1.348786, evaluated before the final update. General
March13/404 and Apr10/303 also have positive last100 loss changes (+7.41e-7/+3.55e-5). Every final update is retained;
no best-checkpoint choice or extra optimization. Raw norms depend on coordinates and AdamW decay. Heat beta
remains1.995079–1.996108 near its frozen upper bound; achieved NLL and model capacity differ.

Poly a1 spans[-.698556,-.292900], a2[.401261,.524966], and noise variance[.003746,.044197]. Full signal and modulation
states remain in ignored artifacts.

Measured seconds below are min–max across the listed fits. Walk construction is one-shot and shared across dates and
C/General: seed303=2.010266s,404=1.964445s. Sum of full training=518.436735s; full parent peak RSS688.92MiB. Phase
timers exclude data validation, checkpoint writes and final gradient audits; official training/progress overhead is
included. No deployment timing comparison.

|Arm/count|Setup s|Train1000 s|Oracle score s|
|---|---:|---:|---:|
|Heat/5|0.000792–0.172936|30.847001–33.117631|0.018191–0.024506|
|Poly/5|0.000718–0.001107|9.181147–10.085443|0.004397–0.006060|
|C/10|0.000183–0.000203|14.805362–15.972025|0.013887–0.016476|
|General/10|0.000235–0.000290|14.584497–15.771168|0.013673–0.018744|

Separate local NumPy checks reconstruct all 30 RMSE/NLPD values from saved posterior moments; max errors
3.55e-15/4.44e-16. This post-run check is in local analysis_command.txt, not implemented in the tracked driver. All
hashes/initializations/1000-loss histories validate. Verified resume skips 30 cases without refitting/scoring and
preserves artifacts. Sparse invariant/CSR UserWarnings are retained; no experiment NumericalWarning.

This is method confirmation within PEMS-BAY: spatial interpolation on the same graph with date-specific retraining. Five
related Mondays and one shared split do not supply independent weekly replications, forecasting, parameter transfer,
other-graph evidence or 10K–50K training-size evidence. Date and split changes are not separately identified. No
p-values or universal-superiority claim.

Reproduce from repository root using the existing pinned traffic environment and PR6 caches. Preserve original logs;
existing outputs verify/resume. Archive the output directory before a fresh rerun.

```sh
results/traffic_env/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
results/traffic_env/bin/python -u experiment_learned_polynomial_fresh.py check
caffeinate -i results/traffic_env/bin/python -u experiment_learned_polynomial_fresh.py pilot
caffeinate -i results/traffic_env/bin/python -u experiment_learned_polynomial_fresh.py full
git diff --check
```

Base main e6e4a795aefc06c88fb8ca395fa73079fa095b61 includes merged #12; no dependency on #9/#11. Pins:
core268fff9a3724cdd480dce4b2946068d8211b4f30; experiments274b7ee393ee516585552f053e9d4a9c6ad2222f. Original HDF/L/map
fingerprints match PR6.

New data SHA256: `d14c1e5a45a094987f0c9409ca949bd90de27b6beac69d3a6a4eba187b457e91`.
Frozen manifest SHA256: `e519bf4e4e414478ead9368010aa7ee08cb4e4a55446dfb88841765d4c2447e0`.
Driver SHA256: `03c9d2aa1be8cee02a852b94f32a7b60b48e00317b51fa501cd363cb91787003`.

Added: 217-line driver and this report. The <=230 allowance covers frozen inputs, verified resume and separate
fit/scoring phases. Deleted: nothing. Kept: all existing code, defaults, dependencies and read-only toolbox. Raw
artifacts/commands stay ignored under results/traffic/learned_polynomial_fresh/. No extra experiment or figure; leave PR
unmerged.
