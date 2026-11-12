# Appendix C.2: sparse GRF ring scaling

Partial C.2 replication: storage and costs of the official sparse GRF on rings with 1024–65536 vertices.

## Frozen sources and protocol

Read grf.pdf §4.1/C.2/Tables1–4, pp7,14–16; hash in ignored provenance.json. Sources:
[paper](https://arxiv.org/abs/2509.03691),
[scaling.py](https://github.com/MatthewZhang473/Graph-Random-Features-for-Scalable-Gaussian-Processes/blob/274b7ee393ee516585552f053e9d4a9c6ad2222f/experiments/scaling/scaling.py),
[core](https://github.com/MatthewZhang473/Fast-Graph-GP/tree/268fff9a3724cdd480dce4b2946068d8211b4f30).

|Topic|Paper C.2|Pinned scaling code|Executed protocol|
|---|---|---|---|
|Sizes / seeds|2^5…2^20 / five|lines37–38: 64…1024 / three|1024,4096,16384,32768,65536 / seeds0…4 target|
|Noise|Variance .1|line40: SD .1, variance .01|Variance .1, SD sqrt(.1)|
|Walks / halt / cutoff|100 / .1 / 3 hops|lines41–43: 1000 / .1 / 5|100 / .1 / API argument3; powers0…2, two contributing hops|
|Signal / split|Smooth periodic; fraction unspecified|lines66–96: explicit sine/cosine, 60% observed|Same signal and RNG order; 60% train, remaining40% test; train-only sample SD|
|Learning|50 epochs; §3 names Adam|lines44–45 + utils.py: AdamW .05, 100 iterations, default decay .01|Official train_model: 50 updates, AdamW .05, decay .01|
|Inference|Posterior mean and covariance|lines145–149: predict(batch_size=64), read mean|Same 64-path Monte Carlo mean readout; internal diagonal sample variance included; no full covariance timing|
|Representation|Sparse arm up to1M|line69 constructs dense adjacency|Direct sparse cycle adjacency; no large dense graph/kernel/observed block|

Signal: 2sin(2t)+.5cos(4t)+.3sin(t), t=2πi/N. Input is **adjacency A**, not L. Argument3 stores powers0…2 and
consumes a final unrecorded hop. Effective seeds: 0→42, others1…4; node RNG adds worker index and start node.
Scaling.py:55 **calls** set_gp_defaults. Scoped: Cholesky0, CG .01, 64 probes, Lanczos1, no preconditioner,
exact root-decomposition setting, maxCG1000. GeneralGRF learns three coefficients. All float64 CPU, four
threads/four workers, 64 predictive paths. Upstream storage counts the three CSR bases' values/row pointers/columns,
not materialized Phi, RSS or posterior covariance. Guard overhead is timed.

Base: approved PR #10 merge `8c7cede6f79a4dd3924c09b75c470c4bf6ccea71`; PR #9 untouched.

## Gates, runs and results

Tiny N=32 passes sparse/dense adjacency and identical official-walk checks, selected covariance/Kv, independent small
Cholesky NLL/raw gradients (errors 0 / 6.94e-18). Finite walk-base errors versus A^0/A^1/A^2: 0/.087621/.127306; these
are sampling errors. Tiny authors settings: NLL error +.239186/obs; noise/modulation gradient errors .004190/.013104;
true residual .007391. Pilots 1024/4096, seed0: 50 updates each, 2.211/5.902 s training; both count below. Frozen
allocation: all25 cases, two hours, 1800 s/case alarm, 12 GiB parent high-water ceiling checked after phases. No
failure, timeout, missing case or forbidden dense allocation. Measured case times sum to 987.804 s (excludes CLI
imports/final JSON write; includes audits/data IO). Parent/child peak RSS 1.023300/0.114273 GiB are separate high-water
marks, not aggregate memory. All 1250 losses and data/source fingerprints validate. No NumericalWarning; sparse
UserWarnings are retained. Eight tests pass in both environments before/after; existing strict-CG test warnings persist.

Mean ± sample SD, five seeds. Phases are separate; training includes optimizer construction; guards are timed.

|N (train/test)|Done|Walk bases MiB|Prep / model setup s|Walk s|Train50 s|Mean readout s|
|---|---:|---:|---|---:|---:|---:|
|1024 (614/410)|5/5|0.117210|0.377±0.052 / 0.001339±0.001466|0.123±0.005|2.062±0.152|0.034±0.002|
|4096 (2457/1639)|5/5|0.468773|0.355±0.003 / 0.000720±0.000065|0.474±0.033|5.496±0.417|0.095±0.002|
|16384 (9830/6554)|5/5|1.875023|0.433±0.068 / 0.000768±0.000028|1.851±0.068|21.852±2.275|0.418±0.005|
|32768 (19660/13108)|5/5|3.750023|0.459±0.057 / 0.000793±0.000031|3.983±0.119|45.643±4.726|0.799±0.020|
|65536 (39321/26215)|5/5|7.500023|0.368±0.003 / 0.000892±0.000058|8.943±0.811|98.626±12.107|1.723±0.131|

Raw per-seed seconds and stochastic NLL/obs; full-precision phases remain in `results.csv`.

|N/seed|Walk s|Train s|Mean s|Case s|Stoch NLL|True residual|
|---|---:|---:|---:|---:|---:|---:|
|1024/0|0.116007|2.211070|0.031300|2.877930|1.146353|0.007510|
|1024/1|0.122272|1.970913|0.035089|2.537089|1.066096|0.007208|
|1024/2|0.128299|1.933175|0.033910|2.493335|1.067658|0.008729|
|1024/3|0.120989|1.951793|0.035925|2.509618|1.075023|0.009424|
|1024/4|0.128917|2.243457|0.034578|2.811832|1.190457|0.008374|
|4096/0|0.531495|5.902044|0.093189|6.989639|1.148011|0.006767|
|4096/1|0.450164|5.236766|0.097094|6.253252|1.070555|0.008960|
|4096/2|0.462262|5.160328|0.095750|6.195428|1.079292|0.009250|
|4096/3|0.459434|5.183137|0.096683|6.212995|1.086306|0.006982|
|4096/4|0.467497|5.999954|0.093507|7.029092|1.177062|0.008183|
|16384/0|1.768982|24.173051|0.423261|27.158001|1.151796|0.007939|
|16384/1|1.869070|20.064050|0.422239|23.283189|1.077196|0.009665|
|16384/2|1.815879|20.134598|0.418585|23.163384|1.081134|0.009074|
|16384/3|1.848566|20.387085|0.410942|23.558997|1.077958|0.009906|
|16384/4|1.953101|24.502941|0.414813|27.809237|1.181661|0.009129|
|32768/0|3.794350|50.174729|0.789709|56.066068|1.157503|0.009217|
|32768/1|3.944477|43.417724|0.793843|49.519092|1.078999|0.009529|
|32768/2|4.024663|41.462708|0.786394|47.632938|1.073488|0.009015|
|32768/3|4.059807|41.856038|0.834460|48.142218|1.077120|0.007369|
|32768/4|4.092841|51.302042|0.788334|57.568416|1.179987|0.009098|
|65536/0|7.993425|101.415119|1.587051|113.081960|1.154716|0.008716|
|65536/1|8.449554|87.021351|1.591508|99.385976|1.074151|0.009379|
|65536/2|8.713845|92.786887|1.751473|105.574942|1.072733|0.008990|
|65536/3|9.653494|93.658227|1.886666|107.546611|1.077263|0.007153|
|65536/4|9.906586|118.249080|1.796980|132.403184|1.178996|0.008779|

Storage is exactly 6N nonzeros / 120N+24 bytes in every trial. Cost point estimates are slightly superlinear for
walks/training and near-linear for mean readout; no speedup claim. Exploratory OLS on log(size means),
N=16384/32768/65536, three points / one residual df:

|Cost|Exponent b|Model-based 95% OLS interval|
|---|---:|---|
|Bases|0.999993|[0.999977, 1.000010]|
|Walks|1.136211|[0.911208, 1.361214]|
|Train50|1.087091|[0.907386, 1.266796]|
|Mean readout|1.021618|[0.378758, 1.664479]|

The paper fits sparse N≥32768; our cap supplies only two such points. These are **not** its fit protocol/CIs or
asymptotic guarantees. Sequential timings and node-based overlapping RNG streams limit independence. CPU/guard timings
cannot reproduce RTX2080Ti absolute times. Inference is 64-path MC mean readout including internal diagonal variance,
not full covariance. Single-RHS endpoint residuals range .006767–.009906; actual iteration counts are not exposed. They
do not certify all training/MC solves. Lanczos1 bias and fit convergence remain unresolved.

## Reproduction and scope

Existing pinned traffic environment: Apple M4/16 GiB, Torch2.11.0/GPyTorch1.15.2; versions in provenance.json. Archive for fresh fits; verified rows resume.

```sh
mkdir -p results/paper_c2
results/traffic_env/bin/python -u experiment_grf_paper_c2_scaling.py check
for c2_n in 1024 4096 16384 32768 65536; do
  for c2_seed in 0 1 2 3 4; do
    caffeinate -i results/traffic_env/bin/python -u experiment_grf_paper_c2_scaling.py "$c2_n" "$c2_seed"
  done
done
results/traffic_env/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

Ignored `results/paper_c2/` retains raw cases/data/hashes, logs and provenance. Separate `analysis_command.txt` derives
aggregation/audits, fits and `scaling.png`; tracked script runs gates/trials. Added: 195-line script/report; deleted:
nothing. Core/defaults/deps/upstream unchanged. Dense/full-covariance/GPU unreplicated. No C.6/C.5 code; recommend C.6
only after review/new authorization. Leave PR unmerged.
