# TB2 Overfit Investigation Notes

Current status: investigation in progress. This file records findings that are already backed by concrete artifacts so we do not lose track while digging further.

## Key Paths

- Training run root:
  `/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_overfit_gspo_lenreward_tb2of_raw_20260620_062054`
- Training rollout dumps:
  `/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_overfit_gspo_lenreward_tb2of_raw_20260620_062054/rollout_dumps`
- External eval run4:
  `/path/to/workspace/slime_harbor_trials/qwen35_27b_tb2of_iter19_v_full89_v512_daytona_run4/qwen35-27b-tb2of-iter19-v-full89-v512-daytona-run4`
- Analysis outputs:
  `.analysis/tb2of_rollout_analysis_full/`
  `.analysis/tb2of_positive_samples_hightrain_fail_decoded/`

## Scripts Added

- `scripts/harbor/analyze_rollout_dumps.py`
  - Reads Harbor rollout dump `.pt` files without importing Harbor runtime code.
  - Aggregates per-rollout and per-task sample counts, reward counts, status counts, response lengths, loss-token counts.
  - Can align rollout stats with an external Harbor eval run by reading `verifier/reward.txt`.
  - A v2 run is currently adding group-level reward distribution fields.

- `scripts/harbor/analyze_rollout_timeseries.py`
  - Builds per-task, per-rollout trend summaries from rollout dump analysis.
  - Used to identify tasks with large positive-rate drops over accepted rollout appearances.

- `scripts/harbor/analyze_rollout_task_mix.py`
  - Decomposes per-rollout reward into task-mix expected reward and residuals.
  - Used to test whether `harbor/reward_mean` drops are mostly task composition or within-task degradation.

- `scripts/harbor/extract_rollout_samples.py`
  - Extracts representative rollout samples for selected tasks.
  - Supports task filter, reward filter, per-task limit, and optional HF tokenizer decoding from `tokens[-response_length:]`.

- `scripts/harbor/summarize_extracted_rollout_text.py`
  - Reads extracted `samples.jsonl` files and summarizes commands, final completion claims, task-specific keywords, and weak/self-verification patterns.
  - Used for the `build-cython-ext` early-positive vs late-zero comparison.

- `scripts/harbor/summarize_largest_eigenval_samples.py`
  - Summarizes extracted `largest-eigenval` rollout samples by final `eigen.py` strategy and self-test behavior.
  - Used to compare early positive and late zero samples where the same scipy-based strategy can receive different rewards.

- `scripts/harbor/summarize_sqlite_truncate_samples.py`
  - Summarizes extracted `sqlite-db-truncate` samples by SQLite record-format reasoning, double decoding, and common wrong-value patterns.
  - Used to compare rollout 4 positive samples with rollout 15 zero-reward samples.

- `scripts/harbor/summarize_headless_terminal_samples.py`
  - Summarizes extracted `headless-terminal` samples by implementation strategy (`pexpect`, `pty`, `tmux`) and self-test coverage.
  - Used to compare high-positive training samples with the run4 failed trajectory.

- `scripts/harbor/extract_training_log_metrics.py`
  - Scans large Harbor/W&B output logs while skipping model-output noise such as `tmux_session.py Sending keys`.
  - Extracts rollout metrics (`harbor/reward_mean`, tail cancellations, discarded groups) and train-step health (`entropy_loss`, `pg_clipfrac`, `ppo_kl`).

- `scripts/harbor/summarize_training_metrics.py`
  - Summarizes extracted training metrics and accepted task coverage into
    `tb2_overfit_local_analysis/training_metrics/summary.md`.

These scripts were syntax-checked with `python -m py_compile`.

## Confirmed Findings

### 1. This Is Not a Simple "No Training Data" Failure

The 28 numeric training rollout dumps cover all 89 TB2 tasks:

```text
files 28
samples 10159
positives 3005
positive_rate 0.2958
tasks_seen 89
tasks_with_pos 62
tasks_zero_pos 27
status_counts {'completed': 9638, 'failed': 55, 'truncated': 466}
```

So the overfit run did generate successful trajectories for many tasks, but not for all tasks.

### 2. 27 Tasks Have Zero Positive Training Samples

These are very unlikely to improve from this RL run because the policy never saw a successful trajectory for them in the collected training samples.

Largest zero-positive tasks by sample count:

```text
dna-assembly 274
make-mips-interpreter 262
path-tracing-reverse 250
path-tracing 225
protein-assembly 223
polyglot-rust-c 203
circuit-fibsqrt 199
regex-chess 198
schemelike-metacircular-eval 189
polyglot-c-py 170
make-doom-for-mips 165
feal-linear-cryptanalysis 151
db-wal-recovery 150
gcode-to-text 137
mteb-leaderboard 134
```

All 27 zero-positive tasks failed in run4. Their run4 episodes are long on average:

```text
zero_pos n=27, run4 pass=0, run4 fail=27
eval episodes mean=68.11, median=100
```

This class looks like "no useful success signal + long-loop behavior".

### 3. 28 run4 Failed Tasks Did Have Positive Training Samples

These are the more interesting regressions/instabilities: the training run found successful trajectories, but the served iter19 model failed the external single eval sample.

High-signal examples:

```text
headless-terminal       77/80 positive, run4 fail
configure-git-webserver 48/64 positive, run4 fail
qemu-startup            69/96 positive, run4 fail
sqlite-db-truncate      31/51 positive, run4 fail
regex-log               39/68 positive, run4 fail
bn-fit-modify           46/81 positive, run4 fail
largest-eigenval        55/99 positive, run4 fail
```

This means the issue is not only coverage. There is also sampling instability / premature completion / shallow self-verification.

### 4. Positive-Rate Buckets Strongly Predict run4

Task-level training positive rate aligns with run4, but not perfectly:

```text
=0 positive rate:        27 tasks, 0/27 pass
(0, 0.05]:                6 tasks, 0/6 pass
(0.05, 0.2]:              8 eval-present tasks, 0/8 pass
(0.2, 0.5]:              12 tasks, 5/12 pass
(0.5, 0.8]:              10 tasks, 4/10 pass
(0.8, 1.01]:             25 tasks, 24/25 pass
```

Interpretation: training positive rate is a useful indicator, but single-sample eval can still fail even with high training success.

### 5. High-Positive run4 Failures Often End Too Early

Manual eval trajectory inspection shows a repeated failure mode: the model runs shallow or incomplete self-tests, concludes success, then emits `task_complete=true`.

Examples:

- `configure-git-webserver`
  - Training positive sample actually tested push/deploy and confirmed `/var/www/html/hello.html`.
  - run4 failed after confirming nginx served a manually written `test.txt`, which does not validate git push deployment.

- `headless-terminal`
  - Training positive samples mention tests for Ctrl+C, bash startup files, imports, command execution.
  - run4 ended after shallow checks; verifier failed `send_non_interactive_command`, `send_interactive_command`, startup files, state persistence, background commands.

- `regex-log`
  - run4 self-tested an insufficient set of edge cases and produced a regex that selected the wrong date on verifier data.

- `largest-eigenval`
  - run4 failed only one speed case. This one looks partly like benchmark variance / marginal optimization rather than pure semantic failure.

### 6. Training Dump `response` Field Is Empty; Need Token Decode

The rollout dump samples have empty `sample["response"]` strings. The actual trajectory text is recoverable from:

```python
tokens[-response_length:]
```

with the Qwen tokenizer. The helper script `extract_rollout_samples.py --hf-checkpoint ...` now supports this decode path.

### 7. Internal Training Eval Metric Is Misleading Under Timeout

In `slime_plugins/harbor/rollout.py`, `_generate_eval_async` cancels pending eval tasks at `harbor_eval_total_timeout_sec`, but computes `eval/harbor` only over completed samples:

```python
final_samples = [sample for group in output_groups for sample in group]
metrics["harbor/eval_reward_mean"] = sum(float(sample.reward) for sample in final_samples) / len(final_samples)
return {"eval": {"rewards": [sample.reward for sample in final_samples], ...}}
```

The code comment says pending tasks are pass-rate failures, but the implementation drops them from the denominator. Therefore internal eval values like `eval/harbor=0.38636` are completed-subset scores, not full89 scores.

Concrete example: `eval_19.pt` has 44 completed samples and 17 positives, so `17/44 = 0.38636`. It is not comparable to external run4's `33/89`.

### 7.1. All-Zero Groups Produce No Useful GSPO Direction

The configured reward hook is `trajectory_grouped_length_shaped_rewards`. It groups by Harbor trajectory group, applies length shaping only to successful trajectories, then calls the same group normalization used by turn-grouped reward:

```python
mean = sum(rewards) / len(rewards)
advantages = [reward - mean for reward in rewards]
std = statistics.stdev(advantages) if len(advantages) > 1 else 0.0
return [advantage / (std + 1e-6) for advantage in advantages]
```

Then GSPO/GRPO uses that processed scalar as the return for every token in the trajectory:

```python
returns = get_grpo_returns(rewards, kl)
advantages = [r for r in returns]
```

So if a group is all failures, every reward is 0, the normalized advantage is all 0, and the group provides almost no policy-gradient direction. This is why the 137 all-zero groups over 27 tasks are effectively wasted for learning success behavior.

### 8. fully-async Rollout Is a Confirmed Overfit Confounder

`slime_plugins/harbor/fully_async_rollout.py` accepts first-completed groups until target group count is reached. Long-running groups remain pending and may be cancelled at shutdown.

Inside each task group, `_run_one_group` launches 20 sample trials and stops as soon as 16 trajectory units are converted:

```python
if _group_unit_count(args, output_group, stats) >= partial_min_units:
    stats["cancelled_tail_trials"] += len(running_tasks)
    for task in running_tasks:
        task.cancel()
```

The outer fully-async worker also keeps consuming from the data source in the background. When `_generate_rollout_async` later drains more completed groups than it needs, it discards the extras:

```python
if len(output_groups) >= target_groups:
    discarded_completed_groups += 1
    continue
```

This means discarded groups have already advanced the training data cursor, but
do not contribute gradients.

Concrete log-backed accounting from
`tb2_overfit_local_analysis/training_metrics/summary.md`:

```text
intended groups, 89 tasks x 5 epochs:       445
accepted train groups:                      448
completed/attempted groups consumed:        631
completed groups discarded after consume:   183
tail sample trials cancelled at cutoff:    2306
accepted train trajectories:               6965
discarded group fraction:                  0.290
tail cancellations / accepted trajectories:0.331
```

So the run did not cleanly train on "89 tasks x 5 epochs". It consumed roughly
631 task groups from the data source and trained only the first-completed 448.
The remaining 183 completed groups were thrown away after consuming task
coverage.

Accepted task coverage confirms the skew:

```text
appearances in accepted rollout dumps:
2 appearances:  6 tasks
3 appearances:  2 tasks
4 appearances: 17 tasks
5 appearances: 29 tasks
6 appearances: 28 tasks
7 appearances:  7 tasks
```

Examples:

```text
2 appearances: financial-document-processor, fix-code-vulnerability,
               fix-ocaml-gc, gpt2-codegolf, nginx-request-logging,
               openssl-selfsigned-cert
7 appearances: crack-7z-hash, password-recovery, pytorch-model-cli,
               adaptive-rejection-sampler, polyglot-c-py, polyglot-rust-c,
               code-from-image
```

This biases training toward faster/easier trajectories, especially with terminal tasks whose difficulty and wall time vary a lot. It is consistent with:

- zero-positive tasks getting many long failed trajectories,
- high-positive tasks becoming short but sometimes shallow,
- training rewarding successful short paths under length reward.

Actionable implication: for overfit diagnosis, the collector should be made
deterministic and coverage-preserving before interpreting reward degradation as
pure policy degradation.

### 8.1. `harbor/reward_mean` Is Mostly a Task-Mix Metric

The per-rollout `harbor/reward_mean` curve should not be interpreted as a clean
same-distribution learning curve. A task-mix decomposition shows that the
rollout-to-rollout reward changes are almost entirely explained by which tasks
the fully-async collector accepted in that rollout.

Method:

- Compute each task's global training positive rate across all accepted dumps.
- For each rollout, compute expected reward as the sample-weighted average of
  those global task rates for the tasks present in that rollout.
- Compare actual rollout reward with this task-mix expected reward.

Result:

```text
source: tb2_overfit_local_analysis/rollout_task_mix/summary.md

rollouts:                         28
actual mean:                      0.2994
expected task-mix mean:           0.2991
actual vs task-mix Pearson:       0.9899
residual min/max:                -0.0298 / +0.0352
actual vs zero-task fraction:    -0.6478
actual vs low-or-worse fraction: -0.8153
actual vs high-task fraction:    +0.8986
```

Lowest actual-reward rollouts are dominated by globally hard / zero-positive
tasks:

```text
rollout 19: actual 0.1550, expected 0.1531
  low-or-worse sample fraction 0.801
  high sample fraction         0.041
  top hard tasks: path-tracing, path-tracing-reverse, db-wal-recovery,
                  mteb-leaderboard, polyglot-c-py

rollout 23: actual 0.1726, expected 0.1762
  low-or-worse sample fraction 0.664
  high sample fraction         0.048
  top hard tasks: regex-chess, protein-assembly,
                  schemelike-metacircular-eval, video-processing
```

Highest actual-reward rollouts have many high-positive tasks:

```text
rollout 0: actual 0.5638, expected 0.5561
  high sample fraction 0.411

rollout 18: actual 0.3958, expected 0.3606
  high sample fraction 0.245
```

Interpretation: the apparent `harbor/reward_mean` decline is largely not
evidence that the same tasks are getting worse every epoch. It is mostly a
non-stationary sampling/composition artifact from accepted rollout groups.
This is made worse by fully-async first-completed selection and discarded
completed groups.

Actionable implications:

- Do not use raw `harbor/reward_mean` as the main overfit progress metric.
- Add a fixed full89 train-set eval metric with timeout failures counted in the
  denominator.
- Add per-task normalized reward curves, e.g. current task reward minus that
  task's moving/global baseline, then average across tasks.
- Log accepted task ids and task-mix expected reward directly in W&B so future
  reward curves can be de-mixed.
- For overfit debugging, force each epoch to contribute one group per task
  before starting the next epoch, or at least report missing/extra task
  appearances.

### 9. Group-Level Reward Signal Separates run4 Pass and Fail

Important accounting detail: rollout dump `samples` are retokenized chunks, not always complete trajectories. The v3 aggregation merges chunks with:

```text
(harbor_rollout_id, harbor_rollout_group_index, harbor_sample_rank)
```

and reports both sample/chunk-level and trajectory-level statistics.

Trajectory-level totals:

```text
total trajectories 6965
positive trajectories 2814
trajectory positive rate 0.404
```

The v3 aggregation uses `group_index` to reconstruct training groups and classifies each group by trajectory reward:

- `all_zero`: no positive samples in the group
- `mixed`: at least one positive and at least one zero/failed sample
- `all_pos`: every sample in the group is positive

Overall group distribution:

```text
total_groups 448
all_zero     163
mixed        210
all_pos       75
any_pos      285
```

By run4 outcome:

```text
run4_pass:
  tasks 33, groups 168
  trajectories 2621, positive trajectories 2170, trajectory positive rate 82.8%
  all_zero 1   (0.6%)
  mixed    94  (56.0%)
  all_pos  73  (43.5%)
  any_pos  167 (99.4%)

run4_fail:
  tasks 55, groups 274
  trajectories 4248, positive trajectories 623, trajectory positive rate 14.7%
  all_zero 162 (59.1%)
  mixed    110 (40.1%)
  all_pos  2   (0.7%)
  any_pos  112 (40.9%)
```

For `zero_pos` tasks:

```text
tasks 27, groups 137
trajectories 2113, positive trajectories 0
all_zero 137 (100%)
```

For tasks with training positives but run4 failure:

```text
tasks 28, groups 137
trajectories 2135, positive trajectories 623, trajectory positive rate 29.2%
all_zero 25  (18.2%)
mixed    110 (80.3%)
all_pos  2   (1.5%)
any_pos  112 (81.8%)
```

For high-positive but run4-failed tasks (`positive_rate > 0.5`):

```text
tasks 7, groups 33
trajectories 527, positive trajectories 357, trajectory positive rate 67.7%
all_zero 0
mixed    31 (93.9%)
all_pos  2  (6.1%)
```

Interpretation:

- The 27 zero-positive tasks have no useful group-level positive signal.
- run4-pass tasks almost always had at least one positive in every training group, and many all-positive groups.
- run4-fail tasks are dominated by all-zero groups unless they are in the "train positive but unstable" class.
- The high-positive but failed tasks are almost entirely mixed groups, meaning GSPO did have preference signal, but that signal did not make the served policy robust under single-sample eval.

High trajectory-positive but run4-failed tasks:

```text
headless-terminal        77/80 trajectories positive, run4 fail
configure-git-webserver  48/64 trajectories positive, run4 fail
qemu-startup             69/96 trajectories positive, run4 fail
sqlite-db-truncate       28/48 trajectories positive, run4 fail
regex-log                37/64 trajectories positive, run4 fail
bn-fit-modify            45/80 trajectories positive, run4 fail
largest-eigenval         53/95 trajectories positive, run4 fail
```

## Useful Commands

Full rollout aggregation:

```bash
python3 scripts/harbor/analyze_rollout_dumps.py \
  --dump-dir /path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_overfit_gspo_lenreward_tb2of_raw_20260620_062054/rollout_dumps \
  --compare-run run4=/path/to/workspace/slime_harbor_trials/qwen35_27b_tb2of_iter19_v_full89_v512_daytona_run4/qwen35-27b-tb2of-iter19-v-full89-v512-daytona-run4 \
  --output-dir .analysis/tb2of_rollout_analysis_full
```

Decode representative positive samples:

```bash
python3 scripts/harbor/extract_rollout_samples.py \
  --dump-dir /path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_overfit_gspo_lenreward_tb2of_raw_20260620_062054/rollout_dumps \
  --task configure-git-webserver,headless-terminal,regex-log,largest-eigenval \
  --reward positive \
  --limit-per-task 2 \
  --max-response-chars 10000 \
  --hf-checkpoint /path/to/models/Qwen3.5-27B \
  --output-dir .analysis/tb2of_positive_samples_hightrain_fail_decoded
```

## Current Hypotheses

1. The overfit run is not actually robustly overfitting all 89 tasks. It only finds positives for 62/89 tasks.
2. For zero-positive and low-positive tasks, the training signal is too sparse; more RL on the same pipeline may not help unless sampling/search changes.
3. For high-positive but run4-failed tasks, the policy learned a distribution with successful modes, but single-sample eval can land in shallow premature-completion modes.
4. Length shaping plus fully-async first-completed group collection may amplify short successful-looking trajectories and underrepresent long hard tasks.
5. Internal eval is currently not a trustworthy full89 progress signal because timeout-cancelled tasks are omitted from `eval/harbor`.

## Next Checks

- Finish v2 aggregation with group-level fields:
  `groups_all_zero`, `groups_mixed`, `groups_all_pos`, `group_size_mean`.
- Quantify how many task groups have zero advantage signal because all samples in a group are failures.
- Compare train/eval decoding settings:
  training `temperature=1.0`, per-response max 4096, max_turns 60;
  run4 `temperature=0.6`, max_tokens 8192, max_turns 100.
- Consider an external pass@N eval on iter19 for the 28 "train positive but run4 fail" tasks.
- Patch internal eval scoring so timed-out groups count as failures in `eval/harbor`.

## Recommended Next Experiments

### A. Fix Observability First

1. Patch `_generate_eval_async` so timeout-cancelled eval groups count as zero rewards in returned `eval["rewards"]`.
   - Current `eval/harbor` is completed-subset mean.
   - Needed so W&B eval is comparable to full89 external eval.

2. Add trajectory-level stats to routine rollout logging.
   - Current sample counts are retokenized chunks.
   - Track `num_trajectories`, `positive_trajectories`, `all_zero_groups`, `mixed_groups`, `all_positive_groups`.

3. Run external pass@N on the 28 tasks where training had positives but run4 failed.
   - If pass@N is high, the checkpoint has the skill but single-sample decoding is unstable.
   - If pass@N is low, training did not actually move the served policy enough.

### B. Separate Two Failure Classes

Class 1: zero-positive tasks.

- 27 tasks, 2113 trajectories, 0 positives.
- These groups produce all-zero advantages.
- More updates on the same all-zero data will not teach success.

Recommended changes:

- Use stronger exploration/search for these tasks before RL update:
  higher `n_samples_per_prompt`, task-specific retry budget, or best-of-N collection.
- Add partial reward if verifier exposes case-level signal.
- Add successful seeds/demonstrations for zero-positive tasks, then RL from there.
- For overfit debugging, consider excluding zero-positive tasks from the policy update until at least one positive exists, while still reporting them separately.

Class 2: train-positive but run4-failed tasks.

- 28 tasks, 2135 trajectories, 623 positive trajectories.
- High-positive examples fail by shallow self-verification / premature completion.

Recommended changes:

- Evaluate these tasks with multiple seeds and with training-like generation settings.
- Reduce or disable length reward (`HARBOR_LENGTH_REWARD_LAMBDA=0`) for one A/B run.
  The current length reward is bounded, but it still explicitly prefers shorter successful trajectories and may encourage the model to compress verification behavior.
- Add reward/penalty for parser-clean JSON if parser warnings remain common.
- Add task-completion gating reward: penalize `task_complete=true` when the final verifier fails, and mine the preceding failed final response as a negative completion pattern.

### C. Reduce fully-async Sampling Bias for Overfit Debugging

For a pure train-on-test overfit experiment, use a less biased collection mode:

- fixed per-task quotas per epoch,
- no first-completed group selection,
- no cancelling long-tail groups before they contribute,
- no discarding of completed groups after consuming them from the data source,
- separate reporting for task wall-time and positive rate.

The current fully-async setup is useful for throughput, but it is a confounder
for an overfit diagnosis because fast/easy groups are preferentially accepted.
The concrete run consumed 631 groups, accepted 448, and discarded 183 completed
groups; that is not a clean five-epoch overfit data stream.

## Time-Series Analysis

Generated with:

```bash
python3 scripts/harbor/analyze_rollout_timeseries.py \
  --dump-dir /path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_overfit_gspo_lenreward_tb2of_raw_20260620_062054/rollout_dumps \
  --output-dir .analysis/tb2of_rollout_timeseries
```

Outputs:

- `.analysis/tb2of_rollout_timeseries/rollout_timeseries.csv`
- `.analysis/tb2of_rollout_timeseries/task_rollout_summary.csv`
- `.analysis/tb2of_rollout_timeseries/task_trend_summary.csv`

### Global Rollout Rate Is Not Strictly Monotonic

Trajectory positive rate by rollout:

```text
0  154/256 0.602
1  117/256 0.457
2  120/256 0.469
3   87/256 0.340
4   86/256 0.336
5  126/256 0.492
6  124/256 0.484
7   95/256 0.371
8   84/256 0.328
9   92/256 0.359
10 123/256 0.480
11 105/256 0.410
12 108/237 0.456
13 104/252 0.413
14  88/254 0.346
15  74/248 0.298
16 104/226 0.460
17 116/242 0.479
18 135/253 0.534
19  56/253 0.221
20  73/222 0.329
21 111/248 0.448
22  76/256 0.297
23  49/218 0.225
24 120/247 0.486
25 101/245 0.412
26  96/248 0.387
27  90/244 0.369
```

Interpretation: rollout 0 is unusually high, but the rest is not a simple monotonic descent. There are deep dips at rollouts 19 and 23, and rebounds afterward. This means global `harbor/reward_mean` must be decomposed by task composition and per-task repeated appearances.

Across tasks with at least 3 appearances:

```text
tasks >=3 appearances: 83
median early2->late2 delta: 0.000
mean early2->late2 delta: -0.002
tasks with delta < -0.1: 5
tasks with delta >  0.1: 5
```

So broad catastrophic forgetting across all tasks is not supported by the rollout dumps. Instead, a small number of tasks degrade materially, while some improve.

Largest early-to-late drops:

```text
regex-log                       early2 0.750 -> late2 0.406, delta -0.344
build-cython-ext                early2 0.469 -> late2 0.281, delta -0.188
model-extraction-relu-logits    early2 0.188 -> late2 0.038, delta -0.149
cancel-async-tasks              early2 0.281 -> late2 0.156, delta -0.125
largest-eigenval                early2 0.594 -> late2 0.484, delta -0.110
```

Largest gains:

```text
qemu-alpine-ssh                 early2 0.094 -> late2 0.312, delta +0.219
count-dataset-tokens            early2 0.500 -> late2 0.688, delta +0.188
mcmc-sampling-stan              early2 0.750 -> late2 0.909, delta +0.159
mailman                         early2 0.156 -> late2 0.281, delta +0.125
tune-mjcf                       early2 0.438 -> late2 0.556, delta +0.118
```

## Case Study: regex-log

`regex-log` is the clearest per-task degradation:

```text
rollout 4:  11/16 positive, rate 0.688
rollout 15: 13/16 positive, rate 0.812
rollout 19:  6/16 positive, rate 0.375
rollout 22:  7/16 positive, rate 0.438
```

Decoded sample sets:

- Early positives:
  `.analysis/case_regex_log/r15_positive/samples.jsonl`
- Late negatives:
  `.analysis/case_regex_log/r22_zero/samples.jsonl`

Findings from 4 early positive samples:

```text
sample 1: 1 capturing group
sample 2: 1 capturing group
sample 3: 1 capturing group
sample 4: 1 capturing group
```

Findings from 6 late negative samples:

```text
sample 1: 3 capturing groups
sample 2: 1 capturing group, but simplified/incorrect date constraints
sample 3: 1 capturing group, but day pattern only allows 01-29 or [12][0-9]
sample 4: 3 capturing groups
sample 5: 4 capturing groups
sample 6: no final `cat /app/regex.txt` captured in clipped text
```

This is a concrete regression mode:

- The task requires a regex that works with Python `re.findall`.
- If the regex has more than one capturing group, `re.findall` returns tuples instead of date strings, failing the verifier even if the model's own simple tests look good.
- Early positives generally preserve exactly one outer capturing group and use non-capturing inner groups `(?:...)`.
- Late negatives often simplify the regex and introduce accidental capturing groups for month/day/year, or simplify date validity to `day <= 29`, missing verifier cases with days 30/31.

The model did not simply forget the task. It shifted toward shorter/simpler self-justified regexes that pass its own tests but violate an interface invariant: exactly one capturing group returning the full date.

## Case Study: build-cython-ext

`build-cython-ext` is the second-largest per-task drop:

```text
rollout 0:   7/16 positive, rate 0.438
rollout 5:   8/16 positive, rate 0.500
rollout 8:   8/16 positive, rate 0.500
rollout 20:  8/16 positive, rate 0.500
rollout 23:  4/16 positive, rate 0.250
rollout 27:  5/16 positive, rate 0.312
```

Decoded sample sets:

- Early positives:
  `.analysis/case_build_cython_ext/r5_positive/samples.jsonl`
- Late negatives, clipped:
  `.analysis/case_build_cython_ext/r23_zero/samples.jsonl`
- Late negatives, full decode:
  `tb2_overfit_case_outputs/build_cython_ext/r23_zero_full/samples.jsonl`
- Local summaries:
  `tb2_overfit_case_outputs/build_cython_ext/summary.md`
  `tb2_overfit_case_outputs/build_cython_ext/summary_late_zero_full.md`

External run4 failed on one verifier case:

```text
test_ccomplexity:
  cc.cython_higher_order_writhe(points, contributions, order)
  AttributeError: module 'numpy' has no attribute 'int'
```

So the served iter19 policy left `np.int` inside `pyknotid/spacecurves/ccomplexity.pyx`, even though its final answer claimed the package was compatible with NumPy 2.3.0.

Concrete late-zero behavior from full decoded samples:

```text
sample 1:
  cat pyknotid/spacecurves/ccomplexity.pyx shows:
    cdef long [:] indices = np.zeros(4, dtype=np.int)
  later residual searches use --include='*.py', so .pyx files are skipped
  final checks are upstream repo tests / README import path
  final response marks task_complete=true

sample 2:
  finds .pyx files and inspects ccomplexity.pyx
  still relies on README/import-style checks and final summary
  no verifier-equivalent call to cython_higher_order_writhe is captured

sample 3:
  same pattern: sees Cython files, then residual grep is mostly --include='*.py'
  ends after 18 upstream tests pass

sample 4:
  hits a runtime NumPy int failure, then fixes Python files / periodiccell.py
  does not perform a final .pyx residual grep for np.int

sample 5:
  upstream repo tests initially expose n.int in Python files
  fixes Python files, reruns 18 tests, marks complete
  ccomplexity.pyx np.int remains outside the tested path

sample 6:
  inspects ccomplexity.pyx early, then later grep commands again only cover *.py
  final verification is import/README/upstream tests
```

The important distinction is not "early knows the task, late does not". Early positive samples often also use weak tests, but they more often actually patch `ccomplexity.pyx` (`np.int -> np.intp` or `np.int64`) before ending. Late zero samples often observe the `.pyx` file but their residual search/patch loop narrows to Python files, leaving the Cython runtime failure intact.

This is a concrete regression mode:

- The model learns that "upstream 18 tests pass + README snippet runs + Cython modules import" is enough to complete the task.
- That self-test is weaker than the verifier. Importing `ccomplexity` does not call `cython_higher_order_writhe`, so `dtype=np.int` can remain latent until verifier time.
- The final response overstates what was fixed. It says all NumPy aliases were fixed, while command history shows residual greps often skipped `.pyx`.
- Length reward can make this worse because the shortest successful-looking path is to stop after upstream tests, rather than construct direct API tests for every named extension.

Actionable improvements from this case:

- Add a "residual bug scan" pattern to rollout prompting or training data for debugging tasks:
  after fixing a deprecated symbol, run a final search over all relevant source extensions (`*.py`, `*.pyx`, `*.pxd`) and inspect remaining hits.
- Encourage verifier-equivalent direct tests for named APIs. For this task, a robust self-test is the verifier's shape:
  construct small `points`, `contributions`, `order` arrays and call `ccomplexity.cython_higher_order_writhe(...)`.
- Consider reward shaping / negative mining for premature completion:
  failed trajectories that mark `task_complete=true` after only upstream tests should become explicit negative completion examples.
- For terminal coding tasks, log and possibly reward "coverage of task-specified files/APIs" rather than only final success. Here the task explicitly names `chelpers`, `ccomplexity`, and `cinvariants`; a final self-test should exercise all three, not merely import them.

## Case Study: model-extraction-relu-logits

`model-extraction-relu-logits` is one of the clearest examples where the
rollout reward can reinforce a brittle shortcut rather than the verifier-robust
solution.

Rollout trend:

```text
rollout 0:   4/16 positive, rate 0.250
rollout 8:   5/16 positive, rate 0.312
rollout 15:  8/16 positive, rate 0.500
rollout 19:  4/16 positive, rate 0.250
rollout 27:  1/16 positive, rate 0.062
```

Decoded sample sets:

- Early positives:
  `tb2_overfit_case_outputs/model_extraction_relu_logits/r19_positive/samples.jsonl`
- Late negatives:
  `tb2_overfit_case_outputs/model_extraction_relu_logits/r27_zero/samples.jsonl`
- Local summary:
  `tb2_overfit_case_outputs/model_extraction_relu_logits/summary.md`

External run4 failed on the hidden-width verifier:

```text
task_checksum: 631dace3d78532648086133764c24bfbfb5f3fad7cb720f223e335640959f826
reward: 0.0
n_episodes: 47
verifier failure:
  Failed to match rows: [4, 7, 10, 13, 16, 20, 21, 22, 23, 24]
  1 failed in 606.73s
```

The failure shape is important: the model usually recovers about 20 directions,
then the verifier uses a 30-row hidden layer. Rows `20..24` missing is a
fingerprint of "trained against the visible 20-row forward.py" rather than a
general extractor.

The task prompt says the shape of `A1` is unknown and the solution should query
`forward(x)`. But the starting environment exposes:

```text
A1 = np.random.randn(20, 10) * 0.1
```

The run4 trajectory repeatedly self-checks with:

```text
from forward import A1, b1, A2, b2, forward
Original A1 shape: (20, 10)
Stolen A1 shape: (20, 10)
```

Its final response explicitly says the task is complete because all 20 neuron
directions match the original `A1`. That is not verifier-equivalent, because
the hidden test rewrites `/app/forward.py` to:

```text
zA1 = np.random.randn(30, 10) * 0.3
```

A positive training trial with the same checksum also exposes a verifier
weakness:

```text
training trial: slime-r16-g404-s0-i8080
task_checksum: 631dace3d78532648086133764c24bfbfb5f3fad7cb720f223e335640959f826
reward: 1.0
verifier stdout:
  PASSED ../tests/test_outputs.py::test_stolen_matrix_matches
captured stderr during verifier:
  File "/app/steal.py", line 134, in <module>
    from forward import A1 as orig
  ImportError: cannot import name 'A1' from 'forward' (/app/forward.py)
```

The verifier uses:

```text
os.popen("python3 /app/steal.py").read()
stolen_path = Path("/app/stolen_A1.npy")
assert stolen_path.exists()
b = np.load("/app/stolen_A1.npy")
```

It does not delete `/app/stolen_A1.npy` before running `steal.py`, and it does
not check the exit code of `steal.py`. Therefore a script can save an output
file, then crash in a debug/self-check tail after the verifier rewrites
`forward.py`, and still be judged only by whatever file remains on disk. This
turns verifier-unsafe behavior into a positive RL sample.

Late zero samples show the same behavioral direction more strongly:

```text
late_zero sample 6 final claim:
  saved matrix with shape (20, 10)
  all 20 neurons correlated with the true A1 rows
  task_complete=true
```

This is a concrete reward-noise / shortcut-learning failure:

- The environment leaks the original `A1` and hidden width through `forward.py`.
- The model learns to validate against the leaked 20-row matrix, not against the
  black-box extraction contract.
- The verifier attempts to prevent that by rewriting `forward.py`, but its
  `os.popen` call ignores script failure and leaves stale or partially generated
  outputs available.
- Positive rollout samples can therefore contain code that is not a clean,
  reproducible solution under the verifier's own rewritten environment.
- Overfit RL can amplify this brittle mode: it is shorter and easier than the
  official kink/second-derivative style solution, but it fails when hidden width
  changes.

Actionable improvements from this case:

- Fix the verifier first:
  delete `/app/stolen_A1.npy` before invoking `/app/steal.py`, run it with
  `subprocess.run(..., check=True, timeout=...)`, and fail on nonzero exit.
- Make the hidden-width test explicit in validation:
  verify that `steal.py` succeeds after `forward.py` is rewritten and that the
  produced matrix covers every row of the 30-row hidden model.
- Remove or mask leaked parameters from the training environment when the task
  claims black-box access. At minimum, self-checking via `from forward import A1`
  should be impossible or treated as a negative signal.
- Add negative mining for trajectories whose final verification is only against
  the visible 20-row `A1`, especially those claiming completion with
  `stolen_A1.npy` shape `(20, 10)`.
- Prefer reward/prompt checks that ask for verifier-equivalent generality:
  "do not assume hidden width; keep scanning until no new kink directions appear"
  rather than "find 20 directions".

## Case Study: cancel-async-tasks

`cancel-async-tasks` is useful because it shows an internal rollout reward drop
even though the external run4 sample for this task passed.

Rollout trend:

```text
rollout 0:   5/16 positive, rate 0.312
rollout 8:   4/16 positive, rate 0.250
rollout 20:  4/16 positive, rate 0.250
rollout 23:  1/16 positive, rate 0.062
rollout 27:  4/16 positive, rate 0.250
```

The degradation is also shorter:

```text
rollout 0 mean:  7.06 turns, 2228 output tokens
rollout 27 mean: 5.38 turns, 1713 output tokens
```

Decoded sample sets:

- Early positives:
  `tb2_overfit_case_outputs/cancel_async_tasks/r0_positive/samples.jsonl`
- Late zero samples:
  `tb2_overfit_case_outputs/cancel_async_tasks/r23_zero/samples.jsonl`
  `tb2_overfit_case_outputs/cancel_async_tasks/r27_zero/samples.jsonl`
- Local summaries:
  `tb2_overfit_local_analysis/cancel_async_tasks/summary.md`
  `tb2_overfit_local_analysis/cancel_async_tasks/replay_quick.csv`

The verifier has six tests. The critical one is:

```text
test_tasks_cancel_above_max_concurrent:
  run test.py --n-tasks 3 --max-concurrent 2
  wait 500ms
  send SIGINT
  assert exactly 2 tasks started
  assert exactly 2 cleanups ran
```

The official solution uses `asyncio.TaskGroup`:

```text
async with asyncio.TaskGroup() as tg:
  for task in tasks:
    tg.create_task(task_with_semaphore(semaphore, task))
```

The model's late zero samples usually implement the simpler pattern:

```text
coroutines = [run_with_semaphore(task) for task in tasks]
await asyncio.gather(*coroutines)
```

This passes normal concurrency tests, and often passes cancellation when all
tasks are already running. It fails the queued-task edge case: when `n_tasks=3`
and `max_concurrent=2`, one coroutine is still waiting on the semaphore. On
SIGINT, the implementation often propagates cancellation before the two started
tasks have completed their `finally` cleanup.

I replayed the extracted variants against a local equivalent of the verifier
with shorter sleeps for speed:

```text
early r0 positives: 5/5 passed replay
late r23 zeros:     0/8 passed replay
late r27 zeros:     0/8 passed replay

failure buckets:
  late23: 7 cancel_above, 1 normal concurrent typo/bug
  late27: 8 cancel_above
```

Representative late failure:

```text
stdout:
  Task started.
  Task started.
stderr:
  traceback from asyncio cancellation path
missing:
  Cleaned up.
```

External run4 for this task passed:

```text
task: cancel-async-tasks__uTDgheM
reward: 1.0
n_episodes: 4
verifier: 6 passed in 13.75s
```

So this is not an external failure case. It is evidence for why training
`harbor/reward_mean` can decline during overfit: the rollout distribution shifts
toward shorter implementations that verify only import/normal concurrency and
then confidently stop. The edge-case verifier is sparse and easy to miss, so a
small change in self-test behavior produces a large reward drop.

Concrete regression mode:

- Early positives often use `gather(..., return_exceptions=True)` or explicit
  task creation; the replayed variants wait long enough for cleanup in the
  queued cancellation case.
- Late negatives mostly use plain `gather(*coroutines)` and rely on a verbal
  claim that asyncio cancellation "will run finally blocks".
- Late samples frequently test only import and normal concurrency. They rarely
  reproduce the verifier's subprocess `SIGINT` scenario with `n_tasks=3` and
  `max_concurrent=2`.
- The length reward / shorter-output pressure can make this worse: the shorter
  solution is to write the semaphore+gather pattern, run one normal concurrency
  test, and stop.

Actionable improvements from this case:

- Add a task-family self-test hint for async cancellation tasks:
  test the queued cancellation edge case, not just normal concurrency.
- Negative-mine trajectories that mark complete after import/concurrency tests
  but never run a subprocess/SIGINT test.
- Prefer structured concurrency (`asyncio.TaskGroup`) or explicit created tasks
  plus `gather(..., return_exceptions=True)` during cancellation cleanup.
- Add a verifier-derived unit-test summary to the prompt or rollout rubric for
  coding tasks with unusual edge cases. Here, the essential hidden requirement is
  "only started tasks should cleanup under SIGINT when some tasks are queued".

### 14. Case Study: `largest-eigenval` Is a Noisy/Misaligned Reward

This task is important because it is not primarily "the model forgot how to
solve the problem". Early positive and late zero samples often use the same
solution family, but the reward hinges on a microbenchmark with very small
margins.

Training rollout trend:

```text
rollout 2:   9/16 positive, rate 0.562
rollout 6:  10/16 positive, rate 0.625
rollout 10: 12/16 positive, rate 0.750
rollout 13:  7/16 positive, rate 0.438
rollout 25:  7/16 positive, rate 0.438
```

Decoded sample sets:

- Early positives:
  `tb2_overfit_case_outputs/largest_eigenval/r10_positive/samples.jsonl`
- Late zero samples:
  `tb2_overfit_case_outputs/largest_eigenval/r25_zero/samples.jsonl`
- Local summary:
  `tb2_overfit_local_analysis/largest_eigenval/summary.md`

External run4 failure:

```text
task: largest-eigenval__afrJ9tG
reward: 0.0
verifier: 26 passed, 1 failed
failed test: test_speedup[9]
failure: 0.000032 seconds/call > 0.000025 seconds/call
```

The hidden verifier tests every size from 2 through 10 and runs timing in an
isolated process:

```text
MAT_SIZES = list(range(2, 11))
with ProcessPoolExecutor(max_workers=1):
  time ref_solution first
  time candidate second
assert candidate_median < ref_median
```

But the public `/app/eval.py` observed in rollout samples only tests even
matrix sizes:

```text
2, 4, 6, 8, 10
```

The model repeatedly concludes success from that public eval, while the hidden
failure occurs on size 9.

Representative early positive strategy:

```python
from scipy.linalg import eig as scipy_eig

def find_dominant_eigenvalue_and_eigenvector(A):
    eigenvalues, eigenvectors = scipy_eig(
        A,
        left=False,
        right=True,
        check_finite=False,
    )
    idx = np.argmax(np.abs(eigenvalues))
    return eigenvalues[idx], eigenvectors[:, idx]
```

Representative late zero strategy:

```python
import scipy.linalg

def find_dominant_eigenvalue_and_eigenvector(A):
    eigenvalues, eigenvectors = scipy.linalg.eig(
        A,
        overwrite_a=True,
        check_finite=False,
    )
    idx = np.argmax(np.abs(eigenvalues))
    return eigenvalues[idx], eigenvectors[:, idx]
```

The difference is not a clean semantic boundary. Both are scipy eig wrappers
with small optimization flags. Late zeros frequently:

- run the public eval,
- do not run hidden-style process-isolated timing,
- do not mention odd-size coverage,
- claim consistent speedup across "2x2 to 10x10",
- mark `task_complete=true`.

This creates noisy labels for RL:

- A trajectory can look correct and pass the visible self-test but fail one
  hidden odd-size timing case.
- The pass/fail boundary is a few microseconds, so hardware noise and process
  isolation can flip the label.
- Positive samples teach "use scipy eig with flags"; zero samples often contain
  the same lesson, so the policy gradient is not learning a stable distinction.

Actionable improvements from this case:

- Align visible/public self-tests with hidden verifier shape: test all sizes
  2..10, not only even sizes.
- For benchmark tasks, self-test with the same isolated-process timing harness
  and the same ref-first/candidate-second ordering.
- Require a margin before allowing `task_complete=true`, for example candidate
  median at least 10-20% faster than reference across repeated runs. A bare
  `dt < ref_dt` at 20-30 microseconds is too fragile.
- Consider reward smoothing for timing tasks: use ratios or repeated medians
  instead of a single binary strict comparison.
- Down-weight or filter microbenchmark tasks during RL unless the reward is made
  stable. Otherwise the model receives contradictory labels for nearly identical
  solutions.

## Case Study: sqlite-db-truncate Fails SQLite Record Type Decoding

This task is a useful "positive training exists but eval still fails" example.
It also shows how fully-async coverage skew can leave an important task
under-trained.

Training trend:

```text
accepted appearances: 3 groups, not the intended 5
rollout 4:  10/16 positive trajectories
rollout 8:  10/16 positive trajectories
rollout 15:  8/16 positive trajectories
run4 eval: reward 0
```

External run4 failure:

```text
trial: sqlite-db-truncate__xzcCvyA
verifier: 6 matched rows, assertion requires score > 6
model output for testword08/testword09: 64 and 63
expected hidden rows include:
  testword08 = 99.99
  testword09 = 0.5
```

The run4 trajectory found the raw `testwordXX` bytes but treated the first byte
after each word as the value. That works for the 1-byte integer rows:

```text
testword02 = 10
testword03 = 25
testword04 = 42
testword05 = 50
testword06 = 75
testword07 = 99
```

It fails for the float rows. In the SQLite record payload, `testword09` is
followed by:

```text
3f e0 00 00 00 00 00 00
```

That is big-endian IEEE-754 double `0.5`, not integer `63`. Similarly,
`testword08` is followed by `40 58 ff 5c 28 f5 c2 8f`, which decodes to
`99.99`, not integer `64`.

Training-sample comparison:

```text
source: tb2_overfit_local_analysis/sqlite_db_truncate/summary.md

r4_positive, 8 samples:
  uses varint/serial types:       7/8
  decodes big-endian double:      8/8
  has correct 99.99 and 0.5 pair: 8/8
  single-byte-after-word pattern: 2/8
  response length mean:           30018 tokens

r15_zero, 8 samples:
  uses varint/serial types:       5/8
  decodes big-endian double:      1/8
  has correct 99.99 and 0.5 pair: 1/8
  single-byte-after-word pattern: 5/8
  wrong 64/63 pair:               5/8
  wrong 8/9 pair:                 1/8
  response length mean:           21041 tokens
```

This is not a missing-data task. The early positive trajectories learned the
right abstraction: parse SQLite B-tree leaf cells, read varints / serial types,
and decode serial type 7 as an 8-byte big-endian float. The later zero
trajectories often fall back to a visually plausible but wrong shortcut:
`value = byte_after("testwordNN")`.

Actionable improvements from this case:

- Add a verifier-aligned self-check snippet or prompt hint for SQLite record
  format: serial type 1 is 1-byte integer, serial type 7 is 8-byte float64,
  and odd serial types >=13 are text with `(type - 13) / 2` bytes.
- Negative-mine trajectories that recover `testword08=64` and
  `testword09=63`; those are high-confidence wrong solutions and should not be
  reinforced by shallow completion language.
- Require self-verification to compare the recovered JSON against the known
  visible float signatures from the raw bytes, not just "I found 9 or 10
  records".
- For overfit debugging, give this task deterministic coverage. The accepted
  rollout dumps contain only 3 groups for this task, even though the run was
  supposed to cover 5 epochs. That makes it hard to interpret later degradation
  as model learning rather than async sampling bias.
- Consider a small task-specific reward diagnostic for partial rows: report
  which expected `testwordXX` rows matched, so we can distinguish "found bytes
  but decoded type wrong" from "did not locate records".

## Case Study: headless-terminal Is High-Positive but Still Fails Single Eval

`headless-terminal` is the opposite of the zero-positive tasks: training found
many successful trajectories, but run4 still sampled a bad implementation.

Training/eval summary:

```text
training accepted groups: 5
training trajectories:    80
training positives:       77
training positive rate:   0.9625
run4 eval:                reward 0
run4 episodes:            14
```

Run4 verifier result:

```text
pytest tests: 7
passed:       2
failed:       5

passed:
  test_import
  test_cancel_command

failed:
  test_send_non_interactive_command
  test_send_interactive_command
  test_startup_files
  test_shell_state_persists_between_commands
  test_background_commands
```

The final run4 implementation used a hand-rolled PTY stack:

```text
pty.openpty()
os.fork()
dup slave fd to stdin/stdout/stderr
os.setsid()
tty.setraw(0)
exec bash -l
```

The model's own final checks were shallow. It verified import, inheritance,
`pwd`, and Ctrl+C-like behavior, then marked complete. It did not run a
verifier-aligned test for:

- splitting command text and Enter across two `send_keystrokes` calls,
- `vim` insert / escape / `:wq`,
- appending to `.bashrc` before constructing `HeadlessTerminal`,
- preserving shell state across commands,
- keeping a background HTTP server alive.

The terminal snapshot even shows a warning sign: its final `pwd` self-test
prints the typed `pwd` and `SUCCESS`, but not the expected `/app` output. The
verifier then confirms the shell is not actually executing ordinary commands in
the required way.

Training sample comparison:

```text
source: tb2_overfit_local_analysis/headless_terminal/summary.md

r1_positive, 8 samples:
  uses pexpect:                 5/8
  uses pty.openpty:             3/8
  uses pty.fork:                1/8
  uses bash -i:                 6/8
  uses bash -l / login shell:   3/8
  runs official hidden tests:   0/8
  self-tests Ctrl+C:            8/8
  self-tests startup files:     8/8
  self-tests background server: 1/8
  claims task_complete:         8/8

r25_zero, 1 sample:
  uses pty.fork:                1/1
  uses bash -i and login shell: 1/1
  claims task_complete:         1/1

run4 trajectory:
  manual openpty/fork/setsid/raw stack: 1
  uses TIOCSCTTY / controlling tty fix: 0
  runs verifier-aligned tests:          0
```

This task does not explain the global training reward drop, because its training
reward stayed high. It does explain why overfit eval can still be worse than
base under single sampling: the model still assigns nontrivial probability to a
bad but plausible PTY implementation. With only one eval sample, that bad mode
can be selected even though most training rollouts pass.

Actionable improvements from this case:

- Add a task-specific negative-mining rule for `pty.openpty + os.fork +
  os.setsid + tty.setraw` implementations that do not set a controlling TTY
  with `TIOCSCTTY` and do not run verifier-aligned behavior checks.
- Prefer robust implementation modes in positive data: `tmux`, `pexpect.spawn`
  with interactive bash, or a tested PTY subprocess pattern. Hand-rolled raw
  PTY code is fragile.
- Add a reusable self-test scaffold to the task prompt or rollout policy for
  this class of terminal task:
  `send_keystrokes("echo ... > file")`, then `send_keystrokes("\n")`;
  `vim` edit; `.bashrc` append before instantiation; exported variable
  persistence; background `http.server`; Ctrl+C cancellation.
- For overfit measurement, report both task-level positive probability and
  single-sample pass rate. A 77/80 train positive task can still fail a single
  sample if the remaining bad mode is not driven close to zero.
- Consider eval temperature / sampling settings separately from training. This
  task is a clear candidate where lower eval stochasticity or best-of-N would
  improve measured overfit, but the better training fix is to suppress the bad
  PTY mode via hard negatives.
