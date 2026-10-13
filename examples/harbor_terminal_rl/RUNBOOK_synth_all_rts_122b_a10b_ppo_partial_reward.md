# RUNBOOK — Qwen3.5-122B-A10B / RTS-37484 / PPO + CTRF DENSE (partial) reward @ 88 GPU

> One-line purpose: run the **validated 88cp2 PPO recipe** unchanged, but replace the
> binary terminal reward with a **CTRF dense / partial-credit reward**, on the
> **Recursive-Task-Synthesis** 37484-task pool.

---

## §0 TL;DR

```bash
cd /path/to/workspace/slime-terminal-RL

# 0) one-time: unpack the data (see §2). Then dry-run:
DRY_RUN=1 examples/harbor_terminal_rl/run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest_partial_reward.sh

# 1) THE ACTUAL RUN as launched 2026-08-16 (R3 + no-save-optim + critic warm-up 5, 3 epochs):
R=/path/to/shared_storage/hf_models/Qwen/RST_models
RUN_TAG="r3_$(date +%Y%m%d_%H%M%S)" \
NUM_ROLLOUT=441 NUM_CRITIC_ONLY_STEPS=5 SAVE_INTERVAL=10 \
HF_CKPT="$R/Qwen3.5-122B-A10B" PRETRAINED_CKPT="$R/Qwen3.5-122B-A10B_torch_dist" \
WANDB_GROUP="q35-122b-a10b-RST-synthRTS37k-88cp2-ppo-partialReward-R3-criticwarm5-3ep" \
SAVE_SUBDIR="Qwen3.5-122B-A10B_RST_synthRTS37k_88cp2_ppo_partialreward_R3" \
examples/harbor_terminal_rl/run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest_partial_reward.sh
# R3 and --no-save-optim are ON BY DEFAULT (USE_ROUTING_REPLAY=1 / NO_SAVE_OPTIM=1).

# 2) watch
examples/harbor_terminal_rl/synth_all_rts_122b_partial_reward_health_check.sh
examples/harbor_terminal_rl/r3_disk_watchdog.sh      # ← the binding constraint, see §3.8

# 3) stop
ray job list && ray job stop <job_id>
```

> The base model for this run is **`RST_models/Qwen3.5-122B-A10B`** (its own HF + torch_dist
> pair, 145 shards / 1949 weight entries, marker `release`, `.metadata` present so Megatron
> reshards to PP4 at load). Its `config.json` is content-identical to the stock 122B
> (48 layers, vocab 248320, 256 experts, full_attention_interval 4), so the 88cp2 memory
> model applies unchanged.


---

## §1 What changed, and what deliberately did NOT

This run is a **2-variable delta** against
`run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest.sh` (`TOPOLOGY=88cp2`).

| | tmax-15k baseline | THIS run |
|---|---|---|
| **Reward** | binary terminal (0/1) | **CTRF dense: `passed_tests / 20`** |
| **Data** | tmax-15k, 14601 tasks | **RTS synth pool, 37484 tasks** |
| shuffle | off | **on** (`rollout_shuffle: true`) |
| topology | 88cp2: 64 train + 24 infer | *identical* |
| parallelism | TP2 × PP4 × CP2 ⇒ DP4, EP16 | *identical* |
| batch | bs256 / rollout 256 / oversample 280 | *identical* |
| context wall | 64k (SEQ_LENGTH 65536) | *identical* |
| PPO | eps-clip 0.2, KL off, entropy off | *identical* |
| critic | cold start, `NUM_CRITIC_ONLY_STEPS=2` | *identical* |
| lr | critic 30x (1.5e-5) / actor 2x (1.0e-6) | *identical* (same yaml) |
| memory | `MAX_TOKENS_PER_GPU` 8192, mem-frac 0.6, `max_split_size_mb:256` | *identical* |
| CP2 anti-straggler | pad-mult 2048, Triton cache, 120min timeouts | *identical* |

The wrapper is deliberately **thin**: it sets no topology / PPO / memory knob at all, so
those cannot drift. It only sets the reward hook, the data config, and run identity, then
`exec`s the parent.

**Files added by this migration** (nothing existing was modified except `partial_reward.py`,
which gained functions only):

| File | Role |
|---|---|
| `slime_plugins/harbor/partial_reward.py` | +4 reward variants (ported from v3; pure superset — 0 lines removed, 492 added) |
| `tests/test_harbor_efficiency_reward.py` | 6 CPU unit tests |
| `tests/test_harbor_length_discount_reward.py` | 7 CPU unit tests |
| `tests/test_harbor_length_penalty_reward.py` | 6 CPU unit tests |
| `examples/.../harbor_synth_all_rts_122b_a10b_daytona_ppo_cp2_bs256_partial_reward.yaml` | data config (fork of the cp2 yaml, 3 keys changed) |
| `examples/.../run_..._strongest_partial_reward.sh` | launcher (thin wrapper + pre-flight) |
| `examples/.../synth_all_rts_122b_partial_reward_health_check.sh` | read-only health probe |

**Deliberately NOT ported from v3**: `train_loss.py` (v3's copy omits `group_mask_sums`
and is a normalizer bug — this repo's version is the correct one), `rollout.py` eval
dedup, `llm.py` sampling params, the 27B-shaped yaml/launch/lr files. None relate to
partial reward; porting them would inject unrelated risk.

---

## §2 Prerequisites

### 2.1 Data — unpack Recursive-Task-Synthesis (one-time, CPU-only, ~3.4 GB)

```bash
SRC=/path/to/workspace/harbor_synth_all_20260725/Recursive-Task-Synthesis
OUT=/path/to/workspace/harbor_synth_all_20260725/dataset_rts
mkdir -p "$OUT"
for t in "$SRC"/data/tasks-*.tar; do tar xf "$t" -C "$OUT" --strip-components=1; done
python3 -c "from pathlib import Path;print(len(list(Path('$OUT').glob('*/task.toml'))))"   # expect 37484
```

Verified before extraction: all 8 shards **sha256-match** `metadata/shard_manifest.jsonl`;
`metadata/tasks.parquet` has **37484 rows, 100% unique `task_id` (0 collisions),
`validation_status == "passed"` for every row** — so all shards extract into one flat
directory safely. `--strip-components=1` drops the leading `tasks/` member prefix.

> ⚠ **Never point the config at the sibling `dataset/` directory.** Those 38573 entries
> are symlinks into `/path/to/shared_storage` (2820) and `/path/to/shared_storage` (35753), and **neither
> mount exists on this cluster** — measured 0 of 300 sampled entries resolve a
> `task.toml`, so `glob` returns 0 tasks and the data source fails at init. The launcher
> pre-flight catches this.

### 2.2 Other assets

- **Eval pool**: `/path/to/shared_storage/.../terminal-bench-2-main` (89 tasks) — already present.
- **Model**: `Qwen3.5-122B-A10B` HF + pre-converted `_torch_dist` (PP-agnostic: `release/.metadata`
  present, so Megatron reshards freely to PP4 at load; no reconversion needed).
- **CP2 requirement**: `pip install -U 'fla-core==0.4.2' 'flash-linear-attention==0.4.2'`
  on **every node** (site-packages is node-local). The parent script fail-fasts on
  `import fla.ops.cp`.
- **Credentials**: `/tmp/bs512_keys.env` (mode 0600) with `DAYTONA_API_KEY`, `WANDB_API_KEY`.
- **Cluster**: long-lived Ray head. `CLEAN_START=0` and `MASTER_ADDR=29.127.33.153` are
  inherited — **never set `CLEAN_START=1`** on this box; it kills the head and drops all workers.

---

## §3 The reward mechanism (the point of this run)

### 3.1 Formula

`slime_plugins.harbor.partial_reward.test_count_progress_rewards`:

```
shaped = passed_tests / HARBOR_TESTCOUNT_SCALE          # SCALE = 20
```

The **ABSOLUTE passed-test count**, not the `passed/total` ratio. The ratio weights every
task equally at max 1.0, which flattens the signal for hard many-test tasks; the absolute
count lets them earn proportionally more. A fully-passed task scores `total/SCALE`, which
**can exceed 1.0** for tasks with >20 tests — intended; PPO's critic+GAE absorbs the scale.
(Measured max on real trials: **2.0** for a 40-test task.)

Under `--advantage-estimator ppo`, `_should_normalize()` is False, so raw shaped rewards
flow **straight into critic + GAE with no group normalization**.

### 3.2 Why it works on THIS pool (measured, not assumed)

Each RTS task's `tests/test.sh` runs:

```bash
uvx -w pytest-json-ctrf pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA
```

so per-test pass/fail lands in `verifier/ctrf.json`, which is exactly what
`_load_progress_score()` parses.

- **4888/5000 = 97.8%** of shard-0 `test.sh` contain `--ctrf`
- tests per task: 4 (166) / 5 (53) / 6 (35) / 7 (29) / 8–9 (46) / 10+ (~60), **max 33**
  → `passed` carries a real gradient rather than a 0/1 outcome
- Validated offline against **40 real trials** from the previous 27B run on this same
  pool: 40/40 ctrf parsed, **30 strictly-partial** (0<passed<total), 17 distinct reward
  values, range 0.0–2.0

> ⚠ **The tmax-15k pool does NOT emit ctrf.** Its `test.sh` writes only
> `verifier/reward.json` (binary + `test__<name>` fields). Point this wrapper at tmax-15k
> and every trajectory falls back to `terminal_reward` — the dense reward becomes a
> **silent no-op** (no error, no warning). This is why the config is pinned to the RTS
> pool and why the health check probes the CTRF hit rate (§6.2).

### 3.3 Reward-variant A/B table

All four v3 variants are available; override `CUSTOM_REWARD_POST_PROCESS_PATH`:

| Variant | Shape | Env knobs |
|---|---|---|
| `test_count_progress_rewards` **(default)** | `passed/SCALE` | `HARBOR_TESTCOUNT_SCALE=20` |
| `test_count_efficiency_rewards` | + step-efficiency bonus on solved trajs (run#5) | `HARBOR_EFFICIENCY_BONUS_BETA=0.05` |
| `test_count_length_penalty_rewards` | − `λ·clamp(out_tokens/BUDGET)` (run#6) | `HARBOR_LENGTH_PENALTY_LAMBDA=0.03`, `HARBOR_LENGTH_TOKEN_BUDGET=40000` |
| `test_count_length_discount_rewards` | `×(1 − α·turns_norm)` + zero-pass penalty (run#7) | `HARBOR_LENGTH_DISCOUNT_{FREE_TURNS=25,FULL_TURNS=50,ALPHA=0.5,ZERO_BETA=0.1}` |
| `trajectory_grouped_progress_rewards` | old ratio: `max(terminal, 0.4·passed/total)` | `HARBOR_PARTIAL_REWARD_MAX=0.4` |

**v3's recorded history**: run#6 (additive length penalty) **failed** — `avg_turns` went
22 → 50+, because a λ kept below one test's value was noise next to the correctness
gradient (more turns → more tests brushed → higher `passed/SCALE`), so length-farming won.
run#7 fixes it multiplicatively so length erodes correctness itself. If you A/B length
shaping, prefer run#7 and watch `avg_turns`.

### 3.4 On `--normalize-advantages` (advantage whitening) — NOT enabled

v3's RUNBOOK §14 claims the 38k run used whitening, but `NORMALIZE_ADVANTAGES` appears
**nowhere** in v3's starter script (grep count 0) — it was never wired to
`--normalize-advantages`, so that run did **not** actually whiten. Keeping it off
preserves "algorithm identical to the parent".

To A/B it on, you **must repeat the parent's own value** or you clobber the DeepEP-off
dispatcher setting:

```bash
EXTRA_MISC_ARGS="--moe-token-dispatcher-type alltoall --normalize-advantages"
```

---

## §3.5 R3 (rollout routing replay) — `--use-rollout-routing-replay`

Enabled by default in this wrapper via `USE_ROUTING_REPLAY=1`.

**What it does.** MoE expert selection during SGLang rollout and during the training-side
log-prob recomputation can diverge, which makes the PPO ratio wrong. R3 (arxiv 2510.11370)
captures the routing indices at rollout time and *replays* them in the actor forward.
`arguments.py:1924` also flips `use_routing_replay=True` as a side effect.

**Measured effect** (r3probe vs the earlier non-R3 smoke, same model/data/topology):
`train/train_rollout_logprob_abs_diff` **0.0171 → 0.0125 (−27%)** — the train/rollout
mismatch really does shrink.

**Two things that look like bugs but are not:**
- `train/kl_loss` / actor-vs-ref KL is **no longer ~0**. `model.py:805-813` and
  `data.py:375-385` explicitly *disable* their KL≈0 assertions when R3 is on, because the
  actor replays rollout routing while the reference forward routes normally. Expected.
- The word `placeholder` appears in the log. Almost always this is the **agent** writing
  the literal string into a shell command (assembly source, checksum files). Grep for the
  framework messages instead:
  ```bash
  grep -acE 'Missing routed_experts|rollout_routed_experts is required|RolloutConversionError' "$LOG"   # must be 0
  ```

**The one real failure mode.** R3 requires routing on EVERY turn:
`harbor/rollout.py:1202` passes `require_routing_replay=True` → `atif_adapter.py:317`
raises `RolloutConversionError` on a missing row → and because this config sets
`harbor_fail_on_conversion_error: true`, `rollout.py:1204-1206` **re-raises and takes the
whole rollout down** (it does not skip the sample). A second net exists on the train side:
`actor.py:310-313` raises `ValueError` if `rollout_routed_experts` is absent from
`rollout_data`. Verified clean on a real rollout: **285 trials converted, 0 routing errors.**

Token/routing lengths stay in sync through the 64k hard trim: `_routing_tail_slice` keeps
`keep-1` rows because routing rows are always `len(tokens)-1`
(`sample_postprocess.py:298`), and `tests/test_harbor_sample_postprocess.py` asserts it.

**Why `EXTRA_MISC_ARGS` is assembled by the script, not by you.** Neither
`--use-rollout-routing-replay` nor `--no-save-optim` has an env knob in the base launcher,
so both can only arrive through `EXTRA_MISC_ARGS` — which the parent *overwrites* rather
than appends. Passing it by hand has historically dropped
`--moe-token-dispatcher-type alltoall` (DeepEP-off) or lost R3 silently. The wrapper now
merges the flags idempotently and prints the result as `[preflight] misc = ...`. Verify
that line, and the `use_rollout_routing_replay`/`use_routing_replay`/`no_save_optim`
entries in the arg dump, after every launch.

## §3.6 `--no-save-optim` — smaller checkpoints, NOT resumable

`NO_SAVE_OPTIM=1` (default here). Checkpoints carry weights only.

**Measured:** one save point is **455 GB** (actor 228 + critic 227) versus **3.2 TB** with
optimizer state — a ~7× reduction.

Safe for the PPO critic: `placement_group.py:196-224` already warm-loads the critic
weights-only by default (`no_load_optim=True`), because loading a source run's optimizer
makes `OptimizerParamScheduler.load_state_dict` assert on an iteration-count mismatch.

**The cost, stated plainly** (from the flag's own help): it *"disables training resumption
from the saved checkpoint."* These checkpoints are for eval / fresh cold-start only. If
the run dies at step 300, you cannot resume — you restart. Combined with the disk ceiling
below, that is the main operational risk of a long run.

## §3.7 Critic warm-up: `NUM_CRITIC_ONLY_STEPS=5`

`rollout_id < 5` trains the **critic only** (`train_async.py:75,85`); the actor joins at
`rollout_id 5`. Raised from the parent's 2 because the dense reward's range is wider than
binary (`passed/SCALE` can exceed 1.0 — measured max 2.0 on a 40-test task), so the
cold-start value head needs longer before its advantages should be trusted to steer the
actor. Verified in the smoke run: steps 0-1 logged only `critic-*` metrics, and
`train/pg_loss` first appeared exactly at the first joint step.

## §3.8 ⚠️ DISK CEILING — the binding constraint on a 441-step run

Two things grow without bound, and slime has **no checkpoint retention/pruning knob**
(every save point is kept forever):

| item | measured | over 441 steps |
|---|---|---|
| checkpoint, weights-only | 455 GB per save | `SAVE_INTERVAL=10` → 44 saves ≈ **20 TB** |
| trial dirs | 92 GB per rollout | 441 rollouts ≈ **40 TB** |

Amortised: **~138 GB per step**. At launch only **2.3 TB** was free on `/path/to/shared_storage`
≈ **17 steps**, so space MUST be freed/extended during the run. Because
`--no-save-optim` makes checkpoints non-resumable, a disk-full ckpt-save failure loses the
entire run rather than a few steps.

Monitor with:
```bash
examples/harbor_terminal_rl/r3_disk_watchdog.sh    # prints remaining-steps headroom
```
Note `/path/to/shared_storage` also holds an unrelated **10 TB** `..._88cp4-84k_bs256_20260813_004233`
tmax15k run (3 ckpts up to iter_69, with optimizer). It is not part of this experiment —
do not delete it without checking who owns it.



Trial dirs are an **output**, not an input: each rollout writes
`<trials_dir>/slime-r{rollout}-g{group}-s{rank}-i{index}/verifier/ctrf.json` and the
reward function reads it back moments later. **Nothing needs pre-populating — you do not
need to "rollout first".** An empty directory is correct.

The hazard is name determinism (`slime_plugins/harbor/rollout.py::_trial_name`):

```python
return f"slime-r{rollout_id}-g{group_index}-s{sample_rank}-i{base_sample.index}"
```

— **no run tag, no timestamp**. If `trials_dir` pointed at the previous 27B run's tree,
this run's `slime-r0-g0-s0-i0` would collide name-for-name. In that old dir only
**47/400** sampled trials have a `ctrf.json` (353 verifier dirs are empty), so whenever a
new trial fails to emit its own ctrf (verifier timeout / agent failure — common), the
reward would **silently read the 27B model's leftover ctrf** and score this run on another
model's results.

Mitigation, implemented in the launcher:
- `trials_dir` is rewritten into a materialized config at
  `${BASE_FOLDER}/harbor_config_resolved.yaml`, scoped by `${RUN_TAG}`
  (the established `sed`-the-template pattern used by other launchers in this repo);
- pre-flight **refuses** paths matching the old 27B trees, and refuses a non-empty
  `trials_dir` unless `ALLOW_DIRTY_TRIALS_DIR=1`.

The old 27B tree remains useful — but **only as a read-only regression fixture** (§7 L2).

---

## §5 Schedule & budget

- 1 epoch = `ceil(37484 / 256)` = **147 steps**; 3 epochs = **441**.
- `NUM_ROLLOUT` is **not** set by the wrapper — it inherits the parent's **100**, so the
  wrapper changes nothing implicitly. Pass `NUM_ROLLOUT=441` for the full 3 epochs.
- `SAVE_INTERVAL=10`, `EVAL_INTERVAL=10`, eval = Terminal-Bench-2 pass@3, eval total
  timeout 10800 s (3 h). Optimizer state IS saved (`--no-save-optim` is never passed) →
  checkpoints are resumable.
- bs256 with DP4 = **64 grad-accum micro-batches per step** — deliberately slow
  (correctness over speed).

---

## §6 Post-launch checklist

### 6.1 First things to confirm

1. `[preflight]` block shows the reward path, task count (37484), `shuffle = true`,
   and a fresh `trials`/`config` path.
2. `[topology]` line reads:
   `train=64 (TP2 x PP4 x CP2 => DP4, EP16), 12 layers/stage, infer=24 (3 x TP8), nodes=11`
3. Log contains `--custom-reward-post-process-path slime_plugins.harbor.partial_reward.test_count_progress_rewards`
4. `grep -ac 'successfully loaded checkpoint'` → 3 (actor + critic)
5. `grep -acE 'CUDA out of memory|torch.OutOfMemoryError'` → **must stay 0**
6. **`rollout_id 2` is the first actor+critic joint step** (`NUM_CRITIC_ONLY_STEPS=2`).
   Steps 0–1 are critic-only. Clearing step 2 is the real validation — that is where
   smaller topologies OOM'd.

### 6.2 Confirm the dense reward is actually firing

This is the one failure mode that is **invisible in loss curves**: if `ctrf.json` is
missing, every trajectory silently falls back to the binary terminal reward and the dense
reward becomes a no-op — no error, no warning.

**Where to look — and where NOT to.** `harbor_progress_*` lands in `sample.metadata` /
`sample.train_metadata`, and `_convert_samples_to_train_data` does copy it to
`train_data["metadata"]`. But `_split_train_data_by_dp` (`slime/ray/rollout.py:1073-1087`)
packages each DP rank's `rollout_data` from an explicit **key whitelist that does not
include `"metadata"`**. So the progress fields never reach the megatron-side
`log_rollout_data`, and **they will NOT appear as wandb metrics**. Do not go looking for a
`harbor_progress_source` chart.

(Aside: that whitelist is also what makes the port safe. `log_rollout_data`'s type
dispatch would hit `sum(val)` on a list of dicts and raise
`TypeError: unsupported operand type(s) for +: 'int' and 'dict'`. Verified by simulating
the branch logic — the whitelist is what prevents it. Anyone who later adds `"metadata"`
to that list will crash the first joint actor step.)

So verify via these three instead:

1. **The health check's `ctrf_probe` line** (the primary, purpose-built signal):
   ```
   ctrf_probe: 47/52 finished trials have ctrf.json (90%) | dense reward: n=47 mean=0.213 max=2.000 strictly-partial=30
   ```
   Escalates automatically if <50% of finished trials emit ctrf.
2. **`rollout/rewards` in wandb** — with dense reward this must be **continuous**
   (multiples of 0.05 at SCALE=20, and values >1.0 for >20-test tasks). If it collapses
   to only `{0.0, 1.0}`, the dense reward has degenerated to binary.
3. **`rollout_dumps/*.pt`** — the debug dump is taken *before* the DP split, so it retains
   full `sample.metadata`. To inspect `harbor_progress_source` / `passed` / `total`
   directly, load a dump and read the metadata.

If the CTRF hit rate is low, stop and check the task pool before burning more GPU-hours.

### 6.3 Normal-but-alarming behaviours (do NOT kill the job for these)

- **GPU util 0% during rollout** — normal; agents are solving on remote Daytona sandboxes.
- **CP2 Triton cold-compile storm on the first steps** — per-microbatch times of 400s+
  decaying to ~20s is the JIT warm-up curve, not a hang. This is exactly why
  `DATA_PAD_SIZE_MULTIPLIER=2048`, a persistent node-local `TRITON_CACHE_DIR`, and
  120-minute NCCL/new-group timeouts are set. The health check's `STALL_MIN` defaults
  to 45 min here for this reason.
- **`RuntimeError` / `Traceback` in the log** — the Terminus2 agent runs arbitrary
  terminal commands; its output routinely contains those words. The health check strips
  agent chatter before crash-grepping. Judge crashes only on framework-level hammers
  **combined with** stalled artifact mtime.

---

## §7 Verification performed before launch (all reproducible)

**L1 — unit tests (seconds)**
```bash
python -m pytest tests/test_harbor_partial_reward.py \
                 tests/test_harbor_efficiency_reward.py \
                 tests/test_harbor_length_discount_reward.py \
                 tests/test_harbor_length_penalty_reward.py -v     # 24 passed
python -m pytest tests/test_harbor_turn_grouped_reward.py \
                 tests/test_harbor_length_shaped_reward.py -q      # 9 passed (no regression)
```

> Known, **pre-existing** and unrelated: `tests/test_harbor_mismatch_metrics.py` fails 4
> tests when the whole `tests/test_harbor_*.py` set runs together, because 15 other
> Harbor tests install a minimal `torch` stub in `sys.modules`. It passes **4/4 in
> isolation**, and fails identically with the ported files removed. Not caused by this
> migration.

**L2 — real-data fixture, zero GPU** (the highest-value check)
Build `Sample`s pointing `harbor_trial_uri` at real trial dirs from the old 27B run and
call the reward directly. Asserted: `reward == passed/SCALE` on all 40 ctrf trials;
PPO branch returns `raw == processed`; partial signal present; and for run#7,
farmed (50 turns) reward is **exactly 0.5×** free-zone (20 turns) reward on all 36 solved
trials, with zero-pass trajectories at −0.1 (the BETA penalty).

**L3 — data + config pre-flight, zero GPU**
sha256-verify the 8 shards, confirm 37484 resolvable `*/task.toml`, then
`DRY_RUN=1 ...run_..._strongest_partial_reward.sh` (prints the resolved plan and exits
without touching the cluster).

**Not yet done**: the actual 88-GPU launch (§0 step 1) — held pending human go-ahead.

---

## §8 Rollback

- Reward: `CUSTOM_REWARD_POST_PROCESS_PATH=""` restores the binary terminal reward
  (the parent's default is empty).
- Code: `partial_reward.py` gained functions only; the original is preserved at
  `/tmp/partial_reward_port_backup/partial_reward.py.target_orig`. Reverting it does not
  affect any other run script, since existing scripts reference
  `trajectory_grouped_progress_rewards` / `post_process_rewards`, both unchanged.
- Data: point `harbor_tasks_dir` back at `slime-terminal-RL_data/tmax15k_terminal_bench`
  (and remember the dense reward then degenerates to binary — §3.2).
