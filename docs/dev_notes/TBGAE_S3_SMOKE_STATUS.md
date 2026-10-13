# TB-GAE S3 Smoke Status — Final Update (2026-09-29 18:15)

## Three Smoke Runs — Summary

| Run | Log | Crash Cause | TB-GAE Verdict |
|---|---|---|---|
| Smoke 1 (`run_tbgae_s3_smoke.log`) | 19,537 lines | `RuntimeError` (broken tmux) → hit missing `TBGAETrialSkip` class | Found real bug |
| Smoke 2 (`run_tbgae_s3_smoke2.log`) | 2,709 lines | `DaytonaValidationError` → `TBGAETrialSkip` raised but `continue` missing in catch block | Found real bug |
| Smoke 3 (`run_tbgae_s3_smoke3.log`) | 79,976 lines | Daytona org CPU quota exhausted → 100% of 3,456 trials failed | Infrastructure outage, not TB-GAE |

## Bugs Found and Fixed in This Session

### Bug 1: `RuntimeError` from non-whitelisted exception crashes rollout (FIXED)
- **Root cause:** `tb_gae_data.py::trial_termination_kind` had no way to signal "skip this trial" vs "abort the rollout". Any non-whitelisted exception type raised `ValueError` which propagated up.
- **Fix:** Added `TBGAETrialSkip` exception class to `tb_gae_data.py`. Raised for infrastructure failures (broken env, non-whitelisted RuntimeError).
- **Files:** `slime_plugins/harbor/tb_gae_data.py` (new class + raise), `slime_plugins/harbor/rollout.py` (import + catch + `_is_tb_gae_skip` helper)

### Bug 2: `TBGAETrialSkip` caught but then hits empty-samples guard (FIXED)
- **Root cause:** The catch block set `samples = []` then fell through to `if enable_tb_gae: raise ValueError("TB-GAE trial produced no samples")`. The `continue` was missing.
- **Fix:** Restructured the entire `_run_one_group` exception handling loop — `TBGAETrialSkip` is caught first, increments `stats["failed"]`, logs a warning, and `continue`s immediately, bypassing both the empty-samples guard and the retry logic.
- **Robustness:** `_is_tb_gae_skip()` helper walks `__cause__`/`__context__` and `RayTaskError.cause` so Ray-wrapped exceptions are detected without depending on Ray reconstructing the custom type.

## What Was Confirmed Working

1. **TB-GAE args parsed correctly** — `enable_tb_gae=True`, `tb_gae_gamma_turn=1.0`, `tb_gae_lambda_turn=1.0`
2. **Config validates** — `validate_tb_gae_config` passes on production args
3. **Checkpoint loads** — RUN2 warm-critic iter263 loaded on both actor and critic
4. **v2 producer fires** — In smoke 1 (before Daytona died), 53 completed trials all had `termination_kind="terminated"`, `termination_metadata_version=2` ✓
5. **TBGAETrialSkip caught and dropped** — Smoke 3 correctly dropped 3,455 `DaytonaValidationError` failures without aborting mid-rollout. The fix works.
6. **No OOM, no NCCL errors** — Through all 3 smokes, no GPU-side failures.

## What Was NOT Tested (Still Needs a Real Smoke)

Because all three smokes were killed before training began (by crashes or Daytona outage), the following remain untested on GPU:
1. **Truncated trial with `bootstrap_token_ids`** — no trial hit max_turns and completed before the outage
2. **`compute_bootstrap_values` on GPU** — the extra old-critic 256k forward has never executed on real hardware
3. **Cross-DP boundary gather** (`tb_gae_boundary.py`) — not exercised
4. **CP=4 successor-value all_gather** — not exercised
5. **Step 0 actor + critic grad_norm** with TB-GAE advantages — no training metrics collected

## Next Step

Wait for Daytona quota to recover (the org hit 500-CPU limit), then relaunch smoke 4. The bug fixes from this session make the code more robust; Daytona availability is an external dependency.

**Command to relaunch when Daytona is back:**
```bash
cd /path/to/workspace/slime-terminal-RL
NUM_ROLLOUT=50 nohup bash examples/harbor_terminal_rl/launch_27b_tbgae_s3_smoke.sh > run_tbgae_s3_smoke4.log 2>&1 &
```

**What to watch after relaunch:**
- Smoke immediately kills itself on `Harbor rollout produced no trainable slime samples` → Daytona still down, wait longer
- Smoke survives past rollout 0 → TB-GAE GPU path is running; check step 0 `grad_norm` and `tb_bootstrap_values` in log


### 1. Producer (termination_kind + v2) — **CONFIRMED WORKING**
- Terminus-2 v2 is firing correctly in real trials
- 46 completed trials all show:
  - `termination_kind: "terminated"`
  - `termination_metadata_version: 2`
  - `bootstrap_token_ids: []` (empty for terminated, as expected)
- Location: `/path/to/workspace/slime_harbor_trials/q38_27b_ppo_256k_TBGAE_S3_SMOKE_20260929_162455/`

### 2. Config Parsing — **PASSED**
- TB-GAE args resolved correctly:
  - `enable_tb_gae: True`
  - `tb_gae_gamma_turn: 1.0`
  - `tb_gae_lambda_turn: 1.0`
- Role yaml `/path/to/workspace/slime-terminal-RL/examples/harbor_terminal_rl/megatron_roles_27b_tbgae_s3.yaml` accepted
- No import/validate errors

### 3. Checkpoint Load — **SUCCESSFUL**
- Actor: RUN2 loadroot (iter 263) ✓
- Critic: RUN2 loadroot_critic (iter 263, warm) ✓
- Both loaded successfully (log lines 2087–2258)

### 4. No Crashes — **CLEAN**
- 16K+ log lines, no OOM, no NCCL errors, no TB-specific exceptions
- `failed=0 / missing=0` across all completed trials

## What Was NOT Yet Verified (Rollout 0 Still Running)

### Criteria Still Pending (waiting for training to start):
1. **Truncated trial bootstrap_token_ids** — no trial hit max_turns yet (all 46 so far are `terminated`)
2. **Bootstrap forward** — training hasn't begun (still collecting rollout 0)
3. **Step 0 grad_norm (actor + critic)** — no training metrics yet
4. **Cross-DP bootstrap value gather** — not exercised until training
5. **Memory (256k bootstrap forward OOM check)** — not tested until training

## Why Rollout 0 Is Slow
- 256 trials × up to 3600s timeout each = potentially hours
- Only 46/~256 trials completed after 30+ minutes
- Mixed3000 tasks (terminal RL) are complex and slow
- This is expected for a full-batch first smoke

## Current Resource Constraints
- **Disk:** 1.9 TB free (98% full) — critically low
- **Risk:** Rollout 0 might not complete before filling disk with trial data
- **Smoke design:** NUM_ROLLOUT=50, SAVE_INTERVAL=1000 → no checkpoint saves, but trials still consume space

## Recommendation

**STOP the smoke now and declare partial success:**
- The v2 producer is proven working (46 real trials with correct metadata)
- Config/load/startup all green
- The remaining criteria (bootstrap forward, step 0 metrics) require a **shorter smoke**:
  - Reduce batch size (e.g., 32 trials instead of 256)
  - Or use simpler tasks that complete faster
  - Then full training metrics will appear within ~10 minutes instead of hours

**Next Steps:**
1. Stop the current smoke: `ray job stop raysubmit_ttGsbYjmsCyBEzmm`
2. Create a **mini-smoke** with `--rollout-batch-size 32 --over-sampling-batch-size 36`
3. Or proceed directly to a short production run (NUM_ROLLOUT=5) accepting that the bootstrap mechanics are CPU-validated + producer is GPU-proven

## Files Created This Session
- `examples/harbor_terminal_rl/megatron_roles_27b_tbgae_s3.yaml` — TB-GAE S3 role config
- `examples/harbor_terminal_rl/launch_27b_tbgae_s3_smoke.sh` — full-batch smoke launcher
- This status file

## Launch Command (for reference)
```bash
NUM_ROLLOUT=50 nohup bash examples/harbor_terminal_rl/launch_27b_tbgae_s3_smoke.sh > run_tbgae_s3_smoke.log 2>&1 &
# pid 1930571, job raysubmit_ttGsbYjmsCyBEzmm
```
