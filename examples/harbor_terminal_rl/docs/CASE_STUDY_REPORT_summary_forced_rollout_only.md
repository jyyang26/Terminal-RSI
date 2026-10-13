# Harbor v0.3 Trajectory Case Study

Generated: 2026-06-04T12:00:51
Case root: `/path/to/workspace/slime_harbor_v030_case_study/cases/summary_forced`
Rollout dump dir: `/path/to/workspace/slime_harbor_v030_case_study/run/summary_forced_rollout_only/rollout_dumps`
Train data dir: `/path/to/workspace/slime_harbor_v030_case_study/train_data/summary_forced_rollout_only`

## Summary

- Cases: 2
- Rollout samples indexed: 2
- Train records indexed: 0

## Interpretation

- Stitched trajectory conversion preserves assistant outputs in 2/2 rollout details (`stitched_loss_mask_sum == raw_output_tokens`).
- No train_data records were supplied for this report; checks stop at rollout/sample dumps.
- At least one raw trajectory contains copied context, continuation metadata, or subagent references. Those fields are raw ATIF evidence of history rewriting/summarization paths; the training sample is still built from `rollout_details`, so copied context can appear as prompt tokens but is not assigned loss.

## Case 1

- Directory: `/path/to/workspace/slime_harbor_v030_case_study/cases/summary_forced/rollout_0/g0000_s0000_i0`
- Mode/status/reward: `trajectory` / `completed` / `0.0`
- Rollout details: 1

### Raw Trajectory Files

- `raw_trial/agent/trajectory.json`: steps=2, continuation=None, copied_steps=1, llm_call_steps=0, subagent_refs=0, sources={'user': 1, 'agent': 1}

### Conversion Details

- detail 0: turns=1, prompt_lens=[590], completion_lens=[2048], raw_output_tokens=2048, prefix_mismatch=[], base_changed=[], prefix_drift=[], v030_len/mask=2048/2048, stitched_len/mask=2048/2048, stitched_strategies=['initial_prompt'], stitched_context_lens=[0], stitched_zero_spans=[]

### Sample/Train Data Checks

- sample 0: case len/mask=2048/2048, rollout len/mask=2048/2048, train len/mask=None/None, group=None, group_mask_sum=None, case_zero_spans=[], train_zero_spans=None

### Interpretation

- Stitched conversion keeps every assistant output trainable (2048/2048 assistant tokens are masked as loss=1).
- Raw trajectory contains copied/continued/subagent history markers (copied_steps=1, subagent_refs=0, continuation=False). The current converter does not reconstruct hidden pre-summary assistant turns from those markers.

## Case 2

- Directory: `/path/to/workspace/slime_harbor_v030_case_study/cases/summary_forced/rollout_1/g0000_s0000_i1`
- Mode/status/reward: `trajectory` / `completed` / `0.0`
- Rollout details: 1

### Raw Trajectory Files

- `raw_trial/agent/trajectory.json`: steps=2, continuation=None, copied_steps=1, llm_call_steps=0, subagent_refs=0, sources={'user': 1, 'agent': 1}

### Conversion Details

- detail 0: turns=1, prompt_lens=[590], completion_lens=[6], raw_output_tokens=6, prefix_mismatch=[], base_changed=[], prefix_drift=[], v030_len/mask=6/6, stitched_len/mask=6/6, stitched_strategies=['initial_prompt'], stitched_context_lens=[0], stitched_zero_spans=[]

### Sample/Train Data Checks

- sample 1000000: case len/mask=6/6, rollout len/mask=6/6, train len/mask=None/None, group=None, group_mask_sum=None, case_zero_spans=[], train_zero_spans=None

### Interpretation

- Stitched conversion keeps every assistant output trainable (6/6 assistant tokens are masked as loss=1).
- Raw trajectory contains copied/continued/subagent history markers (copied_steps=1, subagent_refs=0, continuation=False). The current converter does not reconstruct hidden pre-summary assistant turns from those markers.
