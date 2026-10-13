# Harbor v0.3 Trajectory Case Study

Generated: 2026-06-05T04:26:04
Case root: `/path/to/workspace/slime_harbor_v030_case_study/cases/no_summary_step`
Rollout dump dir: `/path/to/workspace/slime_harbor_v030_case_study/run/no_summary_step_gbs4_rbs2_train2/rollout_dumps`
Train data dir: `/path/to/workspace/slime_harbor_v030_case_study/train_data/no_summary_step_gbs4_rbs2_trainonly_rankdump`

## Summary

- Cases: 4
- Rollout samples indexed: 8
- Train records indexed: 8

## Interpretation

- Converted samples are pure prompt-snapshot response targets in 8/8 samples (`loss_mask_sum == response_length`, no zero-mask context spans).
- Converted sample prompt/response tokens match raw rollout details in 8/8 samples.
- Train data matches converted samples in 8/8 records (tokens, response length, loss-mask sum, and zero-mask spans).

## Case 1

- Directory: `/path/to/workspace/slime_harbor_v030_case_study/cases/no_summary_step/rollout_1/g0000_s0000_i0`
- Mode/status/reward: `step` / `completed` / `0.0`
- Rollout details: 1

### Raw Trajectory Files

- `raw_trial/agent/trajectory.json`: steps=3, continuation=None, copied_steps=0, llm_call_steps=0, subagent_refs=0, sources={'user': 1, 'agent': 2}

### Conversion Details

- detail 0: turns=2, prompt_lens=[1283, 1508], completion_lens=[158, 110], raw_output_tokens=268, prefix_mismatch=None, base_changed=[], prefix_drift=[], v030_len/mask=None/None, stitched_len/mask=None/None, stitched_strategies=[], stitched_context_lens=[], stitched_zero_spans=[]

### Sample/Train Data Checks

- sample 0: mode=step, detail/step=0/0, raw prompt/completion=1283/158, raw_match=True/True, case len/mask=158/158, rollout len/mask=158/158, train len/mask=158/158, group=1000000000000, group_mask_sum=268.0, case_zero_spans=[], train_zero_spans=[], train_match=True/True
- sample 1: mode=step, detail/step=0/1, raw prompt/completion=1508/110, raw_match=True/True, case len/mask=110/110, rollout len/mask=110/110, train len/mask=110/110, group=1000000000000, group_mask_sum=268.0, case_zero_spans=[], train_zero_spans=[], train_match=True/True

### Interpretation

- The final train_data record matches the converted sample tokens and masks for this case.
- Each converted step sample exactly matches its raw rollout prompt/completion snapshot.

## Case 2

- Directory: `/path/to/workspace/slime_harbor_v030_case_study/cases/no_summary_step/rollout_1/g0000_s0001_i1`
- Mode/status/reward: `step` / `completed` / `0.0`
- Rollout details: 1

### Raw Trajectory Files

- `raw_trial/agent/trajectory.json`: steps=3, continuation=None, copied_steps=0, llm_call_steps=0, subagent_refs=0, sources={'user': 1, 'agent': 2}

### Conversion Details

- detail 0: turns=2, prompt_lens=[1283, 5411], completion_lens=[183, 114], raw_output_tokens=297, prefix_mismatch=None, base_changed=[], prefix_drift=[], v030_len/mask=None/None, stitched_len/mask=None/None, stitched_strategies=[], stitched_context_lens=[], stitched_zero_spans=[]

### Sample/Train Data Checks

- sample 1000000: mode=step, detail/step=0/0, raw prompt/completion=1283/183, raw_match=True/True, case len/mask=183/183, rollout len/mask=183/183, train len/mask=183/183, group=1000000010000, group_mask_sum=297.0, case_zero_spans=[], train_zero_spans=[], train_match=True/True
- sample 1000001: mode=step, detail/step=0/1, raw prompt/completion=5411/114, raw_match=True/True, case len/mask=114/114, rollout len/mask=114/114, train len/mask=114/114, group=1000000010000, group_mask_sum=297.0, case_zero_spans=[], train_zero_spans=[], train_match=True/True

### Interpretation

- The final train_data record matches the converted sample tokens and masks for this case.
- Each converted step sample exactly matches its raw rollout prompt/completion snapshot.

## Case 3

- Directory: `/path/to/workspace/slime_harbor_v030_case_study/cases/no_summary_step/rollout_1/g0001_s0000_i2`
- Mode/status/reward: `step` / `completed` / `0.0`
- Rollout details: 1

### Raw Trajectory Files

- `raw_trial/agent/trajectory.json`: steps=3, continuation=None, copied_steps=0, llm_call_steps=0, subagent_refs=0, sources={'user': 1, 'agent': 2}

### Conversion Details

- detail 0: turns=2, prompt_lens=[1283, 1482], completion_lens=[166, 119], raw_output_tokens=285, prefix_mismatch=None, base_changed=[], prefix_drift=[], v030_len/mask=None/None, stitched_len/mask=None/None, stitched_strategies=[], stitched_context_lens=[], stitched_zero_spans=[]

### Sample/Train Data Checks

- sample 2000000: mode=step, detail/step=0/0, raw prompt/completion=1283/166, raw_match=True/True, case len/mask=166/166, rollout len/mask=166/166, train len/mask=166/166, group=1000100000000, group_mask_sum=285.0, case_zero_spans=[], train_zero_spans=[], train_match=True/True
- sample 2000001: mode=step, detail/step=0/1, raw prompt/completion=1482/119, raw_match=True/True, case len/mask=119/119, rollout len/mask=119/119, train len/mask=119/119, group=1000100000000, group_mask_sum=285.0, case_zero_spans=[], train_zero_spans=[], train_match=True/True

### Interpretation

- The final train_data record matches the converted sample tokens and masks for this case.
- Each converted step sample exactly matches its raw rollout prompt/completion snapshot.

## Case 4

- Directory: `/path/to/workspace/slime_harbor_v030_case_study/cases/no_summary_step/rollout_1/g0001_s0001_i3`
- Mode/status/reward: `step` / `completed` / `0.0`
- Rollout details: 1

### Raw Trajectory Files

- `raw_trial/agent/trajectory.json`: steps=3, continuation=None, copied_steps=0, llm_call_steps=0, subagent_refs=0, sources={'user': 1, 'agent': 2}

### Conversion Details

- detail 0: turns=2, prompt_lens=[1283, 5361], completion_lens=[232, 104], raw_output_tokens=336, prefix_mismatch=None, base_changed=[], prefix_drift=[], v030_len/mask=None/None, stitched_len/mask=None/None, stitched_strategies=[], stitched_context_lens=[], stitched_zero_spans=[]

### Sample/Train Data Checks

- sample 3000000: mode=step, detail/step=0/0, raw prompt/completion=1283/232, raw_match=True/True, case len/mask=232/232, rollout len/mask=232/232, train len/mask=232/232, group=1000100010000, group_mask_sum=336.0, case_zero_spans=[], train_zero_spans=[], train_match=True/True
- sample 3000001: mode=step, detail/step=0/1, raw prompt/completion=5361/104, raw_match=True/True, case len/mask=104/104, rollout len/mask=104/104, train len/mask=104/104, group=1000100010000, group_mask_sum=336.0, case_zero_spans=[], train_zero_spans=[], train_match=True/True

### Interpretation

- The final train_data record matches the converted sample tokens and masks for this case.
- Each converted step sample exactly matches its raw rollout prompt/completion snapshot.
