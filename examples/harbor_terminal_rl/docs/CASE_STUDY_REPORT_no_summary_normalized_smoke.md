# Harbor v0.3 Trajectory Case Study

Generated: 2026-06-05T10:46:55
Case root: `/path/to/workspace/slime_harbor_v030_case_study/cases/no_summary_normalized`
Rollout dump dir: `/path/to/workspace/slime_harbor_v030_case_study/run/no_summary_normalized/rollout_dumps`
Train data dir: `/path/to/workspace/slime_harbor_v030_case_study/train_data/no_summary_normalized`

## Summary

- Cases: 4
- Rollout samples indexed: 4
- Train records indexed: 4

## Interpretation

- Normalized trajectory conversion produced 4 stitched samples; 4/4 stayed as one chunk without fallback splitting.
- Strict train/inference conditioning match is marked true for 0/4 samples. A false value here means the stitcher used normalization, such as dropping renderer/template boundary tokens, rather than a raw-token prefix.
- Train data matches converted samples in 4/4 records (tokens, response length, loss-mask sum, and zero-mask spans).
- Train-data advantage tensors are present; maximum absolute advantage in indexed records is 0.

## Case 1

- Directory: `/path/to/workspace/slime_harbor_v030_case_study/cases/no_summary_normalized/rollout_1/g0000_s0000_i0`
- Mode/status/reward: `trajectory_normalized` / `completed` / `0.0`
- Rollout details: 1

### Raw Trajectory Files

- `raw_trial/agent/trajectory.json`: steps=3, continuation=None, copied_steps=0, llm_call_steps=0, subagent_refs=0, sources={'user': 1, 'agent': 2}

### Conversion Details

- detail 0: turns=2, prompt_lens=[1283, 5391], completion_lens=[169, 1024], raw_output_tokens=1193, prefix_mismatch=None, base_changed=[], prefix_drift=[], v030_len/mask=None/None, stitched_len/mask=None/None, stitched_strategies=[], stitched_context_lens=[], stitched_zero_spans=[]

### Sample/Train Data Checks

- sample 0: mode=trajectory_normalized, detail/step=0/None, chunk=0/1 turns=[0, 1], conditioning_match=False, norm_context/output=3943/1193, norm_prefix=1, strict_prefix=0, dropped(prompt/resp_prefix/resp_suffix)=4/0/0, raw prompt/completion=None/None, raw_match=None/None, case len/mask=5136/1193, rollout len/mask=5136/1193, train len/mask=5136/1193, group=1000000000000, group_mask_sum=1193.0, reward/raw=0.0/0.0, adv={'len': 5136, 'sum': 0.0, 'mean': 0.0, 'min': 0.0, 'max': 0.0, 'max_abs': 0.0}, returns={'len': 5136, 'sum': 0.0, 'mean': 0.0, 'min': 0.0, 'max': 0.0, 'max_abs': 0.0}, case_zero_spans=[[169, 4112]], train_zero_spans=[[169, 4112]], train_match=True/True

### Interpretation

- The final train_data record matches the converted sample tokens and masks for this case.
- Advantages are all zero for this case; this is expected when every sibling in the GRPO group has identical normalized reward.
- Normalized stitching emitted chunk 0/1 over turns [0, 1]: context_tokens=3943, output_tokens=1193, zero_spans=[[169, 4112]], conditioning_match=False, dropped_tokens(prompt_suffix/response_prefix/response_suffix)=4/0/0.

## Case 2

- Directory: `/path/to/workspace/slime_harbor_v030_case_study/cases/no_summary_normalized/rollout_1/g0000_s0001_i1`
- Mode/status/reward: `trajectory_normalized` / `completed` / `0.0`
- Rollout details: 1

### Raw Trajectory Files

- `raw_trial/agent/trajectory.json`: steps=3, continuation=None, copied_steps=0, llm_call_steps=0, subagent_refs=0, sources={'user': 1, 'agent': 2}

### Conversion Details

- detail 0: turns=2, prompt_lens=[1283, 5386], completion_lens=[190, 1024], raw_output_tokens=1214, prefix_mismatch=None, base_changed=[], prefix_drift=[], v030_len/mask=None/None, stitched_len/mask=None/None, stitched_strategies=[], stitched_context_lens=[], stitched_zero_spans=[]

### Sample/Train Data Checks

- sample 1000000: mode=trajectory_normalized, detail/step=0/None, chunk=0/1 turns=[0, 1], conditioning_match=False, norm_context/output=3917/1214, norm_prefix=1, strict_prefix=0, dropped(prompt/resp_prefix/resp_suffix)=4/0/0, raw prompt/completion=None/None, raw_match=None/None, case len/mask=5131/1214, rollout len/mask=5131/1214, train len/mask=5131/1214, group=1000000010000, group_mask_sum=1214.0, reward/raw=0.0/0.0, adv={'len': 5131, 'sum': 0.0, 'mean': 0.0, 'min': 0.0, 'max': 0.0, 'max_abs': 0.0}, returns={'len': 5131, 'sum': 0.0, 'mean': 0.0, 'min': 0.0, 'max': 0.0, 'max_abs': 0.0}, case_zero_spans=[[190, 4107]], train_zero_spans=[[190, 4107]], train_match=True/True

### Interpretation

- The final train_data record matches the converted sample tokens and masks for this case.
- Advantages are all zero for this case; this is expected when every sibling in the GRPO group has identical normalized reward.
- Normalized stitching emitted chunk 0/1 over turns [0, 1]: context_tokens=3917, output_tokens=1214, zero_spans=[[190, 4107]], conditioning_match=False, dropped_tokens(prompt_suffix/response_prefix/response_suffix)=4/0/0.

## Case 3

- Directory: `/path/to/workspace/slime_harbor_v030_case_study/cases/no_summary_normalized/rollout_1/g0001_s0000_i2`
- Mode/status/reward: `trajectory_normalized` / `completed` / `0.0`
- Rollout details: 1

### Raw Trajectory Files

- `raw_trial/agent/trajectory.json`: steps=3, continuation=None, copied_steps=0, llm_call_steps=0, subagent_refs=0, sources={'user': 1, 'agent': 2}

### Conversion Details

- detail 0: turns=2, prompt_lens=[1283, 1511], completion_lens=[192, 109], raw_output_tokens=301, prefix_mismatch=None, base_changed=[], prefix_drift=[], v030_len/mask=None/None, stitched_len/mask=None/None, stitched_strategies=[], stitched_context_lens=[], stitched_zero_spans=[]

### Sample/Train Data Checks

- sample 2000000: mode=trajectory_normalized, detail/step=0/None, chunk=0/1 turns=[0, 1], conditioning_match=False, norm_context/output=40/301, norm_prefix=1, strict_prefix=0, dropped(prompt/resp_prefix/resp_suffix)=4/0/0, raw prompt/completion=None/None, raw_match=None/None, case len/mask=341/301, rollout len/mask=341/301, train len/mask=341/301, group=1000100000000, group_mask_sum=301.0, reward/raw=0.0/0.0, adv={'len': 341, 'sum': 0.0, 'mean': 0.0, 'min': 0.0, 'max': 0.0, 'max_abs': 0.0}, returns={'len': 341, 'sum': 0.0, 'mean': 0.0, 'min': 0.0, 'max': 0.0, 'max_abs': 0.0}, case_zero_spans=[[192, 232]], train_zero_spans=[[192, 232]], train_match=True/True

### Interpretation

- The final train_data record matches the converted sample tokens and masks for this case.
- Advantages are all zero for this case; this is expected when every sibling in the GRPO group has identical normalized reward.
- Normalized stitching emitted chunk 0/1 over turns [0, 1]: context_tokens=40, output_tokens=301, zero_spans=[[192, 232]], conditioning_match=False, dropped_tokens(prompt_suffix/response_prefix/response_suffix)=4/0/0.

## Case 4

- Directory: `/path/to/workspace/slime_harbor_v030_case_study/cases/no_summary_normalized/rollout_1/g0001_s0001_i3`
- Mode/status/reward: `trajectory_normalized` / `completed` / `0.0`
- Rollout details: 1

### Raw Trajectory Files

- `raw_trial/agent/trajectory.json`: steps=3, continuation=None, copied_steps=0, llm_call_steps=0, subagent_refs=0, sources={'user': 1, 'agent': 2}

### Conversion Details

- detail 0: turns=2, prompt_lens=[1283, 1486], completion_lens=[167, 104], raw_output_tokens=271, prefix_mismatch=None, base_changed=[], prefix_drift=[], v030_len/mask=None/None, stitched_len/mask=None/None, stitched_strategies=[], stitched_context_lens=[], stitched_zero_spans=[]

### Sample/Train Data Checks

- sample 3000000: mode=trajectory_normalized, detail/step=0/None, chunk=0/1 turns=[0, 1], conditioning_match=False, norm_context/output=40/271, norm_prefix=1, strict_prefix=0, dropped(prompt/resp_prefix/resp_suffix)=4/0/0, raw prompt/completion=None/None, raw_match=None/None, case len/mask=311/271, rollout len/mask=311/271, train len/mask=311/271, group=1000100010000, group_mask_sum=271.0, reward/raw=0.0/0.0, adv={'len': 311, 'sum': 0.0, 'mean': 0.0, 'min': 0.0, 'max': 0.0, 'max_abs': 0.0}, returns={'len': 311, 'sum': 0.0, 'mean': 0.0, 'min': 0.0, 'max': 0.0, 'max_abs': 0.0}, case_zero_spans=[[167, 207]], train_zero_spans=[[167, 207]], train_match=True/True

### Interpretation

- The final train_data record matches the converted sample tokens and masks for this case.
- Advantages are all zero for this case; this is expected when every sibling in the GRPO group has identical normalized reward.
- Normalized stitching emitted chunk 0/1 over turns [0, 1]: context_tokens=40, output_tokens=271, zero_spans=[[167, 207]], conditioning_match=False, dropped_tokens(prompt_suffix/response_prefix/response_suffix)=4/0/0.
