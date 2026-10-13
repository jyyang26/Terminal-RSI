# Harbor v0.3 Trajectory Case Study

Generated: 2026-06-04T12:00:50
Case root: `/path/to/workspace/slime_harbor_v030_case_study/cases/no_summary_stitched`
Rollout dump dir: `/path/to/workspace/slime_harbor_v030_case_study/run/no_summary_stitched_gbs4_train2/rollout_dumps`
Train data dir: `/path/to/workspace/slime_harbor_v030_case_study/train_data/no_summary_stitched_gbs4_train2`

## Summary

- Cases: 4
- Rollout samples indexed: 4
- Train records indexed: 4

## Interpretation

- Stitched trajectory conversion preserves assistant outputs in 4/4 rollout details (`stitched_loss_mask_sum == raw_output_tokens`).
- Native v0.3 `merge_turns` would train on fewer assistant tokens in 4/4 rollout details; this indicates prompt-base drift where earlier assistant outputs are not represented by the merged response.
- Train data matches converted samples in 4/4 records (response length, loss-mask sum, and zero-mask spans).

## Case 1

- Directory: `/path/to/workspace/slime_harbor_v030_case_study/cases/no_summary_stitched/rollout_1/g0000_s0000_i0`
- Mode/status/reward: `trajectory` / `completed` / `0.0`
- Rollout details: 1

### Raw Trajectory Files

- `raw_trial/agent/trajectory.json`: steps=3, continuation=None, copied_steps=0, llm_call_steps=0, subagent_refs=0, sources={'user': 1, 'agent': 2}

### Conversion Details

- detail 0: turns=2, prompt_lens=[1283, 5341], completion_lens=[166, 96], raw_output_tokens=262, prefix_mismatch=[0], base_changed=[1], prefix_drift=[], v030_len/mask=96/96, stitched_len/mask=5603/262, stitched_strategies=['initial_prompt', 'full_prompt'], stitched_context_lens=[0, 5341], stitched_zero_spans=[[166, 5507]]

### Sample/Train Data Checks

- sample 0: case len/mask=5603/262, rollout len/mask=5603/262, train len/mask=5603/262, group=1000000000000, group_mask_sum=262.0, case_zero_spans=[[166, 5507]], train_zero_spans=[[166, 5507]]

### Interpretation

- Native v0.3 merge would drop or hide earlier assistant tokens for this detail (96/262 assistant tokens remain trainable).
- Stitched conversion keeps every assistant output trainable (262/262 assistant tokens are masked as loss=1).
- Zero-mask spans in the stitched response are re-rendered context/terminal tokens, not model outputs.
- The final train_data record matches the converted sample masks for this case.

## Case 2

- Directory: `/path/to/workspace/slime_harbor_v030_case_study/cases/no_summary_stitched/rollout_1/g0000_s0001_i1`
- Mode/status/reward: `trajectory` / `completed` / `0.0`
- Rollout details: 1

### Raw Trajectory Files

- `raw_trial/agent/trajectory.json`: steps=3, continuation=None, copied_steps=0, llm_call_steps=0, subagent_refs=0, sources={'user': 1, 'agent': 2}

### Conversion Details

- detail 0: turns=2, prompt_lens=[1283, 5378], completion_lens=[182, 2021], raw_output_tokens=2203, prefix_mismatch=[0], base_changed=[1], prefix_drift=[], v030_len/mask=2021/2021, stitched_len/mask=7581/2203, stitched_strategies=['initial_prompt', 'full_prompt'], stitched_context_lens=[0, 5378], stitched_zero_spans=[[182, 5560]]

### Sample/Train Data Checks

- sample 1000000: case len/mask=7581/2203, rollout len/mask=7581/2203, train len/mask=7581/2203, group=1000000010000, group_mask_sum=2203.0, case_zero_spans=[[182, 5560]], train_zero_spans=[[182, 5560]]

### Interpretation

- Native v0.3 merge would drop or hide earlier assistant tokens for this detail (2021/2203 assistant tokens remain trainable).
- Stitched conversion keeps every assistant output trainable (2203/2203 assistant tokens are masked as loss=1).
- Zero-mask spans in the stitched response are re-rendered context/terminal tokens, not model outputs.
- The final train_data record matches the converted sample masks for this case.

## Case 3

- Directory: `/path/to/workspace/slime_harbor_v030_case_study/cases/no_summary_stitched/rollout_1/g0000_s0002_i2`
- Mode/status/reward: `trajectory` / `completed` / `0.0`
- Rollout details: 1

### Raw Trajectory Files

- `raw_trial/agent/trajectory.json`: steps=3, continuation=None, copied_steps=0, llm_call_steps=0, subagent_refs=0, sources={'user': 1, 'agent': 2}

### Conversion Details

- detail 0: turns=2, prompt_lens=[1283, 5380], completion_lens=[183, 1299], raw_output_tokens=1482, prefix_mismatch=[0], base_changed=[1], prefix_drift=[], v030_len/mask=1299/1299, stitched_len/mask=6862/1482, stitched_strategies=['initial_prompt', 'full_prompt'], stitched_context_lens=[0, 5380], stitched_zero_spans=[[183, 5563]]

### Sample/Train Data Checks

- sample 2000000: case len/mask=6862/1482, rollout len/mask=6862/1482, train len/mask=6862/1482, group=1000000020000, group_mask_sum=1482.0, case_zero_spans=[[183, 5563]], train_zero_spans=[[183, 5563]]

### Interpretation

- Native v0.3 merge would drop or hide earlier assistant tokens for this detail (1299/1482 assistant tokens remain trainable).
- Stitched conversion keeps every assistant output trainable (1482/1482 assistant tokens are masked as loss=1).
- Zero-mask spans in the stitched response are re-rendered context/terminal tokens, not model outputs.
- The final train_data record matches the converted sample masks for this case.

## Case 4

- Directory: `/path/to/workspace/slime_harbor_v030_case_study/cases/no_summary_stitched/rollout_1/g0000_s0003_i3`
- Mode/status/reward: `trajectory` / `completed` / `0.0`
- Rollout details: 1

### Raw Trajectory Files

- `raw_trial/agent/trajectory.json`: steps=3, continuation=None, copied_steps=0, llm_call_steps=0, subagent_refs=0, sources={'user': 1, 'agent': 2}

### Conversion Details

- detail 0: turns=2, prompt_lens=[1283, 1606], completion_lens=[215, 122], raw_output_tokens=337, prefix_mismatch=[0], base_changed=[1], prefix_drift=[], v030_len/mask=122/122, stitched_len/mask=1943/337, stitched_strategies=['initial_prompt', 'full_prompt'], stitched_context_lens=[0, 1606], stitched_zero_spans=[[215, 1821]]

### Sample/Train Data Checks

- sample 3000000: case len/mask=1943/337, rollout len/mask=1943/337, train len/mask=1943/337, group=1000000030000, group_mask_sum=337.0, case_zero_spans=[[215, 1821]], train_zero_spans=[[215, 1821]]

### Interpretation

- Native v0.3 merge would drop or hide earlier assistant tokens for this detail (122/337 assistant tokens remain trainable).
- Stitched conversion keeps every assistant output trainable (337/337 assistant tokens are masked as loss=1).
- Zero-mask spans in the stitched response are re-rendered context/terminal tokens, not model outputs.
- The final train_data record matches the converted sample masks for this case.
