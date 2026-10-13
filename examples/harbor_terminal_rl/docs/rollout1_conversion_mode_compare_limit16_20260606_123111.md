# Harbor Conversion Mode Comparison

- trials_dir: `/path/to/workspace/slime_harbor_trials/tb2_full89_32gpu_9b_max100_actor1_rollout3_batch3_n32_resp4096_ctx128k_timeout3600_trajectory_normalized_20260606_123111`
- selected result.json files: `16`
- converted trials: `16`
- conversion errors: `0`

## Mode Stats

### trajectory
- samples: `16`
- total_len: n=16, min=1022684, mean=4506205.69, p50=5115876, p90=6106087, p95=6474364, p99=6474364, max=6795288, over_64k=16, over_128k=16
- response_len: n=16, min=1021401, mean=4504922.69, p50=5114593, p90=6104804, p95=6473081, p99=6473081, max=6794005, over_64k=16, over_128k=16
- loss_tokens: n=16, min=22467, mean=47285.44, p50=47302, p90=62993, p95=65212, p99=65212, max=70451, over_64k=1, over_128k=0
- zero_mask_tokens: n=16, min=998934, mean=4457637.25, p50=5068477, p90=6050889, p95=6410088, p99=6410088, max=6723554, over_64k=16, over_128k=16

### trajectory_normalized
- samples: `114`
- total_len: n=114, min=8582, mean=50178.75, p50=43507, p90=93179, p95=102998, p99=115064, max=126998, over_64k=38, over_128k=0
- response_len: n=114, min=191, mean=11795.43, p50=7162, p90=25693, p95=35508, p99=49930, max=69163, over_64k=1, over_128k=0
- loss_tokens: n=114, min=191, mean=6636.55, p50=3694, p90=14303, p95=18687, p99=27619, max=39212, over_64k=0, over_128k=0
- zero_mask_tokens: n=114, min=0, mean=5158.88, p50=2875, p90=12361, p95=16835, p99=22311, max=29951, over_64k=0, over_128k=0

## Normalized Merge Lower Bound

- samples before: `114`
- samples after: `16`
- reduction factor: `7.125`
- total_len: n=16, min=36981, mean=85325.44, p50=88015, p90=105254, p95=113124, p99=113124, max=125569, over_64k=13, over_128k=0
- response_len: n=16, min=35698, mean=84042.44, p50=86732, p90=103971, p95=111841, p99=111841, max=124286, over_64k=13, over_128k=0

## Trajectory Stitch Stats

- context_tokens: n=16, min=998934, mean=4457637.25, p50=5068477, p90=6050889, p95=6410088, p99=6410088, max=6723554, over_64k=16, over_128k=16
- output_tokens: n=16, min=22467, mean=47285.44, p50=47302, p90=62993, p95=65212, p99=65212, max=70451, over_64k=1, over_128k=0
- full_prompt_turns: n=16, min=36, mean=84.06, p50=97, p90=99, p95=99, p99=99, max=99, over_64k=0, over_128k=0
- suffix_prompt_turns: n=16, min=0, mean=0.0, p50=0, p90=0, p95=0, p99=0, max=0, over_64k=0, over_128k=0
- prefix_tail_turns: n=16, min=0, mean=0.0, p50=0, p90=0, p95=0, p99=0, max=0, over_64k=0, over_128k=0

## Chunk Count Histogram

- `3` chunks: `3` trajectories
- `4` chunks: `1` trajectories
- `5` chunks: `3` trajectories
- `7` chunks: `4` trajectories
- `8` chunks: `3` trajectories
- `11` chunks: `1` trajectories
- `23` chunks: `1` trajectories

## Largest Trajectory-Mode Samples

- slime-r1-g0-s13-i13 total=6795288 response=6794005 context=6723554 full_prompt_turns=96
- slime-r1-g0-s18-i18 total=6474364 response=6473081 context=6410088 full_prompt_turns=99
- slime-r1-g0-s14-i14 total=6106087 response=6104804 context=6050889 full_prompt_turns=99
- slime-r1-g0-s10-i10 total=5990700 response=5989417 context=5937241 full_prompt_turns=99
- slime-r1-g0-s17-i17 total=5791835 response=5790552 context=5725340 full_prompt_turns=97
- slime-r1-g0-s1-i1 total=5521677 response=5520394 context=5472066 full_prompt_turns=99
- slime-r1-g0-s15-i15 total=5490677 response=5489394 context=5442462 full_prompt_turns=99
- slime-r1-g0-s0-i0 total=5115876 response=5114593 context=5068477 full_prompt_turns=99
- slime-r1-g0-s19-i19 total=4805613 response=4804330 context=4750392 full_prompt_turns=89
- slime-r1-g0-s21-i21 total=4726413 response=4725130 context=4679742 full_prompt_turns=99
