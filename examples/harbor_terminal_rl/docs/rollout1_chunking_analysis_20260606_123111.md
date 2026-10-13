# Harbor Rollout Chunking Analysis

- dump: `/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_actor1_rollout3_batch3_n32_fully_async_trajectory_normalized_timeout3600_20260606_123111/rollout_dumps/1.pt`
- rollout_id: `1`
- samples/chunks: `644`
- trajectories: `96`
- reduction if one sample per trajectory: `6.708x`
- optimistic merged samples over 64k: `32`
- optimistic merged samples over 128k: `0`

## Chunk Count Histogram

- `1` chunks: `4` trajectories
- `2` chunks: `5` trajectories
- `3` chunks: `11` trajectories
- `4` chunks: `7` trajectories
- `5` chunks: `17` trajectories
- `6` chunks: `6` trajectories
- `7` chunks: `14` trajectories
- `8` chunks: `8` trajectories
- `9` chunks: `7` trajectories
- `10` chunks: `5` trajectories
- `11` chunks: `5` trajectories
- `12` chunks: `1` trajectories
- `13` chunks: `1` trajectories
- `14` chunks: `1` trajectories
- `15` chunks: `1` trajectories
- `16` chunks: `1` trajectories
- `23` chunks: `1` trajectories
- `25` chunks: `1` trajectories

## Summaries

### chunks_per_trajectory
n=96, min=1, mean=6.71, p50=6, p90=11, p95=13, p99=23, max=25, over_64k=0, over_128k=0

### chunk_total_len
n=644, min=1584, mean=34230.13, p50=32706, p90=65536, p95=65536, p99=65536, max=65536, over_64k=0, over_128k=0

### chunk_prompt_len
n=644, min=1, mean=25923.83, p50=23915, p90=55040, p95=61611, p99=64729, max=65345, over_64k=0, over_128k=0

### chunk_response_len
n=644, min=170, mean=8306.3, p50=5256, p90=21435, p95=25693, p99=37671, max=65535, over_64k=0, over_128k=0

### chunk_loss_tokens
n=644, min=170, mean=4643.02, p50=3095, p90=10551, p95=13518, p99=19698, max=36244, over_64k=0, over_128k=0

### chunk_zero_mask_tokens
n=644, min=0, mean=3663.29, p50=1997, p90=9587, p95=12816, p99=18104, max=29291, over_64k=0, over_128k=0

### common_suffix_candidate_tokens
n=644, min=9, mean=9.0, p50=9, p90=9, p95=9, p99=9, max=9, over_64k=0, over_128k=0

### one_sample_total_lower_bound
n=96, min=3988, mean=56767.46, p50=53414, p90=93893, p95=103692, p99=116893, max=125569, over_64k=32, over_128k=0

### one_sample_response_tokens
n=96, min=3161, mean=55721.46, p50=52587, p90=92610, p95=102409, p99=115610, max=124286, over_64k=30, over_128k=0

### one_sample_loss_tokens
n=96, min=1678, mean=31146.91, p50=28104, p90=53915, p95=56737, p99=70451, max=76358, over_64k=2, over_128k=0

### one_sample_zero_mask_tokens
n=96, min=1483, mean=24574.55, p50=23916, p90=41256, p95=44196, p99=48494, max=53835, over_64k=0, over_128k=0

## Top Trajectories By Chunk Count

- group=1000200060000 chunks=25 turns=100 split_chunks=24 normalized_chunks=18 merged_lb=53414 loss=34596 zero=17991
- group=1000000180000 chunks=23 turns=100 split_chunks=22 normalized_chunks=18 merged_lb=105254 loss=62993 zero=40978
- group=1000000290000 chunks=16 turns=100 split_chunks=15 normalized_chunks=11 merged_lb=101286 loss=55087 zero=44916
- group=1000100210000 chunks=15 turns=42 split_chunks=14 normalized_chunks=9 merged_lb=68611 loss=37884 zero=29699
- group=1000200160000 chunks=14 turns=100 split_chunks=13 normalized_chunks=13 merged_lb=46325 loss=28104 zero=17394
- group=1000000250000 chunks=13 turns=74 split_chunks=12 normalized_chunks=10 merged_lb=66709 loss=34033 zero=31393
- group=1000000070000 chunks=12 turns=73 split_chunks=11 normalized_chunks=12 merged_lb=65528 loss=37627 zero=26618
- group=1000000300000 chunks=11 turns=84 split_chunks=10 normalized_chunks=9 merged_lb=116893 loss=76358 zero=39252
- group=1000000060000 chunks=11 turns=100 split_chunks=10 normalized_chunks=11 merged_lb=97692 loss=57800 zero=38609
- group=1000000000000 chunks=11 turns=100 split_chunks=10 normalized_chunks=8 merged_lb=84120 loss=46116 zero=36721
- group=1000200000000 chunks=11 turns=100 split_chunks=10 normalized_chunks=9 merged_lb=59953 loss=34716 zero=24410
- group=1000200280000 chunks=11 turns=100 split_chunks=10 normalized_chunks=10 merged_lb=44289 loss=27989 zero=15473
- group=1000000260000 chunks=10 turns=100 split_chunks=9 normalized_chunks=10 merged_lb=104053 loss=55390 zero=47380
- group=1000000050000 chunks=10 turns=100 split_chunks=9 normalized_chunks=10 merged_lb=93075 loss=56459 zero=35333
- group=1000100110000 chunks=10 turns=55 split_chunks=9 normalized_chunks=9 merged_lb=91683 loss=49399 zero=41256
- group=1000000090000 chunks=10 turns=100 split_chunks=9 normalized_chunks=8 merged_lb=74059 loss=41091 zero=31685
- group=1000200020000 chunks=10 turns=100 split_chunks=9 normalized_chunks=9 merged_lb=35380 loss=22377 zero=12176
- group=1000000310000 chunks=9 turns=100 split_chunks=8 normalized_chunks=5 merged_lb=93893 loss=56737 zero=35873
- group=1000000230000 chunks=9 turns=100 split_chunks=8 normalized_chunks=9 merged_lb=64166 loss=36283 zero=26600
- group=1000200230000 chunks=9 turns=100 split_chunks=8 normalized_chunks=9 merged_lb=39283 loss=26601 zero=11855
