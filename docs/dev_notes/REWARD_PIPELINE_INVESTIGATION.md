================================================================================
SLIME HARBOR V0.3.0 REWARD PIPELINE INVESTIGATION REPORT
TerminalWorld Terminus2 RL Task - Length-Based Reward Shaping
================================================================================

## EXECUTIVE SUMMARY

The Harbor reward pipeline flows:
  1. Trial verifier result (binary checker) → extract_reward() → float value
  2. atif_adapter converts Harbor trial to slime Sample, sets sample.reward
  3. Optional custom_reward_post_process hook (via --custom-reward-post-process-path)
  4. Sample.response_length is always available for post-processing

For length-based reward shaping, the cleanest integration point is a custom 
post-process hook that receives the final samples (with reward + response_length) 
and returns shaped rewards.

================================================================================
1. WHERE DOES THE REWARD ORIGINATE?
================================================================================

ENTRY POINT: extract_reward() function
File: slime_plugins/harbor/atif_adapter.py:33-39

    def extract_reward(result: Any, reward_key: str = "reward", default: float = 0.0) -> float:
        verifier_result = _get_attr_or_item(result, "verifier_result")
        rewards = _get_attr_or_item(verifier_result, "rewards")
        if not rewards:
            return default
        value = rewards.get(reward_key, default)
        return float(value)

The reward originates from: result.verifier_result.rewards[reward_key]
  - SUCCESS: some positive value (e.g., 1.0) from binary checker
  - FAILURE: 0.0 (default) when rewards dict is empty or key absent


REWARD FLOW INTO SAMPLE:

File: slime_plugins/harbor/atif_adapter.py:72-96 (samples_from_trial_result)
  72  def samples_from_trial_result(
  73      result: Any,
  74      base_sample: Sample,
  75      *,
  76      reward_key: str = "reward",
  77      mode: str = "turn",
  78      include_subagents: bool = False,
  79      metadata: dict[str, Any] | None = None,
  80      require_routing_replay: bool = False,
  81  ) -> list[Sample]:
  82      reward = extract_reward(result, reward_key=reward_key)    # LINE 82: Extract
  83      status = extract_status(result)
  84      details = extract_rollout_details(result, include_subagents=include_subagents)
  85      return samples_from_rollout_details(
  86          details,
  87          base_sample,
  88          reward=reward,  # LINE 88: Pass to conversion
  89          ...

File: slime_plugins/harbor/atif_adapter.py:509-579 (samples_from_rollout_details)
  This function dispatches to various sample conversion modes.

For turn-based samples (most common):
File: slime_plugins/harbor/atif_adapter.py:582-656 (_turn_samples_from_detail)
  613      for turn_index, (prompt_ids, completion_ids) in enumerate(...):
  614          if not completion_ids:
  615              continue
  616          sample = _clone_base_sample(base_sample, ...)
  617          sample.tokens = prompt_ids + completion_ids
  618          sample.response_length = len(completion_ids)
  619          sample.reward = reward                        ← LINE 620: REWARD SET
  620          sample.status = status
  621          sample.loss_mask = [1] * sample.response_length
  ...

For trajectory samples:
File: slime_plugins/harbor/atif_adapter.py:659-724 (_trajectory_sample_from_detail)
  692          sample = _clone_base_sample(base_sample, ...)
  693          sample.tokens = list(segment.prompt_ids) + list(segment.response_ids)
  694          sample.response_length = len(segment.response_ids)
  695          sample.reward = reward                        ← LINE 694: REWARD SET
  696          sample.status = status
  697          sample.loss_mask = list(segment.loss_mask)
  ...

For normalized trajectory samples:
File: slime_plugins/harbor/atif_adapter.py:871-954 (_normalized_trajectory_samples_from_detail)
  873          sample = _clone_base_sample(...)
  874          sample.tokens = list(chunk["prompt_ids"]) + list(chunk["response_ids"])
  875          sample.response_length = len(chunk["response_ids"])
  876          sample.reward = reward                        ← LINE 877: REWARD SET
  877          sample.status = status
  878          sample.loss_mask = list(chunk["loss_mask"])
  ...


CURRENT VALUES:
  - SUCCESS:  reward = 1.0 (typical terminal/binary checker success)
  - FAILURE:  reward = 0.0 (no verifier result or empty rewards dict)


================================================================================
2. WHAT REWARD POST-PROCESSING HOOKS EXIST?
================================================================================

HOOK 1: Custom Reward Post-Process (--custom-reward-post-process-path)
Location: slime/ray/rollout.py:857-875

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)  ← LINE 859
        
        # Default: extract, normalize, reshape
        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization):
            # reshape and normalize by group...

FUNCTION SIGNATURE:
    post_process_rewards(args, samples: list[Sample], **kwargs) → tuple[list[float], list[float]]

Returns: (raw_rewards, processed_rewards)
  - raw_rewards: list of floats, original reward values
  - processed_rewards: list of floats, post-processed/shaped rewards
  Both lists must match samples length and order.


HOOK 2: Turn-Grouped Reward (harbor_sample_mode=turn_grouped or step)
File: slime_plugins/harbor/turn_grouped_reward.py:11-40

    def trajectory_grouped_turn_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
        """Normalize Harbor rewards by trajectory group, then broadcast to turns."""
        raw_rewards = [_reward_value(args, sample) for sample in samples]
        if not _should_normalize(args):
            return raw_rewards, raw_rewards
        
        # Group by (rollout_id, group_index) → (rollout_id, group_index, sample_rank)
        group_to_trajectories: OrderedDict[Any, OrderedDict[Any, list[int]]] = OrderedDict()
        for index, sample in enumerate(samples):
            group_key, trajectory_key = _group_and_trajectory_key(sample)
            trajectories = group_to_trajectories.setdefault(group_key, OrderedDict())
            trajectories.setdefault(trajectory_key, []).append(index)
        
        # Normalize per trajectory, then broadcast to all turns in that trajectory
        processed = [0.0] * len(samples)
        for trajectories in group_to_trajectories.values():
            trajectory_items = list(trajectories.items())
            trajectory_rewards = [raw_rewards[indices[0]] for _, indices in trajectory_items]
            trajectory_advantages = _normalize_rewards(args, trajectory_rewards)
            for (_, indices), advantage in zip(trajectory_items, trajectory_advantages):
                for index in indices:
                    processed[index] = advantage
        
        return raw_rewards, processed

ACTIVATED BY: harbor_sample_mode: "turn_grouped" or "step" in config
WIRING: Set in YAML or programmatically; automatically loaded


HOOK 3: Partial/Progress Reward (trajectory-grouped CTRF-based)
File: slime_plugins/harbor/partial_reward.py:23-62

    def trajectory_grouped_progress_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
        """Add CTRF-based progress reward, then normalize by Harbor trajectory group."""
        partial_reward_max = _partial_reward_max(args)
        progress_cache: dict[str | None, ProgressScore] = {}
        
        # Group by trajectory (harbor_rollout_id, harbor_rollout_group_index, harbor_sample_rank)
        group_to_trajectories: OrderedDict[Any, OrderedDict[Any, list[int]]] = OrderedDict()
        for index, sample in enumerate(samples):
            group_key, trajectory_key = _group_and_trajectory_key(sample)
            trajectories = group_to_trajectories.setdefault(group_key, OrderedDict())
            trajectories.setdefault(trajectory_key, []).append(index)
        
        raw_rewards = [0.0] * len(samples)
        for trajectories in group_to_trajectories.values():
            for indices in trajectories.values():
                trajectory_reward = _trajectory_shaped_reward(args, samples[indices[0]], 
                                                             partial_reward_max, progress_cache)
                for index in indices:
                    _annotate_sample(samples[index], trajectory_reward)
                    raw_rewards[index] = trajectory_reward.shaped_reward  # LINE 48
        
        if not _should_normalize(args):
            return raw_rewards, raw_rewards
        
        # Further normalize by trajectory
        processed = [0.0] * len(samples)
        for trajectories in group_to_trajectories.values():
            trajectory_items = list(trajectories.items())
            trajectory_rewards = [raw_rewards[indices[0]] for _, indices in trajectory_items]
            trajectory_advantages = _normalize_rewards(args, trajectory_rewards)
            for (_, indices), advantage in zip(trajectory_items, trajectory_advantages):
                for index in indices:
                    processed[index] = advantage
        
        return raw_rewards, processed

LOGIC:
  - terminal_reward >= 1.0 → shaped_reward = terminal_reward (success unchanged)
  - terminal_reward < 1.0  → shaped_reward = max(terminal_reward, 
                                                  partial_reward_max * progress.score)
  
WIRING: --custom-reward-post-process-path slime_plugins.harbor.partial_reward:post_process_rewards
OR via: harbor_partial_reward_max env var / args.harbor_partial_reward_max
ACTIVATED: In full1530 config, set to "" (EMPTY) = DISABLED


HOOK 4: Alternative Normalization (fanout/compact test helper)
File: slime/rollout/_fanout_test_helpers.py (grpo_normalize_by_group_index)

    def grpo_normalize_by_group_index(args, samples):
        """Drop-in custom-reward-post-process-path for compact/fanout."""
        raw_rewards = [s.get_reward_value(args) for s in samples]
        groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for i, s in enumerate(samples):
            groups[s.group_index].append((i, raw_rewards[i]))
        
        out = [0.0] * len(samples)
        use_std = getattr(args, "grpo_std_normalization", True)
        for indexed in groups.values():
            positions = [p for p, _ in indexed]
            rewards = torch.tensor([r for _, r in indexed], dtype=torch.float)
            rewards = rewards - rewards.mean()
            if use_std:
                rewards = rewards / (rewards.std() + 1e-6)
            for pos, r in zip(positions, rewards.tolist()):
                out[pos] = r
        
        return raw_rewards, out

PURPOSE: Custom grouping (by group_index instead of reshaped n_samples_per_prompt)


================================================================================
3. WHAT FIELDS DOES SAMPLE CARRY ABOUT OUTPUT LENGTH?
================================================================================

File: slime/utils/types.py:9-134 (Sample dataclass definition)

RELEVANT FIELDS FOR LENGTH:

    response_length: int = 0
        (LINE 29) Length of the response (output tokens) for this sample.
        Set during conversion in atif_adapter.py (e.g., line 619, 694, 876).
        
        Available at post-process time: YES
        Value at success: len(completion_ids) for that turn/trajectory chunk
        Value at failure: len(completion_ids) (still set even if reward=0)

    tokens: list[int] = field(default_factory=list)
        (LINE 24) Full token sequence [prompt_tokens + response_tokens]
        Calculated as: len(tokens) - response_length = prompt_length
        
        Available at post-process time: YES

    loss_mask: list[int] | None = None
        (LINE 32) Binary mask over response tokens (1=learn, 0=don't).
        Length equals response_length.
        useful for "effective response length": sum(loss_mask)
        
        Available at post-process time: YES
        Computed: sum(loss_mask) if loss_mask is not None else response_length
        (See line 185-186: effective_response_length property)

    effective_response_length: property (LINE 184-186)
        @property
        def effective_response_length(self):
            return sum(self.loss_mask) if self.loss_mask is not None else self.response_length

SPECTRAL INFO (for speculative decoding):
    spec_info: SpecInfo = field(default_factory=SpecInfo)  (LINE 99)
        With: spec_accept_token_num, spec_draft_token_num, completion_token_num
        
        Available: YES, but less relevant for output length shaping

PREFIX CACHE INFO:
    prefix_cache_info: PrefixCacheInfo = field(default_factory=PrefixCacheInfo)  (LINE 128)
        With: cached_tokens, total_prompt_tokens
        
        Available: YES, tracks prompt-side caching

METADATA (arbitrary):
    metadata: dict = field(default_factory=dict)  (LINE 51)
        In Harbor context, contains harbor_* prefixed keys:
        - harbor_terminal_reward: original reward value
        - harbor_progress_score: CTRF progress ratio
        - harbor_num_turns: number of turns in trajectory
        - harbor_trajectory_chunk_index: chunk number
        - and many more context signals

    Available at post-process time: YES
    Can be used for auxiliary shaping decisions


SUMMARY: LENGTH SIGNALS AVAILABLE FOR REWARD SHAPING
    ✓ response_length: primary signal (completion token count)
    ✓ loss_mask: secondary signal (learnable response token count)
    ✓ effective_response_length property: convenience accessor
    ✓ tokens: full sequence (derive lengths as needed)
    ✓ metadata: contextual information (turns, chunks, etc.)


================================================================================
4. HOW IS THE REWARD CURRENTLY CONFIGURED IN HARBOR RUN?
================================================================================

File: examples/harbor_terminal_rl/run_terminalworld_full1530_96gpu_27b_daytona_n20_partial16_fully_async.sh

REWARD CONFIGURATION (Lines 95-101):

    # >>CHG 1: PROCESS/PARTIAL REWARD (CTRF) OFF. Empty CUSTOM_REWARD_POST_PROCESS_PATH
    # => the launcher's `if [ -n ... ]` skips --custom-reward-post-process-path, so the
    # ONLY reward is the binary terminal checker (harbor_reward_key: reward). The two
    # cap_total_tokens_* paths below are TOKEN-CAPPING (not reward) and are KEPT.
    export CUSTOM_REWARD_POST_PROCESS_PATH=""
    export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
    export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"

⚠️  CUSTOM_REWARD_POST_PROCESS_PATH = "" (EMPTY)
    → Reward post-processing DISABLED
    → Only binary terminal checker reward used (1.0 for success, 0.0 for failure)

File: examples/harbor_terminal_rl/harbor_terminalworld_full1530_96gpu_27b_daytona_n20_partial16_fully_async.yaml

YAML CONFIGURATION (Lines 10-16, 60-62):

    # REWARD DESIGN (per user request):
    #   - process/partial reward (CTRF) OFF  -> set via run wrapper:
    #     CUSTOM_REWARD_POST_PROCESS_PATH="" (no --custom-reward-post-process-path),
    #     so the only reward is the binary terminal checker (harbor_reward_key: reward).
    #   - entropy loss OFF -> ENTROPY_COEF=0.0 in the wrapper (train_loss.py gates
    #     with_entropy on entropy_coef != 0; 0.0 => no entropy term in the loss).
    #   - token-capping postprocess KEPT (cap_total_tokens_*; not reward shaping).
    
    harbor_sample_mode: trajectory_retokenized
    harbor_group_unit: trajectory
    harbor_reward_key: reward

ACTIVE FIELDS:
    harbor_reward_key: reward
        → extract reward from result.verifier_result.rewards["reward"]
        → SUCCESS: 1.0, FAILURE: 0.0 (binary checker only)

    harbor_sample_mode: trajectory_retokenized
        → converts multi-turn Harbor trajectory into single slime Sample
        → response_length = total tokens across all turns (after retokenization)

    harbor_group_unit: trajectory
        → groups samples by trajectory (all turns/steps from same rollout = same group)

NOT SET:
    harbor_partial_reward_max: (default 0.4 in partial_reward.py)
        → unused because CUSTOM_REWARD_POST_PROCESS_PATH=""

NOT SET:
    turn-grouped-reward: (implicit from harbor_sample_mode)
        → would only apply if harbor_sample_mode="turn_grouped" or "step"


LAUNCHER COMMAND CONSTRUCTION (inferred from run-wrapper pattern):
The wrapper sources this config and constructs something like:

    slime-train \
      --config harbor_terminalworld_full1530_96gpu_27b_daytona_n20_partial16_fully_async.yaml \
      --custom-reward-post-process-path ""  # (empty, skipped by if [ -n ... ])
      --rollout-all-samples-process-path slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples \
      --rollout-data-postprocess-path slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data \
      ...

RESULT: Binary reward only. No custom post-processing.


================================================================================
5. WHERE WOULD LENGTH-BASED REWARD SHAPING BEST PLUG IN?
================================================================================

RECOMMENDED APPROACH: Custom Reward Post-Process Hook

BEST INTEGRATION POINT:
    File: (create new) slime_plugins/harbor/length_based_reward_shaping.py

FUNCTION SIGNATURE (matching existing hooks):

    def length_shaped_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
        """
        Shape terminal rewards by output length: reward = base * f(length).
        
        For Harbor Terminus2 RL (binary checker):
          - Success (reward > 0): Shape by response length
            reward_shaped = reward * f(response_length)
            where f(length) ∈ (0, 1], shorter output → higher factor
          - Failure (reward = 0): Keep zero (no shaping)
        
        Args:
            args: HyperParameters (access args.custom_length_shaping_* for tuning)
            samples: list[Sample] with pre-set sample.reward from binary checker
        
        Returns:
            (raw_rewards, processed_rewards)
            Both lists match samples length and order
        """
        raw_rewards = []
        shaped_rewards = []
        
        for sample in samples:
            raw_reward = sample.get_reward_value(args)
            raw_rewards.append(raw_reward)
            
            # Only shape successful outputs
            if raw_reward > 0:
                # Length shaping: shorter responses get higher multiplier
                response_length = sample.response_length
                
                # Option 1: Exponential decay by length
                # reward_shaped = raw_reward * exp(-α * response_length)
                
                # Option 2: Inverse linear (clip at min)
                # reward_shaped = raw_reward * max(min_factor, 1.0 / (1.0 + β * response_length))
                
                # Option 3: Quadratic cost (length^2)
                # reward_shaped = raw_reward * max(min_factor, 1.0 / (1.0 + γ * response_length^2))
                
                # Example implementation (inverse linear):
                alpha = getattr(args, "custom_length_shaping_alpha", 0.01)  # decay rate
                min_factor = getattr(args, "custom_length_shaping_min_factor", 0.1)  # minimum multiplier
                
                multiplier = max(min_factor, 1.0 / (1.0 + alpha * response_length))
                reward_shaped = raw_reward * multiplier
            else:
                # Failure: no shaping
                reward_shaped = 0.0
            
            shaped_rewards.append(reward_shaped)
        
        return raw_rewards, shaped_rewards


ACTIVATION: Add to run-wrapper shell script

    export CUSTOM_REWARD_POST_PROCESS_PATH="slime_plugins.harbor.length_based_reward_shaping:length_shaped_rewards"

    # New tuning parameters (optional, with sensible defaults)
    export CUSTOM_LENGTH_SHAPING_ALPHA="${CUSTOM_LENGTH_SHAPING_ALPHA:-0.01}"
    export CUSTOM_LENGTH_SHAPING_MIN_FACTOR="${CUSTOM_LENGTH_SHAPING_MIN_FACTOR:-0.1}"


EXECUTION FLOW:
    1. atif_adapter converts Harbor trial → Sample with sample.reward = 1.0 or 0.0
    2. sample.response_length = total completion tokens
    3. _post_process_rewards() calls custom_reward_post_process_func(args, samples)
    4. length_shaped_rewards() applies f(response_length) multiplier to successes
    5. Normalized rewards (if enabled) applied after shaping
    6. Final shaped + normalized rewards used in GRPO advantage estimation


AVAILABLE SIGNALS AT HOOK TIME:
    ✓ sample.reward: original binary reward (0.0 or 1.0)
    ✓ sample.response_length: completion token count (primary length signal)
    ✓ sample.loss_mask: learnable span mask
    ✓ sample.effective_response_length: property = sum(loss_mask) or response_length
    ✓ sample.metadata: harbor context (num_turns, chunks, trial_uri for per-task tuning)
    ✓ sample.tokens: full sequence
    ✓ sample.status: COMPLETED/TRUNCATED/FAILED
    ✓ args.*: any command-line argument for tuning


INTERACTION WITH EXISTING HOOKS:
    - Turn-grouped reward (if harbor_sample_mode="turn_grouped"):
      Runs BEFORE length shaping. Length shaping sees already-grouped samples.
    
    - Partial/CTRF reward (if enabled):
      Replace entirely with length shaping (mutually exclusive).
    
    - Normalization (grpo_std_normalization):
      Runs AFTER custom post-process. Length shaping outputs can be normalized.


WHY THIS IS THE CLEANEST APPROACH:
    1. Centralized logic: All reward shaping in one function
    2. Full Sample visibility: response_length + metadata available
    3. GRPO-compatible: Returns (raw, processed) tuple matching other hooks
    4. Configurable: Easy to add tuning hyperparams (alpha, min_factor, mode)
    5. Testable: Deterministic function, no I/O dependencies
    6. Non-invasive: Doesn't modify atif_adapter or core rollout logic
    7. Disable-friendly: Set CUSTOM_REWARD_POST_PROCESS_PATH="" to turn off


ALTERNATIVE APPROACHES (less recommended):
    
    Option A: Modify atif_adapter directly
      ✗ Tight coupling to Harbor plugin
      ✗ Harder to test and iterate
      ✗ Harder to A/B test (reward vs. no reward shaping)
    
    Option B: Modify Sample before post-process (in rollout.py)
      ✗ Core framework change, not plugin-friendly
      ✗ Breaks separation of concerns
    
    Option C: Modify training loss (post-reward, pre-advantage)
      ✗ Happens too late, after GRPO grouping/normalization
      ✗ Hard to reason about final gradient flow


================================================================================
FUNCTION SIGNATURE REFERENCE (for implementation)
================================================================================

REQUIRED SIGNATURE:

    def length_shaped_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]

PARAMETERS:
    args: SimpleNamespace or argparse.Namespace
        - Has .reward_key (default None) for dict-based rewards
        - Has .advantage_estimator (default "grpo") for normalization logic
        - Has .rewards_normalization (default True) for post-normalization
        - Custom args can be added: args.custom_length_shaping_*
    
    samples: list[Sample]
        - Each has .reward (from binary checker)
        - Each has .response_length (completion token count)
        - Each has .loss_mask (optional, learnable span)
        - Each has .metadata dict (contextual info)
    
    **kwargs: reserved for future extensions

RETURN VALUE:
    tuple[list[float], list[float]]
    
    Element 0: raw_rewards
      - Original reward values (before shaping)
      - Used for logging/metrics
      - One entry per sample, matches input order
    
    Element 1: processed_rewards
      - Shaped reward values (length-adjusted)
      - Passed to advantage estimation
      - One entry per sample, matches input order


EXAMPLE TEST CALL:

    from slime_plugins.harbor.length_based_reward_shaping import length_shaped_rewards
    from slime.utils.types import Sample
    from types import SimpleNamespace
    
    args = SimpleNamespace(
        reward_key=None,
        advantage_estimator="grpo",
        custom_length_shaping_alpha=0.01,
        custom_length_shaping_min_factor=0.1,
    )
    
    samples = [
        Sample(reward=1.0, response_length=100, loss_mask=[1]*100),  # Success, 100 tokens
        Sample(reward=1.0, response_length=50, loss_mask=[1]*50),    # Success, 50 tokens
        Sample(reward=0.0, response_length=200, loss_mask=[1]*200),  # Failure, 200 tokens
    ]
    
    raw, shaped = length_shaped_rewards(args, samples)
    # raw:    [1.0, 1.0, 0.0]
    # shaped: [~0.91, ~0.95, 0.0]  # Shorter gets higher factor


================================================================================
END OF REPORT
================================================================================
