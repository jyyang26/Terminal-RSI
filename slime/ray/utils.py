# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ray/utils.py#L1
import logging
import os
import time

import ray
import torch
from slime.ray.ray_actor import RayActor


# Refer to
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]


def ray_noset_visible_devices(env_vars=os.environ):
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


@ray.remote
class Lock(RayActor):
    """Mutex guarding the SGLang weight broadcast, with a stale-holder escape hatch.

    WHY THE TIMEOUT EXISTS (sh9 122B PPO, 2026-08-21): the previous version held only a
    bare ``_locked`` bool -- no owner, no timestamp, no timeout. If a holder died or stalled
    between ``acquire`` and ``release``, the lock stayed True forever and every other rank
    span in ``while not ray.get(lock.acquire.remote()): time.sleep(0.1)``
    (update_weight_from_distributed.py:260-262) with no diagnostic and no recovery. Observed
    exactly that: one PP-source rank stuck in ``broadcast`` while holding the lock, the other
    PP-source ranks parked at the GLOO barrier in ``_send_weights``, and rank 0 spinning
    indefinitely. The job burned 128 GPUs until killed by hand.

    Under PPO this path runs every step (``use_critic`` forces ``offload_train``), so the
    exposure is large. The broadcast itself is minutes at most, so a holder still holding
    after ``timeout_s`` is stuck, not slow -- breaking the lock is strictly better than
    hanging forever, and it is logged loudly so the real stall is still visible.
    """

    def __init__(self, timeout_s: float | None = None):
        self._locked = False  # False: unlocked, True: locked
        self._owner = None
        self._acquired_at = None
        # Generous by default: the weight broadcast is minutes, so 30 min means "stuck".
        self._timeout_s = float(os.environ.get("SLIME_ENGINE_LOCK_TIMEOUT_S", "1800")) if timeout_s is None else timeout_s

    def acquire(self, owner=None):
        """Try to acquire. Returns True if acquired, False otherwise.

        Caller should retry until it returns True. ``owner`` is optional and only used to
        make a broken-lock message actionable; passing nothing preserves the old signature.
        """
        if self._locked and self._acquired_at is not None and self._timeout_s > 0:
            held_for = time.time() - self._acquired_at
            if held_for > self._timeout_s:
                logging.getLogger(__name__).error(
                    "engine lock FORCIBLY BROKEN: held %.0fs by owner=%r (timeout %.0fs). "
                    "The previous holder stalled or died mid-broadcast; it did not release. "
                    "Investigate that rank -- this recovery keeps the job alive but the "
                    "underlying stall is the real fault.",
                    held_for,
                    self._owner,
                    self._timeout_s,
                )
                self._locked = False
                self._owner = None
                self._acquired_at = None

        if not self._locked:
            self._locked = True
            self._owner = owner
            self._acquired_at = time.time()
            return True
        return False

    def release(self):
        """Release the lock, allowing others to acquire."""
        # Deliberately NOT an assert any more: if the lock was force-broken above, the old
        # holder eventually calls release() and an assert here would crash the job for a
        # condition we already recovered from.
        if not self._locked:
            logging.getLogger(__name__).warning(
                "engine lock released while already unlocked (owner=%r); it was most likely "
                "force-broken after a timeout. Ignoring.",
                self._owner,
            )
            return
        self._locked = False
        self._owner = None
        self._acquired_at = None
