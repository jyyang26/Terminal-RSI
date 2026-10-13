import time

import pytest
import torch.multiprocessing as mp

from tb_gae_bootstrap_distributed_worker import worker

NUM_GPUS = 0
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("cp_size", [1, 4])
def test_cross_dp_chunk_continuation_with_real_cp_gloo(cp_size, tmp_path):
    context = mp.spawn(worker, args=(cp_size, (tmp_path / "gloo").as_uri()), nprocs=2 * cp_size, join=False)
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("TB-GAE DP/CP continuation timed out")
    finally:
        for p in context.processes:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)
