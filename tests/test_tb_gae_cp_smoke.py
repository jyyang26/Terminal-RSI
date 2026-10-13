import time

import pytest
import torch.multiprocessing as mp

from tb_gae_cp_worker import worker

NUM_GPUS = 0
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("cp_size", [1, 4])
@pytest.mark.parametrize("gamma", [1.0, 0.999])
@pytest.mark.parametrize("interleaved", [False, True])
def test_gloo_cp_roundtrip(tmp_path, cp_size, gamma, interleaved):
    context = mp.spawn(
        worker, args=(cp_size, (tmp_path / "rendezvous").as_uri(), gamma, interleaved), nprocs=cp_size, join=False
    )
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("TB-GAE CP test exceeded 120s (possible collective mismatch)")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
