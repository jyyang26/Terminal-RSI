from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "examples/harbor_terminal_rl/tb2_pass3_gate.py"


def _metrics(**overrides):
    data = {
        "harbor/eval_pass_at_k": 3,
        "harbor/eval_num_tasks": 89,
        "harbor/eval_num_samples": 267,
        "harbor/eval_num_missing_attempts": 0,
        "harbor/eval_attempt_coverage": 1.0,
        "harbor/eval_resolved_rate_full_pass@3": 36 / 89,
        "harbor/eval_resolved_rate_full_avg@3": 93 / 267,
        "harbor/eval_resolved_rate_full_pass@1": 30 / 89,
    }
    data.update(overrides)
    return data


def test_cli_gate_accepts_complete_plausible_pass3(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_metrics()))
    proc = subprocess.run(
        [sys.executable, str(GATE), str(path), "--min", "0.35", "--max", "0.45"],
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "pass@3" in proc.stdout and "avg@3" in proc.stdout and "pass@1" in proc.stdout


def test_cli_gate_rejects_old_fake_pass3(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_metrics(**{
        "harbor/eval_num_samples": 87,
        "harbor/eval_num_missing_attempts": 180,
        "harbor/eval_attempt_coverage": 87 / 267,
        "harbor/eval_resolved_rate_full_pass@3": 0.337,
        "harbor/eval_resolved_rate_full_avg@3": 30 / 267,
        "harbor/eval_resolved_rate_full_pass@1": 0.348,
    })))
    proc = subprocess.run(
        [sys.executable, str(GATE), str(path), "--min", "0.35", "--max", "0.45"],
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "expected 267" in proc.stdout
    assert "pass@3" in proc.stdout


def test_train_async_gate_uses_named_metrics(monkeypatch):
    # Import lazily; this module initializes framework imports at import time.
    from slime_plugins.harbor.eval_gate import validate_baseline_eval_gate

    args = SimpleNamespace(
        harbor_eval_gate_enabled=True,
        harbor_eval_gate_expected_tasks=89,
        harbor_eval_gate_expected_attempts=267,
        harbor_eval_gate_min_resolved_rate_full=0.35,
        harbor_eval_gate_max_resolved_rate_full=0.45,
    )
    validate_baseline_eval_gate(args, _metrics())
    with pytest.raises(RuntimeError, match="Baseline Harbor eval gate failed"):
        validate_baseline_eval_gate(
            args,
            _metrics(**{"harbor/eval_resolved_rate_full_pass@3": 0.33}),
        )
