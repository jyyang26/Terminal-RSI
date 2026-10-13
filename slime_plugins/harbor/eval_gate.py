from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def validate_baseline_eval_gate(args, metrics):
    """Reject a malformed/out-of-band baseline eval before training starts."""
    if not bool(getattr(args, "harbor_eval_gate_enabled", False)):
        return
    metrics = metrics or {}
    errors = []
    expected_tasks = int(getattr(args, "harbor_eval_gate_expected_tasks", 0) or 0)
    expected_attempts = int(getattr(args, "harbor_eval_gate_expected_attempts", 0) or 0)
    if expected_tasks and int(metrics.get("harbor/eval_num_tasks", -1)) != expected_tasks:
        errors.append(f"eval_num_tasks={metrics.get('harbor/eval_num_tasks')} expected={expected_tasks}")
    if expected_attempts and int(metrics.get("harbor/eval_num_samples", -1)) != expected_attempts:
        errors.append(f"eval_num_samples={metrics.get('harbor/eval_num_samples')} expected={expected_attempts}")
    if int(metrics.get("harbor/eval_num_missing_attempts", -1)) != 0:
        errors.append(f"eval_num_missing_attempts={metrics.get('harbor/eval_num_missing_attempts')} expected=0")
    coverage = float(metrics.get("harbor/eval_attempt_coverage", 0.0) or 0.0)
    if coverage < 0.999999:
        errors.append(f"eval_attempt_coverage={coverage} expected=1.0")
    k = int(metrics.get("harbor/eval_pass_at_k", 1) or 1)
    rate = float(
        metrics.get(f"harbor/eval_resolved_rate_full_pass@{k}", metrics.get("harbor/eval_resolved_rate_full", -1.0))
        or 0.0
    )
    pass1 = float(
        metrics.get("harbor/eval_resolved_rate_full_pass@1", metrics.get("harbor/eval_pass_at_1_first_attempt_full", -1.0))
        or 0.0
    )
    if rate + 1e-12 < pass1:
        errors.append(f"pass@k={rate} < first-attempt pass@1={pass1}")
    minimum = float(getattr(args, "harbor_eval_gate_min_resolved_rate_full", 0.0) or 0.0)
    maximum = float(getattr(args, "harbor_eval_gate_max_resolved_rate_full", 1.0) or 1.0)
    if not minimum <= rate <= maximum:
        errors.append(f"pass@k={rate} outside [{minimum}, {maximum}]")
    if errors:
        raise RuntimeError("Baseline Harbor eval gate failed: " + "; ".join(errors))
    logger.info(
        "Baseline Harbor eval gate passed: pass@k=%.6f, first-attempt pass@1=%.6f, coverage=%.3f",
        rate,
        pass1,
        coverage,
    )
