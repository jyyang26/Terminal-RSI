# TB-GAE pre-patch baseline

Captured from the actual dirty working tree on 2026-09-27 **before this TB-GAE implementation**.
These are source snapshots, not generated expected outputs. `manifest.json` records SHA-256 hashes.
The offline regression tests execute the frozen adapter, postprocessor, GAE dispatcher and loss functions.
The Ray methods are extracted verbatim by AST to avoid constructing a second Ray actor class.
Native PPO, CP helpers, Sample and DP scheduling remain unchanged and are checked against their snapshots.
Do not regenerate these fixtures from the patched implementation: that would invalidate the comparison.

The snapshots intentionally include pre-existing local changes; upstream Git HEAD was not the running baseline.
They are not imported by production code.
