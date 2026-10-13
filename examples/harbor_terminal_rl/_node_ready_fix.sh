#!/bin/bash
# Idempotent per-node readiness fix for the 96-GPU 27B Harbor run.
# Remounts the (flaky) shared-storage code share, then verifies apptainer +
# harbor import. Safe to run repeatedly; a healthy node just re-reports OK.
CODE=/path/to/workspace/code
SLIME=$CODE/slime-v0.3-harbor
PP="$CODE/harbor/src:$SLIME"

# 1. Ensure the code share is readable (not just stat-able). listdir is the real test.
if ! python3 -c "import os; os.listdir('$SLIME')" >/dev/null 2>&1; then
  jizhi_client mount --token _enCIzo8hifIn-7tr4CixQ --business_flag AILab_Seattle >/dev/null 2>&1
fi

MNT=N; APPT=N; IMP=N
python3 -c "import os; os.listdir('$SLIME')" >/dev/null 2>&1 && MNT=Y
command -v singularity >/dev/null 2>&1 && APPT=Y
if PYTHONPATH="$PP" python3 -c "from slime_plugins.harbor.import_utils import ensure_harbor_distribution_metadata as e; e(); import slime_plugins.harbor.rollout, slime.backends.sglang_utils, harbor.trial.trial, harbor.agents.terminus_2.terminus_2" >/dev/null 2>&1; then
  IMP=Y
fi
echo "READY mnt:$MNT appt:$APPT imp:$IMP"
