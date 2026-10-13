#!/bin/bash

set -x

tmux kill-session -t slime_qwen35_launcher 2>/dev/null || true
tmux kill-session -t slime_qwen35_worker 2>/dev/null || true

pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 -f "raylet|gcs_server|dashboard_agent|runtime_env_agent|log_monitor" 2>/dev/null || true
pkill -9 -f "python3 train_async.py|python train_async.py" 2>/dev/null || true

sleep 3
pgrep -af "raylet|gcs_server|dashboard_agent|runtime_env_agent|log_monitor|train_async.py|SGLangEngine|sglang" || true
