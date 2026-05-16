#!/bin/bash
# Run on YOUR local machine after experiments complete.
# Syncs results from remote vast.ai instance to local for analysis.
#
# Usage:
#   bash sync_back.sh <ssh_user>@<host>:<port>
#   e.g. bash sync_back.sh root@ssh4.vast.ai:12345

REMOTE=${1:?"Pass remote as user@host:port"}
USER_HOST=$(echo $REMOTE | cut -d: -f1)
PORT=$(echo $REMOTE | cut -d: -f2)

LOCAL_ROOT="/home/corzent/caspian/thesis"
REMOTE_ROOT="/workspace/cellvit-distill"

mkdir -p $LOCAL_ROOT/remote_runs

# Sync: model checkpoints (best_model.pth only — heavy others optional)
rsync -avh -e "ssh -p $PORT" \
    --include='*/' \
    --include='best_model.pth' \
    --include='config.yaml' \
    --include='eval_*.json' \
    --exclude='*' \
    $USER_HOST:$REMOTE_ROOT/cellvit_distill/runs/ \
    $LOCAL_ROOT/remote_runs/

# Sync logs
rsync -avh -e "ssh -p $PORT" \
    $USER_HOST:$REMOTE_ROOT/logs/ \
    $LOCAL_ROOT/remote_logs/

echo "Sync complete. Results in:"
echo "  $LOCAL_ROOT/remote_runs/"
echo "  $LOCAL_ROOT/remote_logs/"
