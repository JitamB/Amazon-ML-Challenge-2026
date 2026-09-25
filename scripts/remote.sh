#!/usr/bin/env bash
# Work on the SageMaker space from this Mac, through the AWS Toolkit SSH tunnel.
# The remote VS Code window must be connected (it starts the tunnel's local server).
#   scripts/remote.sh push          # sync code/, utils/, doc/, configs to ~/ml2026 on the space
#   scripts/remote.sh push-data     # sync dataset/ (first time only; compressed transfer)
#   scripts/remote.sh pull          # fetch output/ and work/runs/*/{metrics.json,log.txt,...} back
#   scripts/remote.sh run '<cmd>'   # run a shell command in ~/ml2026 on the space
set -euo pipefail
T="$HOME/Library/Application Support/Code/User/globalStorage/amazonwebservices.aws-toolkit-vscode"
export SAGEMAKER_LOCAL_SERVER_FILE_PATH="$T/sagemaker-local-server-info.json"
export PATH="$T/tools/Amazon/sessionmanagerplugin/bin:$PATH"
# Space host comes from scripts/remote.env (git-ignored); see scripts/remote.env.example.
[ -f "$(dirname "$0")/remote.env" ] && source "$(dirname "$0")/remote.env"
HOST="${SM_HOST:?set SM_HOST in scripts/remote.env (copy scripts/remote.env.example)}"
REMOTE_DIR="ml2026"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Same tunnel as ~/.ssh/config, with the connect script's debug output silenced.
CFG="${TMPDIR:-/tmp}/sm_ssh_config"
cat > "$CFG" <<CONF
Host sm_*
    StrictHostKeyChecking accept-new
    ProxyCommand "$T/sagemaker_connect" %n 2>/dev/null
    # Reuse one connection: every new SSM session makes the Toolkit refresh the token in the browser.
    ControlMaster auto
    ControlPath /tmp/sm-ssh-%C
    ControlPersist 4h
CONF
SSH="ssh -F $CFG -o BatchMode=yes -o ConnectTimeout=90 -o ServerAliveInterval=30 -o LogLevel=ERROR"

case "${1:-}" in
  push)
    rsync -az --delete -e "$SSH" --exclude '__pycache__' --exclude '.DS_Store' --exclude 'remote.env' \
      "$ROOT/code" "$ROOT/utils" "$ROOT/doc" "$ROOT/scripts" "$ROOT/README.md" "$HOST:$REMOTE_DIR/" ;;
  push-data)
    rsync -az --partial --stats -e "$SSH" --exclude '.DS_Store' "$ROOT/dataset" "$HOST:$REMOTE_DIR/" ;;
  pull)
    mkdir -p "$ROOT/remote_results"
    rsync -az -e "$SSH" --include '*/' --include 'output/*.tsv' --include 'work/experiments.csv' \
      --include 'work/runs/*/metrics.json' --include 'work/runs/*/log.txt' --include 'work/runs/*/config.yaml' \
      --include 'work/runs/*/feature_importance.csv' --include 'work/runs/*/selection.json' --exclude '*' \
      --prune-empty-dirs "$HOST:$REMOTE_DIR/" "$ROOT/remote_results/" ;;
  run)
    shift; $SSH "$HOST" "cd $REMOTE_DIR && source ~/venv-ber/bin/activate 2>/dev/null; $*" ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac
