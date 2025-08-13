
#!/bin/bash

# Usage: ./rsync_to_remote.sh <REMOTE_IP>
# Example: ./rsync_to_remote.sh 3.141.200.118

# ---- Configuration ----
REMOTE_IP="$1"
REMOTE_USER="ubuntu"
REMOTE_PATH="/home/ubuntu"
SSH_KEY="$HOME/.ssh/id_rsa"

# Source paths
MODEL_CHECKPOINT="openwebtext_pretok_tokens.pkl"
TRAIN_SCRIPT="/Users/michaelli/Documents/CS336_Assignment/train.py"
CS336_BASICS_DIR="/Users/michaelli/Documents/CS336_Assignment/cs336_basics"

# ---- Functions ----
function usage() {
  echo "Usage: $0 <REMOTE_IP>"
  echo "Example: $0 8.8.8.8"
}

function sync_file() {
  local SRC_PATH="$1"
  local DEST_PATH="$2"
  echo "Syncing $SRC_PATH to $DEST_PATH"
  rsync -avz --progress -e "ssh -i $SSH_KEY" "$SRC_PATH" "$DEST_PATH"
}

# ---- Main ----
if [ -z "$REMOTE_IP" ]; then
  usage
  exit 1
fi

DEST="$REMOTE_USER@$REMOTE_IP:$REMOTE_PATH"

# Sync files and directories
sync_file "$MODEL_CHECKPOINT" "$DEST"
sync_file "$TRAIN_SCRIPT" "$DEST"
sync_file "$CS336_BASICS_DIR" "$DEST"

echo "Sync complete to $DEST"
