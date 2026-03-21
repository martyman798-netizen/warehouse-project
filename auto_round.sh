#!/bin/bash
# auto_round.sh — Automatically observe, predict, and submit for any active round.
#
# Setup:
#   1. chmod +x auto_round.sh
#   2. Add your token to a .env file:  echo 'ASTAR_TOKEN=your_token' > .env
#   3. Add to crontab (crontab -e):
#        0 1 * * * /full/path/to/warehouse-project/auto_round.sh
#        0 3 * * * /full/path/to/warehouse-project/auto_round.sh
#      (runs at 1am and 3am — double safety in case Mac wakes late)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/auto_round.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

cd "$SCRIPT_DIR" || exit 1

# Load token from .env if not already set
if [ -z "$ASTAR_TOKEN" ] && [ -f "$SCRIPT_DIR/.env" ]; then
    export ASTAR_TOKEN=$(grep '^ASTAR_TOKEN=' "$SCRIPT_DIR/.env" | cut -d'=' -f2-)
fi

if [ -z "$ASTAR_TOKEN" ]; then
    log "ERROR: ASTAR_TOKEN not set. Add it to .env file."
    exit 1
fi

log "=== Auto-round starting ==="

# Get active round status
STATUS=$(python3 main.py status 2>&1)
log "Status check: $STATUS"

ROUND_ID=$(echo "$STATUS" | grep "^Round:" | awk '{print $2}')
QUERIES_USED=$(echo "$STATUS" | grep "^Budget:" | grep -oE '[0-9]+/[0-9]+' | cut -d'/' -f1)
IS_ACTIVE=$(echo "$STATUS" | grep "^Active: True")

if [ -z "$ROUND_ID" ]; then
    log "Could not determine round ID. Exiting."
    exit 1
fi

if [ -z "$IS_ACTIVE" ]; then
    log "No active round. Nothing to do."
    exit 0
fi

log "Round $ROUND_ID is active. Queries used: $QUERIES_USED"

if [ "$QUERIES_USED" -gt 0 ]; then
    log "Queries already spent — skipping observe, running predict + submit only."
    python3 main.py predict --round-id "$ROUND_ID" >> "$LOG" 2>&1
else
    log "Running full pipeline: observe + predict..."
    python3 main.py run --round-id "$ROUND_ID" >> "$LOG" 2>&1
fi

log "Submitting all seeds..."
python3 main.py submit --round-id "$ROUND_ID" --seed all >> "$LOG" 2>&1

log "=== Done ==="
