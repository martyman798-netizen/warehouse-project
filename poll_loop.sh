#!/bin/bash
# poll_loop.sh — Runs every 75 minutes, auto-observes/predicts/submits/trains
# for any new active Viking prediction round.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/poll_loop.log"
INTERVAL=4500  # 75 minutes in seconds

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "=== Poll loop started (interval=${INTERVAL}s / 75 min) ==="

LAST_ROUND_ID=""

while true; do
    log "--- Checking for active round ---"

    STATUS=$(cd "$SCRIPT_DIR" && python3 main.py status 2>&1)
    log "Status: $STATUS"

    ROUND_ID=$(echo "$STATUS" | grep "^Round:" | awk '{print $2}')
    IS_ACTIVE=$(echo "$STATUS" | grep "^Active: True")
    QUERIES_USED=$(echo "$STATUS" | grep "^Budget:" | grep -oE '[0-9]+/[0-9]+' | cut -d'/' -f1)

    if [ -z "$IS_ACTIVE" ]; then
        log "No active round — waiting."
    elif [ "$ROUND_ID" = "$LAST_ROUND_ID" ]; then
        log "Round $ROUND_ID already processed — waiting for next round."
    else
        log "New active round: $ROUND_ID (queries used: $QUERIES_USED)"

        if [ "$QUERIES_USED" = "0" ]; then
            log "Running full pipeline (observe + predict)..."
            cd "$SCRIPT_DIR" && python3 main.py run --round-id "$ROUND_ID" >> "$LOG" 2>&1
        else
            log "Queries already spent — running predict only..."
            cd "$SCRIPT_DIR" && python3 main.py predict --round-id "$ROUND_ID" >> "$LOG" 2>&1
        fi

        log "Submitting all seeds..."
        cd "$SCRIPT_DIR" && python3 main.py submit --round-id "$ROUND_ID" >> "$LOG" 2>&1

        log "Running autolearn (train)..."
        cd "$SCRIPT_DIR" && python3 main.py train >> "$LOG" 2>&1

        log "Committing and pushing changes..."
        cd "$SCRIPT_DIR" && git add -A && git commit -m "Auto: R$(date '+%Y%m%d-%H%M') observe+predict+submit+train

https://claude.ai/code/session_01Xgcfa2HAmz6Bq2qsVTkNDA" >> "$LOG" 2>&1
        cd "$SCRIPT_DIR" && git push -u origin claude/viking-prediction-model-WXUyO >> "$LOG" 2>&1

        LAST_ROUND_ID="$ROUND_ID"
        log "Round $ROUND_ID complete."
    fi

    log "Sleeping ${INTERVAL}s until next check..."
    sleep $INTERVAL
done
