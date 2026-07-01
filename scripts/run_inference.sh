#!/bin/bash
# One-shot launcher: starts inference + keepalive, both survive container restarts
# Usage: bash scripts/run_inference.sh

set -euo pipefail

SCRATCHPAD=/tmp/infer_session
mkdir -p "$SCRATCHPAD"

INFER_LOG="$SCRATCHPAD/infer.log"
KA_LOG="$SCRATCHPAD/keepalive.log"

# --- keepalive ---
cat > "$SCRATCHPAD/keepalive.sh" << 'EOF'
#!/bin/bash
INFER_LOG=/tmp/infer_session/infer.log
while true; do
    ts=$(date '+%H:%M:%S')
    progress=$(tail -1 "$INFER_LOG" 2>/dev/null \
        | grep -oP '\d+/\d+ patches \(\d+%\)' \
        || tail -1 "$INFER_LOG" 2>/dev/null \
        || echo "starting...")
    echo "$ts | $progress"
    touch data/minnie65_1um.zarr/zarr.json 2>/dev/null || true
    sleep 60
done
EOF

nohup bash "$SCRATCHPAD/keepalive.sh" >> "$KA_LOG" 2>&1 &
echo "keepalive PID: $!"

# --- inference ---
nohup python scripts/infer_full_volume.py \
    --checkpoint runs/microns_256/checkpoint_final.pt \
    --zarr data/minnie65_1um.zarr \
    --threshold 0.5 --patch 96 --overlap 16 --min-size 100 \
    >> "$INFER_LOG" 2>&1 &
echo "inference PID: $!"

echo "Logs: $INFER_LOG  |  keepalive: $KA_LOG"
