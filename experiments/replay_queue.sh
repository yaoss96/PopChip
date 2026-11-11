#!/bin/sh
# Sequential LEGOSim replay queue.
#
# Each line of $1 is "PARAMS|CASE:METHOD:LAYOUT|OUTDIR", with paths relative to
# the repository root (see make_replay_queue.py).  Environment:
#   LEGOSIM_ROOT  LEGOSim checkout that contains benchmark/All2all-MoE-E2E
#                 (copy legosim/All2all-MoE-E2E from this repository there)
#   PYTHON        interpreter for the replay driver (default: python3)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
: "${LEGOSIM_ROOT:?set LEGOSIM_ROOT to your LEGOSim checkout}"
cd "$ROOT" || exit 1
while IFS='|' read -r params layout outdir; do
  [ -z "$params" ] && continue
  echo "=== $(date +%T) $layout"
  "$PY" -u src/replay_legosim_solutions.py --params "$params" --layout "$layout" \
    --out-dir "$outdir" --legosim-root "$LEGOSIM_ROOT" --rounds 2 \
    --moe-e2e-timeout 43200 --legosim-timeout 50400 \
    --moe-gpgpusim-max-parallel 4 < /dev/null > "$outdir.log" 2>&1
  tail -n 1 "$outdir/results.csv" | cut -d, -f1-3,17-19
done < "$1"
echo "QUEUE DONE $(date +%T)"
