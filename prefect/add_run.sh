REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
prefect deployment run "EMU-Time-Sync/emu-local" \
--params "$(cat "$REPO_DIR/prefect/params.json")" \
--flow-run-name YFP-0082_ART
