REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
tmux new -s prefect-worker -d \
  "export PREFECT_API_URL=http://127.0.0.1:4200/api; \
   cd $REPO_DIR; \
   prefect worker start -p default-agent-pool --type process"
