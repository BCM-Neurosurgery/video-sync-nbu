tmux new -s prefect-worker -d \
  'export PREFECT_API_URL=http://127.0.0.1:4200/api; \
   cd /home/auto/CODE/utils/video-sync-nbu; \
   source ~/miniconda3/bin/activate videosyncnbu; \
   prefect worker start -p default-agent-pool --type process'
