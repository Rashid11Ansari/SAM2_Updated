#!/bin/bash
# Run the app on a Slurm GPU node (University of Rostock cluster) and push its port
# back to the login node. From the login node, inside the repo folder:
#   srun --partition=gpu-node-mig --gres=gpu:1g.33gb:1 --cpus-per-task=4 --mem=32G --time=04:00:00 --pty bash slurm/run_app.sh
# Then on your laptop:  ssh -N -L 8917:localhost:8917 slurm   and open http://localhost:8917
# One-time prerequisite: a cluster-internal SSH key so the node can reach sl-li (see README).
PORT=${SAM2_PORT:-8917}
LOGIN=${SAM2_LOGIN:-sl-li}
cd "$(dirname "$0")/.."
source .venv/bin/activate
if [ -d /scratch ]; then
  export TMPDIR=/scratch/$USER/tmp; mkdir -p "$TMPDIR"
  export GRADIO_TEMP_DIR=$TMPDIR/gradio
fi
echo "Node: $(hostname)  |  port: $PORT"
ssh -N -o ExitOnForwardFailure=yes -o BatchMode=yes -R ${PORT}:localhost:${PORT} $LOGIN &
TUNNEL_PID=$!
trap 'kill $TUNNEL_PID 2>/dev/null' EXIT
sleep 3
kill -0 $TUNNEL_PID 2>/dev/null || { echo "Reverse tunnel failed (port $PORT taken, or no internal SSH key). Retry with SAM2_PORT=89xx"; exit 1; }
python app.py --port $PORT ${SAM2_MODEL:+--model $SAM2_MODEL}
