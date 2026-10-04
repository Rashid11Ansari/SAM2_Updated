#!/bin/bash
#SBATCH --job-name=sam2-updated
#SBATCH --partition=gpu-node-mig
#SBATCH --gres=gpu:1g.33gb:1           # one ~33 GB MIG slice
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00                # app stays up until this runs out or you scancel
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

# Usage (login node, inside the repo):   uv sync  (once)  ->  mkdir -p logs && sbatch run_slurm.sh
# Laptop:  ssh -N -L PORT:localhost:PORT slurm   (exact line is printed in logs/sam2-updated-<jobid>.out)

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PATH="$HOME/.local/bin:$PATH"   # uv

PORT=${SAM2_PORT:-$(( 20000 + $(id -u) % 10000 ))}   # one port per user -> no clashes on the login node
DATA_DIR=${SAM2_DATA_DIR:-$HOME/sam2-updated-data}  # videos/, outputs/, checkpoints/ (visible from all nodes)
MODEL=${SAM2_MODEL:-large}
LOGIN_NODE=${SAM2_LOGIN:-sl-li}
mkdir -p "$DATA_DIR"/videos "$DATA_DIR"/outputs

# extracted frames go to the node's fast local disk, not the shared home
JOB_SCRATCH=""
if [ -d /scratch ]; then
  JOB_SCRATCH=/scratch/$USER/sam2-${SLURM_JOB_ID:-$$}
  mkdir -p "$JOB_SCRATCH"
  export TMPDIR=$JOB_SCRATCH GRADIO_TEMP_DIR=$JOB_SCRATCH/gradio
fi

# Reverse tunnel: PORT on the login node -> PORT on this compute node
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new \
    -R ${PORT}:127.0.0.1:${PORT} ${LOGIN_NODE} &
TUNNEL_PID=$!
trap 'kill $TUNNEL_PID 2>/dev/null; [ -n "$JOB_SCRATCH" ] && rm -rf "$JOB_SCRATCH"' EXIT   # close tunnel, clean only our scratch dir
sleep 5
kill -0 $TUNNEL_PID 2>/dev/null || { echo "Tunnel failed (cluster SSH key missing, or port $PORT taken -> sbatch --export=ALL,SAM2_PORT=2xxxx run_slurm.sh)"; exit 1; }

echo "Node:            $(hostname)"
echo "Data folder:     $DATA_DIR"
echo "On your laptop:  ssh -N -L ${PORT}:localhost:${PORT} slurm"
echo "Then open:       http://localhost:${PORT}"
nvidia-smi -L

# .venv was built by `uv sync` on the login node; --no-sync = no internet needed on the node
srun uv run --no-sync sam2-updated --host 127.0.0.1 --port ${PORT} --data-dir "${DATA_DIR}" --model "${MODEL}"
