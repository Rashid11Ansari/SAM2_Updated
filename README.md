# SAM2 Point-Prompt Video Labeling

Pick a video, click points (or draw boxes) on a frame to mark objects, and Meta's
[SAM2](https://github.com/facebookresearch/sam2) tracks the masks through the whole video.
Export as overlay video, PNG masks and COCO JSON -- all from one web page.

A `uv` Python package: SAM2 logic in `model.py`, web UI (Gradio inside FastAPI/uvicorn) in `app.py`,
started with one command. Runs on a laptop or as a Slurm job on a GPU node.

```
SAM2_Updated/
├── pyproject.toml          # dependencies + the `sam2-updated` command
├── uv.lock                 # exact versions (commit it)
├── run_slurm.sh            # Slurm job: MIG 33 GB slice + reverse tunnel
├── notebooks/              # Colab launcher (runs the same app)
└── src/sam2_updated/
    ├── model.py            # SAM2 + video helpers (no UI, no globals)
    └── app.py              # web layer + main()
```

## 1. Run locally

```bash
git clone git@github.com:Rashid11Ansari/SAM2_Updated.git
cd SAM2_Updated
uv sync                     # first time: installs everything into .venv
uv run sam2-updated         # -> http://127.0.0.1:8000
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir` | `./data` | where videos are read from and exports written to |
| `--port` | `8000` | web port |
| `--host` | `127.0.0.1` | only this machine (reach it from elsewhere via SSH tunnel) |
| `--model` | `large` | `tiny` / `small` / `base_plus` / `large` -- use `small`/`tiny` without an NVIDIA GPU |
| `--checkpoint-dir` | `<data-dir>/checkpoints` | where the SAM2 weights live (downloaded on first start) |
| `--compile` | off | `torch.compile` -- NVIDIA + Linux only, faster tracking after a slow first run |

GPU: NVIDIA (CUDA) is used automatically, Apple Silicon works (slower), CPU only is very slow.
Windows + NVIDIA: PyPI's torch is CPU-only on Windows -- run on the cluster, or install the CUDA build of torch.

## 2. Data folder

```
<data-dir>/
├── videos/        # put videos here -- they appear in the "Video" dropdown (or upload in the browser)
├── outputs/       # every export: <video>_<date-time>/ (labeled_video.mp4, masks/, annotations.json) + .zip
└── checkpoints/   # SAM2 weights
```

Copy videos to the cluster from your laptop: `scp my_video.mp4 slurm:sam2-updated-data/videos/`
Fetch results: `scp -r slurm:sam2-updated-data/outputs/<folder> .`

## 3. SSH setup (once)

**Laptop** -- key + `~/.ssh/config`:

```bash
ssh-keygen -t ed25519 -C "<uni-username>@uni-rostock"
ssh-copy-id <uni-username>@sl-li.informatik.uni-rostock.de
ssh-add --apple-use-keychain ~/.ssh/id_ed25519      # macOS; Linux: eval "$(ssh-agent -s)" && ssh-add
```

```
Host slurm
    HostName sl-li.informatik.uni-rostock.de
    User <uni-username>
    IdentityFile ~/.ssh/id_ed25519
    AddKeysToAgent yes
    ForwardAgent yes        # git clone/push on the cluster with your laptop key
    # ProxyJump <gateway>   # only if you have to hop through a gateway first
```

Add the public key (`cat ~/.ssh/id_ed25519.pub`) to GitHub -> Settings -> SSH and GPG keys, then
`ssh slurm` and `ssh -T git@github.com` should both work without a password.

**Cluster** -- the job must SSH back to the login node without a password. Test:

```bash
srun -p compute-node -t 1 ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new sl-li true
```

If that says `Permission denied`, create a cluster-only key that can only open tunnels:

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519_cluster
echo "restrict,port-forwarding,command=\"echo tunnel-only key\" $(cat ~/.ssh/id_ed25519_cluster.pub)" >> ~/.ssh/authorized_keys
printf '\nHost sl-li\n    IdentityFile ~/.ssh/id_ed25519_cluster\n' >> ~/.ssh/config
chmod 600 ~/.ssh/authorized_keys ~/.ssh/config
```

## 4. Run on the Slurm cluster (GPU job)

```
laptop ──ssh -L──▶ login node sl-li ◀──ssh -R── GPU node: sam2-updated on 127.0.0.1:PORT
```

On the login node, inside the repo:

```bash
uv sync                           # once, needs internet (login node only)
mkdir -p logs && sbatch run_slurm.sh
squeue --me                       # wait for state R
cat $(ls -t logs/sam2-updated-*.out | head -1)   # newest log: shows the ssh -L line and the URL
```

On your laptop: run the printed `ssh -N -L <PORT>:localhost:<PORT> slurm` and open `http://localhost:<PORT>`.
Stop: `scancel <jobid>` (frees the GPU).

`run_slurm.sh` requests one ~33 GB MIG slice (`gpu-node-mig`, `gpu:1g.33gb:1`) for 4 h, uses a port derived from
your user id (no clashes), stores data in `~/sam2-updated-data`, and puts extracted frames on the node's `/scratch`.
Override with environment variables, e.g. `sbatch --export=ALL,SAM2_MODEL=small,SAM2_DATA_DIR=$HOME/mydata run_slurm.sh`
(also `SAM2_PORT`).

## 5. Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rashid11Ansari/SAM2_Updated/blob/main/notebooks/SAM2_Updated.ipynb)

The notebook installs this package from GitHub and starts **the same app** (`launch_colab()`) with a public
`gradio.live` link -- same features as on the cluster, nothing to keep in sync.

1. `Runtime -> Change runtime type -> T4 GPU`, then run the cells top to bottom.
2. With `USE_DRIVE = True`, videos / exports / model weights live in `MyDrive/sam2-data/` and survive the session.
3. Open the printed `https://....gradio.live` link.
