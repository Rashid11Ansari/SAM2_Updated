# SAM2 Point-Prompt Video Labeling

Upload a video, click points (or draw boxes) on a frame to mark objects, and Meta's
[SAM2](https://github.com/facebookresearch/sam2) tracks the masks through the whole video.
Export as an overlay video, PNG masks, and COCO JSON — all from one web page (Gradio).

There are three ways to run it. Pick one:

| | Where it runs | GPU | Setup |
|---|---|---|---|
| **A. Google Colab** | Google's servers | free T4 | nothing to install |
| **B. Your own computer** | your laptop / PC | NVIDIA GPU best; Apple Silicon or CPU works (slower) | ~5 min |
| **C. Uni Rostock Slurm cluster** | GPU node (H200) | yes | needs cluster account |

---

## A. Google Colab (easiest)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rashid11Ansari/SAM2_Updated/blob/main/SAM2_Updated.ipynb)

1. Click the badge above.
2. `Runtime → Change runtime type → T4 GPU`.
3. `Runtime → Run all`. Allow Google Drive access when asked (it caches the model weights there so later runs are fast).
4. The last cell prints a `https://….gradio.live` link — open it.

---

## B. Run locally

### 1. Get the code

```bash
git clone https://github.com/Rashid11Ansari/SAM2_Updated.git
cd SAM2_Updated
```

### 2. Install (choose **uv** or **pip**)

**With [uv](https://docs.astral.sh/uv/) (recommended):**

```bash
# install uv once:  macOS/Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh
#                   Windows:      powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync
```

**With pip:**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Needs Python 3.10+ and `git` (used to install SAM2 from GitHub).

> **Windows + NVIDIA GPU:** the default `torch` from PyPI is CPU-only on Windows.
> Install the CUDA build *first*, then the rest:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
> pip install -r requirements.txt
> ```
> (Linux gets a CUDA build by default; macOS uses Apple Silicon automatically.)

### 3. Run

```bash
uv run python app.py               # or, with pip/venv active:  python app.py
```

Open **http://localhost:7860**. The first run downloads the model checkpoint into `checkpoints/` (one time).

Options:

| Flag | Meaning |
|---|---|
| `--model tiny\|small\|base_plus\|large` | model size (default `large`). Use `small`/`tiny` without an NVIDIA GPU |
| `--port 7861` | different port |
| `--share` | also print a public `gradio.live` link (e.g. to open on another device) |
| `--compile` | `torch.compile` for faster tracking — NVIDIA + Linux only, slow first run |

**What speed to expect:** NVIDIA GPU — real time-ish with `large`. Apple Silicon — works, use `small`.
CPU only — very slow (a few seconds per frame with `tiny`); use Colab instead for real videos.

---

## C. Uni Rostock Slurm cluster

On the login node, inside the repo: `uv sync` once (the login node has internet, the GPU nodes may not).

One-time: give the cluster an internal SSH key so the GPU node can tunnel back to the login node:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_cluster -N "" -C "$USER-internal"
cat ~/.ssh/id_cluster.pub >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
printf 'Host sl-li\n    IdentityFile ~/.ssh/id_cluster\n' >> ~/.ssh/config && chmod 600 ~/.ssh/config
```

Then each time — **terminal 1 (cluster):**

```bash
srun --partition=gpu-node-mig --gres=gpu:1g.33gb:1 --cpus-per-task=4 --mem=32G --time=04:00:00 --pty bash slurm/run_app.sh
```

**terminal 2 (your laptop):** `ssh -N -L 8917:localhost:8917 slurm` → open **http://localhost:8917**.
`Ctrl+C` in terminal 1 frees the GPU. If port 8917 is taken, prefix the `srun` with `SAM2_PORT=8931` and use 8931 in terminal 2.

---

## Files

| File | What it is |
|---|---|
| `SAM2_Updated.ipynb` | Colab notebook (option A) |
| `app.py` | same app as a standalone script (options B and C) |
| `requirements.txt` / `pyproject.toml` + `uv.lock` | dependencies for pip / uv |
| `slurm/run_app.sh` | launcher for the Slurm cluster (option C) |
| `checkpoints/` | model weights, downloaded automatically — not in git |

SAM2 is pinned to commit `2b90b9f` so upstream changes can't break the app.
