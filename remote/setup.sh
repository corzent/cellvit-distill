#!/bin/bash
# Setup script for RTX 5090 vast.ai / RunPod instance.
# Idempotent: safe to re-run; skips work that is already done.
# Run on remote: bash setup.sh
set -e
set -o pipefail

# ============ 0. Sanity checks ============
echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "== CUDA =="
nvcc --version | grep release || echo "nvcc not installed (ok if image has torch)"

# ============ 1. System deps ============
apt-get update -qq
apt-get install -y -qq git wget curl tmux htop build-essential

# ============ 2. uv (fast python) ============
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# ============ 3. Clone repo ============
cd /workspace
if [ ! -d cellvit-distill ]; then
    git clone https://github.com/corzent/cellvit-distill.git
fi
cd cellvit-distill

# ============ 4. Python environment ============
if [ ! -d .venv ]; then
    uv venv --python 3.13
fi
source .venv/bin/activate

# Skip torch install if a working CUDA-enabled torch is already present
# (e.g. pre-baked vast.ai/RunPod image). Installing again risks downgrading
# the CUDA wheel and breaking the driver match.
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    uv pip install -r requirements.txt
    # PyTorch with CUDA 12.4 for Blackwell support
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
    echo "Existing torch with CUDA detected — keeping it; only topping up project deps."
    uv pip install -r requirements.txt
fi
uv pip install python-docx markitdown gdown
# vendor/CellViT post-proc module imports numba; not in requirements.txt.
# vendor/CellViT decoder utils import einops; also not in requirements.txt.
uv pip install numba einops

# Fail fast if torch can't see the GPU (driver/wheel mismatch surfaces here, not 30 min in).
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to torch'; print('torch sees:', torch.cuda.get_device_name())"

# ============ 5. Vendor CellViT (needs git lfs or zip from author repo) ============
if [ ! -d vendor/CellViT ]; then
    mkdir -p vendor
    cd vendor
    git clone https://github.com/TIO-IKIM/CellViT.git
    cd ..
fi

# ============ 6. Download PanNuke ============
# Option A: from HuggingFace (fast, official mirror) — recommended
mkdir -p datasets/pannuke
cd datasets/pannuke
for fold in 1 2 3; do
    if [ ! -f fold${fold}/images.npy ]; then
        echo "Downloading fold ${fold}..."
        wget --show-progress https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_${fold}.zip
        unzip -q fold_${fold}.zip
        mv "Fold ${fold}" fold${fold}
        rm fold_${fold}.zip
        # Warwick zip nests npy files: fold${N}/{images,masks}/fold${N}/{images,masks,types}.npy
        # Code expects them at fold${N}/{images,masks,types}.npy — flatten.
        mv fold${fold}/images/fold${fold}/images.npy fold${fold}/images.npy
        mv fold${fold}/images/fold${fold}/types.npy  fold${fold}/types.npy
        mv fold${fold}/masks/fold${fold}/masks.npy   fold${fold}/masks.npy
        rm -rf fold${fold}/images fold${fold}/masks
    fi
done
cd ../..

# ============ 7. Download teacher checkpoint ============
# SAM-H weights live on the CellViT authors' Google Drive (see vendor/CellViT/README.md).
mkdir -p checkpoints
if [ ! -f checkpoints/CellViT-SAM-H-x40.pth ]; then
    echo "Downloading CellViT-SAM-H checkpoint via gdown..."
    # gdown >=5 dropped --id; pass the file id as a positional arg instead.
    gdown 1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV \
        -O checkpoints/CellViT-SAM-H-x40.pth
fi
if [ "$(stat -c%s checkpoints/CellViT-SAM-H-x40.pth)" -lt 1000000000 ]; then
    echo "ERROR: teacher checkpoint is <1 GB — download likely failed or returned an HTML error page." >&2
    rm -f checkpoints/CellViT-SAM-H-x40.pth
    exit 1
fi

# ============ 8. Precompute teacher outputs (on 5090, ~20 minutes total) ============
export PYTHONPATH="$(pwd)/vendor/CellViT:$PYTHONPATH"
if [ ! -d datasets/pannuke/soft_targets ] || [ "$(ls datasets/pannuke/soft_targets | wc -l)" -lt 7000 ]; then
    echo "Precomputing soft targets..."
    # batch 32: prior run at batch 8 took ~50 min and used only ~4 GB
    # of 32 GB VRAM; batch 32 ~3-4x faster, fits comfortably.
    python -m cellvit_distill.scripts.precompute_soft_targets \
        --data_dir datasets/pannuke \
        --checkpoint checkpoints/CellViT-SAM-H-x40.pth \
        --output_dir datasets/pannuke/soft_targets \
        --batch_size 32
fi

# ============ 9. (Optional) NuLite-T released weights for Hybrid A eval ============
# Set DOWNLOAD_NULITE=1 to also fetch NuLite-T weights from Zenodo (~48 MB).
# Used by cellvit_distill.scripts.eval_nulite to reproduce Hybrid A comparison.
if [ "${DOWNLOAD_NULITE:-0}" = "1" ]; then
    mkdir -p vendor/nulite_checkpoints
    if [ ! -f vendor/nulite_checkpoints/NuLite-T-Weights.pth ]; then
        echo "Downloading NuLite-T weights from Zenodo..."
        wget --show-progress \
            "https://zenodo.org/records/13272655/files/NuLite-T-Weights.pth?download=1" \
            -O vendor/nulite_checkpoints/NuLite-T-Weights.pth
    fi
    # Clone NuLite repo so its model class is importable
    if [ ! -d vendor/NuLite ]; then
        git clone --depth 1 https://github.com/CosmoIknosLab/NuLite.git vendor/NuLite
    fi
    uv pip install torchinfo  # only NuLite needs it
fi

# Note: soft_features precompute is intentionally skipped — not needed for
# baseline + response-KD 3-fold CV. Re-enable when running feature distillation.

echo ""
echo "============================================"
echo "Setup complete. Ready for experiments."
echo "============================================"
echo ""
echo "Next steps (run inside tmux to survive disconnect):"
echo "  tmux new -s train"
echo "  bash remote/run_3fold_cv.sh"
