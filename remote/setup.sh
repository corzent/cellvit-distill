#!/bin/bash
# Setup script for RTX 5090 vast.ai / RunPod instance
# Run on remote: bash setup.sh
set -e

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
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
# PyTorch with CUDA 12.4 for Blackwell support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install python-docx markitdown

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
        wget -q https://nuke.warwick.ac.uk/static/files/fold_${fold}.zip
        unzip -q fold_${fold}.zip
        mv "Fold ${fold}" fold${fold} 2>/dev/null || true
        rm fold_${fold}.zip
    fi
done
cd ../..

# ============ 7. Download teacher checkpoint ============
mkdir -p checkpoints
if [ ! -f checkpoints/CellViT-SAM-H-x40.pth ]; then
    echo "Downloading CellViT-SAM-H checkpoint..."
    # Via HF: requires login OR direct mirror if exists
    # Replace with actual URL from CellViT repo releases
    wget -q -O checkpoints/CellViT-SAM-H-x40.pth \
        https://github.com/TIO-IKIM/CellViT/releases/download/v1.0/CellViT-SAM-H-x40.pth
fi

# ============ 8. Precompute teacher outputs (on 5090, ~20 minutes total) ============
export PYTHONPATH="$(pwd)/vendor/CellViT:$PYTHONPATH"
if [ ! -d datasets/pannuke/soft_targets ] || [ "$(ls datasets/pannuke/soft_targets | wc -l)" -lt 7000 ]; then
    echo "Precomputing soft targets..."
    python -m cellvit_distill.scripts.precompute_soft_targets \
        --data_dir datasets/pannuke \
        --checkpoint checkpoints/CellViT-SAM-H-x40.pth \
        --output_dir datasets/pannuke/soft_targets \
        --batch_size 8  # 5090 handles batch 8 in fp16 easily
fi

if [ ! -d datasets/pannuke/soft_features ] || [ "$(ls datasets/pannuke/soft_features | wc -l)" -lt 7000 ]; then
    echo "Precomputing dense features..."
    python -m cellvit_distill.scripts.precompute_features \
        --data_dir datasets/pannuke \
        --checkpoint checkpoints/CellViT-SAM-H-x40.pth \
        --output_dir datasets/pannuke/soft_features \
        --batch_size 8
fi

echo ""
echo "============================================"
echo "Setup complete. Ready for experiments."
echo "============================================"
echo ""
echo "Next steps (run inside tmux to survive disconnect):"
echo "  tmux new -s train"
echo "  # then any of:"
echo "  bash remote/run_3fold_cv.sh"
echo "  bash remote/run_sam3_distill.sh"
