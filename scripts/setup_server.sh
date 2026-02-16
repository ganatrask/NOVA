#!/bin/bash
# GR00T Training Server Setup - One-command setup for GPU servers
# Usage: curl -O https://raw.githubusercontent.com/ganatrask/NOVA/main/scripts/setup_server.sh && bash setup_server.sh

set -e

echo "=============================================="
echo "GR00T Training Server Setup"
echo "=============================================="

if ! command -v conda &> /dev/null; then
    echo "[1/7] Installing Miniconda..."
    cd ~
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    rm Miniconda3-latest-Linux-x86_64.sh
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    source ~/.bashrc
else
    echo "[1/7] Miniconda found"
    eval "$(conda shell.bash hook)" 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
fi

if conda env list | grep -q "^groot "; then
    echo "[2/7] Activating existing 'groot' environment..."
    conda activate groot
else
    echo "[2/7] Creating 'groot' environment..."
    conda create -n groot python=3.10 -y
    conda activate groot
fi

WORK_DIR="${WORK_DIR:-$HOME/groot_training}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ ! -d "NOVA" ]; then
    echo "[3/7] Cloning NOVA..."
    git clone --recurse-submodules https://github.com/ganatrask/NOVA.git
else
    echo "[3/7] Updating NOVA..."
    cd NOVA && git pull && cd ..
fi

if [ ! -d "Isaac-GR00T" ]; then
    echo "      Cloning Isaac-GR00T..."
    git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
fi

echo "[4/7] Installing dependencies..."
cd "$WORK_DIR/Isaac-GR00T"

pip install torch==2.7.1 torchvision==0.22.1
python -c "import torch; print(f'  PyTorch {torch.__version__}')"

FLASH_ATTN_WHEEL="flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${FLASH_ATTN_WHEEL}"

cd /tmp
if wget -q "$FLASH_ATTN_URL" 2>/dev/null; then
    pip install "$FLASH_ATTN_WHEEL" && rm -f "$FLASH_ATTN_WHEEL"
else
    pip install ninja
    pip install flash-attn==2.7.4.post1 --no-build-isolation 2>/dev/null || echo "  flash-attn failed, training will be slower"
fi
cd "$WORK_DIR/Isaac-GR00T"

pip install -e .
pip install omegaconf wandb einops pyarrow decord "numpy==1.26.4"

echo "[5/7] Applying Reachy2 patch..."
cd "$WORK_DIR/Isaac-GR00T"

if grep -q "REACHY2" gr00t/data/embodiment_tags.py 2>/dev/null; then
    echo "  Already patched"
else
    patch -p1 < ../NOVA/patches/add_reachy2_embodiment.patch
fi

if grep -q 'video_backend: str = "torchcodec"' gr00t/configs/data/data_config.py 2>/dev/null; then
    sed -i 's/video_backend: str = "torchcodec"/video_backend: str = "decord"/' gr00t/configs/data/data_config.py
fi

echo "[6/7] Copying configs..."
cp ../NOVA/scripts/*.sh ./scripts/ 2>/dev/null || true
cp -r ../NOVA/configs ./ 2>/dev/null || true
chmod +x ./scripts/*.sh

echo "[7/7] Configuring HuggingFace cache..."
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

if ! grep -q "HF_HOME" ~/.bashrc; then
    echo 'export HF_HOME="$HOME/.cache/huggingface"' >> ~/.bashrc
    echo 'export HF_HUB_CACHE="$HF_HOME/hub"' >> ~/.bashrc
fi

echo ""
echo "=============================================="
echo "Verifying setup..."
echo "=============================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from gr00t.data.embodiment_tags import EmbodimentTag; print(f'REACHY2: {EmbodimentTag.REACHY2.value}')" 2>/dev/null || echo "REACHY2 tag: NOT FOUND"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. huggingface-cli login"
echo "  2. cd $WORK_DIR/Isaac-GR00T && ./scripts/train_groot_cloud.sh"
echo ""
