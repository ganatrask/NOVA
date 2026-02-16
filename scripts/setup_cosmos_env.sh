#!/bin/bash
# Setup reachy_cosmos conda environment for Cosmos Reason 2

set -e

ENV_NAME="reachy_cosmos"

echo "=========================================="
echo "Setting up $ENV_NAME environment"
echo "=========================================="

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found"
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment exists."
    read -p "Recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME -y
    else
        conda activate $ENV_NAME
        pip install --upgrade transformers>=5.0.0 flask pillow requests torch
        echo "Done!"
        exit 0
    fi
fi

conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=5.0.0" flask pillow requests accelerate

echo ""
echo "Verifying..."
python -c "
import transformers, torch
print(f'transformers: {transformers.__version__}')
print(f'torch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Start server:  conda activate $ENV_NAME && python scripts/cosmos_server.py --port 8100"
echo ""
