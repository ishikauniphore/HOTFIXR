#!/bin/bash
set -e

# If there are any issues regarding CUDA, please check out the CUDA_802_FIX.md file for help.

ENV_NAME="verl"
PYTHON_VERSION="3.12"

echo "=== Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION) ==="
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo ""
echo "=== Installing PyTorch 2.7.0 (CUDA 12.6) ==="
# Pin torch so vllm and verl both build against the same version
conda run -n "$ENV_NAME" --no-capture-output \
    pip install torch==2.7.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126

echo ""
# vllm 0.9.0 is the version that targets torch 2.7.0; newer vllm requires torch 2.11+
echo "=== Installing vLLM 0.9.0 ==="
conda run -n "$ENV_NAME" --no-capture-output \
    pip install vllm==0.9.0

echo ""
echo "=== Installing verl 0.4.0 ==="
# verl 0.8+ requires numpy<2.0 but also conflicts with vllm 0.9; pin to 0.4.0
conda run -n "$ENV_NAME" --no-capture-output \
    pip install verl==0.4.0 "numpy<2.0"

echo ""
# vllm/verl dependency resolution can upgrade torch to a different CUDA version from PyPI;
# force-reinstall from cu126 to keep torch/torchvision/torchaudio aligned.
echo "=== Re-pinning PyTorch to CUDA 12.6 (overrides any vllm/verl upgrade) ==="
conda run -n "$ENV_NAME" --no-capture-output \
    pip install torch==2.7.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

echo ""
echo "=== Installing flash-attn 2.8.3 (prebuilt wheel for cu126 + torch2.7 + cp312) ==="
conda run -n "$ENV_NAME" --no-capture-output \
    pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu126torch2.7-cp312-cp312-linux_x86_64.whl

echo ""
echo "=== Installing scikit-learn and hdbscan ==="
conda run -n "$ENV_NAME" --no-capture-output \
    pip install scikit-learn hdbscan

echo ""
echo "=== Installing remaining project dependencies ==="
conda run -n "$ENV_NAME" --no-capture-output \
    pip install \
        "transformers==4.51.3" \
        "trl==0.8.6" \
        accelerate \
        peft \
        sentence-transformers \
        rouge-score \
        evaluate \
        wandb \
        "ray[default]" \
        fastapi \
        uvicorn \
        tabulate

echo ""
echo "=== Creating 'py' symlink (run scripts call 'py' directly) ==="
CONDA_ENV_BIN="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')/bin"
ln -sf python "$CONDA_ENV_BIN/py"
echo "Symlink: $CONDA_ENV_BIN/py -> python"

echo ""
echo "=== Verifying key packages ==="
conda run -n "$ENV_NAME" --no-capture-output python -c "
import torch, vllm, verl, sklearn, hdbscan
print(f'torch:        {torch.__version__}  (CUDA available: {torch.cuda.is_available()})')
print(f'vllm:         {vllm.__version__}')
print(f'verl:         {verl.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
"

echo ""
echo "=== Done! Run: conda activate verl ==="
