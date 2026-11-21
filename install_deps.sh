#!/usr/bin/env bash
# ==============================================================================
# Dependency Installer & Verifier for Indic / NLLB Fine-Tuning Project
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

# Allow running verification only: ./install_deps.sh --check
CHECK_ONLY="${1:-}"

if [ "${CHECK_ONLY}" != "--check" ]; then
    echo "=== [1/2] Installing dependencies from requirements.txt ==="
    if command -v uv &>/dev/null; then
        echo "Using 'uv' for installation..."
        uv pip install -r "${REQ_FILE}"
    elif command -v pip3 &>/dev/null; then
        pip3 install -r "${REQ_FILE}"
    elif command -v pip &>/dev/null; then
        pip install -r "${REQ_FILE}"
    else
        echo "Error: Neither uv nor pip found."
        exit 1
    fi
fi

echo "=== [2/2] Verifying core project imports ==="
python3 -c "
import torch
import transformers
import peft
import datasets
import sacrebleu
import sklearn
import scipy
import matplotlib
import sentence_transformers

try:
    import IndicTransToolkit
    indic_status = f'OK (v{getattr(IndicTransToolkit, \"__version__\", \"installed\")})'
except ImportError:
    indic_status = 'NOT FOUND (install via: pip install git+https://github.com/VarunGumma/IndicTransToolkit.git)'

print('Installed and verified packages:')
print(f'  ✓ PyTorch:             {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'  ✓ Transformers:        {transformers.__version__}')
print(f'  ✓ PEFT (LoRA):         {peft.__version__}')
print(f'  ✓ Datasets:            {datasets.__version__}')
print(f'  ✓ SacreBLEU:           {sacrebleu.__version__}')
print(f'  ✓ Scikit-Learn:        {sklearn.__version__}')
print(f'  ✓ SciPy:               {scipy.__version__}')
print(f'  ✓ Matplotlib:          {matplotlib.__version__}')
print(f'  ✓ SentenceTransformers:{sentence_transformers.__version__}')
print(f'  ✓ IndicTransToolkit:   {indic_status}')
"

echo "=== Environment check completed successfully! ==="
