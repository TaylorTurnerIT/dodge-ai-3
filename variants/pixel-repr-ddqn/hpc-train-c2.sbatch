#!/bin/bash
#SBATCH --job-name=hpc-train-c2
#SBATCH --account=ai4200-001-2026f
#SBATCH --partition=batch-warp
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out

# Second HPC training chunk: 14336 -> 20000 (single GPU). Base triple is
# the verified c1 run dir; driver re-verifies digests before training.
set -u
ROOT="$HOME/hpc-probe2"
export VIRTUAL_ENV="$ROOT/venv"
export PATH="$ROOT/venv/bin:$HOME/.cargo/bin:$PATH"
export OMP_NUM_THREADS=4
export PATH="$HOME/.local/bin:$PATH"
if ! "$ROOT/venv/bin/python" -c "import torch, gymnasium, einops; import transformers; assert transformers.__version__.startswith(\"4.\")" 2>/dev/null; then
  echo "=== installing python packages ==="
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv pip install --python "$ROOT/venv/bin/python" -q torch numpy gymnasium transformers==4.57.6 einops Pillow
fi
"$ROOT/venv/bin/python" -c "import torch, dodge_native, gymnasium; print('deps ok', torch.cuda.get_device_name(0))"

export HPC_TRAIN_CODE="$ROOT/code/src"
export HPC_TRAIN_DATASET="$ROOT/large-practice-20260914-v2"
export HPC_TRAIN_BASE="$ROOT/chunks/history/lewm-hpc-20260917-v1-c1"
export HPC_TRAIN_HISTORY="$ROOT/chunks/history"
export HPC_TRAIN_RUN_ID="lewm-hpc-20260918-v1-c2"
export HPC_TRAIN_EXTRA="5664"
export HPC_TRAIN_EVERY="1024"
export HPC_TRAIN_BATCH="64"
export HPC_TRAIN_PRECISION="bf16"
"$ROOT/venv/bin/python" "$ROOT/hpc_train_chunk.py"
echo "=== done ==="; date
