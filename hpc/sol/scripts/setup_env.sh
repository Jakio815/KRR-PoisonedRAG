#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_common.sh"

sol_load_config
sol_load_modules
sol_prepare_runtime_dirs
sol_export_cache_env

ENV_PATH="$(sol_env_path)"
PYTHON_VERSION="${POISONEDRAG_PYTHON_VERSION:-3.10}"
PYTORCH_CUDA_VERSION="${POISONEDRAG_PYTORCH_CUDA_VERSION:-12.1}"
FORCE_RECREATE="${POISONEDRAG_FORCE_RECREATE:-0}"
REQUIREMENTS_FILE="${SOL_REPO_ROOT}/hpc/sol/requirements-sol.txt"

echo "Repository root: ${SOL_REPO_ROOT}"
echo "Environment path: ${ENV_PATH}"
echo "Cache root: $(sol_cache_root)"

if [[ "${FORCE_RECREATE}" == "1" && -d "${ENV_PATH}" ]]; then
  echo "Removing existing environment because POISONEDRAG_FORCE_RECREATE=1"
  rm -rf "${ENV_PATH}"
fi

if [[ ! -d "${ENV_PATH}" ]]; then
  echo "Creating mamba environment..."
  mamba create -y -p "${ENV_PATH}" "python=${PYTHON_VERSION}" pip

  echo "Installing CUDA-enabled PyTorch via mamba..."
  mamba install -y -p "${ENV_PATH}" \
    -c pytorch -c nvidia -c conda-forge \
    pytorch torchvision torchaudio "pytorch-cuda=${PYTORCH_CUDA_VERSION}"
else
  echo "Environment already exists, skipping creation."
fi

source activate "${ENV_PATH}"

echo "Installing project dependencies..."
python -m pip install --upgrade pip wheel
python -m pip install -r "${REQUIREMENTS_FILE}"

if [[ "${POISONEDRAG_PREFETCH_CONTRIEVER:-1}" == "1" ]]; then
  echo "Prefetching Contriever model artifacts..."
  python - <<'PY'
from transformers import AutoTokenizer, AutoModel

model_name = "facebook/contriever"
AutoTokenizer.from_pretrained(model_name)
AutoModel.from_pretrained(model_name)
print(f"Prefetched {model_name}")
PY
fi

echo "Environment setup complete."
echo "Next step: bash hpc/sol/submit.sh"
