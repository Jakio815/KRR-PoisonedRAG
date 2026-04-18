#!/usr/bin/env bash
set -euo pipefail

SOL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOL_REPO_ROOT="$(cd "${SOL_SCRIPT_DIR}/../../.." && pwd)"

sol_load_config() {
  local cfg="${POISONEDRAG_SOL_CONFIG:-${SOL_REPO_ROOT}/hpc/sol/sol.env}"
  if [[ -f "${cfg}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${cfg}"
    set +a
  fi
}

sol_load_modules() {
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi

  if command -v module >/dev/null 2>&1; then
    module purge
    module load "${POISONEDRAG_MAMBA_MODULE:-mamba/latest}"
  fi
}

sol_abs_path() {
  local p="$1"
  if [[ "${p}" = /* ]]; then
    printf "%s\n" "${p}"
  else
    printf "%s/%s\n" "${SOL_REPO_ROOT}" "${p}"
  fi
}

sol_env_path() {
  sol_abs_path "${POISONEDRAG_ENV_PATH:-.envs/poisonedrag-sol}"
}

sol_cache_root() {
  local default_root="${SCRATCH:-${SOL_REPO_ROOT}/.cache}/poisonedrag"
  sol_abs_path "${POISONEDRAG_CACHE_ROOT:-${default_root}}"
}

sol_prepare_runtime_dirs() {
  mkdir -p \
    "${SOL_REPO_ROOT}/logs/slurm" \
    "${SOL_REPO_ROOT}/results/runtime_configs" \
    "${SOL_REPO_ROOT}/results/asr_evaluation"
}

sol_export_cache_env() {
  local cache_root
  cache_root="$(sol_cache_root)"

  mkdir -p \
    "${cache_root}/hf/hub" \
    "${cache_root}/hf/transformers" \
    "${cache_root}/torch" \
    "${cache_root}/tmp"

  export HF_HOME="${cache_root}/hf"
  export HF_HUB_CACHE="${cache_root}/hf/hub"
  export HUGGINGFACE_HUB_CACHE="${cache_root}/hf/hub"
  export TRANSFORMERS_CACHE="${cache_root}/hf/transformers"
  export TORCH_HOME="${cache_root}/torch"
  export XDG_CACHE_HOME="${cache_root}"
  export TMPDIR="${cache_root}/tmp"
  export TEMP="${cache_root}/tmp"
  export TMP="${cache_root}/tmp"
}

sol_activate_env() {
  local env_path
  env_path="$(sol_env_path)"
  if [[ ! -d "${env_path}" ]]; then
    echo "ERROR: environment not found at ${env_path}" >&2
    echo "Run: bash hpc/sol/scripts/setup_env.sh" >&2
    exit 2
  fi

  source activate "${env_path}"
}
