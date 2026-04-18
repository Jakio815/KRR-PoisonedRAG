#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${POISONEDRAG_SOL_CONFIG:-${ROOT_DIR}/hpc/sol/sol.env}"

if [[ -f "${CONFIG_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
  set +a
fi

SBATCH_SCRIPT="${ROOT_DIR}/hpc/sol/slurm/asr.sbatch"
mkdir -p "${ROOT_DIR}/logs/slurm"

PARTITION="${POISONEDRAG_PARTITION:-htc}"
QOS="${POISONEDRAG_QOS:-public}"
TIME_LIMIT="${POISONEDRAG_TIME:-0-04:00:00}"
NODES="${POISONEDRAG_NODES:-1}"
CPUS_PER_TASK="${POISONEDRAG_CPUS_PER_TASK:-8}"
MEMORY="${POISONEDRAG_MEM:-64G}"
GPUS="${POISONEDRAG_GPUS:-1}"
GPU_TYPE="${POISONEDRAG_GPU_TYPE:-}"
ACCOUNT="${POISONEDRAG_ACCOUNT:-}"
CONSTRAINT="${POISONEDRAG_CONSTRAINT:-}"
JOB_NAME="${POISONEDRAG_JOB_NAME:-poisonedrag_asr}"
EXPORT_MODE="${POISONEDRAG_EXPORT_MODE:-NONE}"

if [[ -n "${GENAI_API_KEY:-}" || -n "${GOOGLE_API_KEY:-}" || -n "${POISONEDRAG_GENAI_API_KEY:-}" ]]; then
  EXPORT_MODE="ALL,GENAI_API_KEY,GOOGLE_API_KEY,POISONEDRAG_GENAI_API_KEY"
fi

SBATCH_ARGS=(
  --job-name "${JOB_NAME}"
  --partition "${PARTITION}"
  --qos "${QOS}"
  --time "${TIME_LIMIT}"
  --nodes "${NODES}"
  --cpus-per-task "${CPUS_PER_TASK}"
  --mem "${MEMORY}"
  --export "${EXPORT_MODE}"
)

if [[ -n "${GPU_TYPE}" ]]; then
  SBATCH_ARGS+=(--gpus "${GPU_TYPE}:${GPUS}")
else
  SBATCH_ARGS+=(--gpus "${GPUS}")
fi

if [[ -n "${ACCOUNT}" ]]; then
  SBATCH_ARGS+=(--account "${ACCOUNT}")
fi

if [[ -n "${CONSTRAINT}" ]]; then
  SBATCH_ARGS+=(--constraint "${CONSTRAINT}")
fi

if [[ "$#" -gt 0 ]]; then
  SBATCH_ARGS+=("$@")
fi

cd "${ROOT_DIR}"

echo "Submitting job:"
printf '  sbatch'
printf ' %q' "${SBATCH_ARGS[@]}"
printf ' %q\n' "${SBATCH_SCRIPT}"

sbatch "${SBATCH_ARGS[@]}" "${SBATCH_SCRIPT}"
