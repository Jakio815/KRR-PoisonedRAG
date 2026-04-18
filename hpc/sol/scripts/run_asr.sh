#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_common.sh"

sol_load_config
sol_load_modules
sol_prepare_runtime_dirs
sol_export_cache_env
sol_activate_env

PYTHON_BIN="${POISONEDRAG_PYTHON_EXEC:-python}"
MODEL_NAME="${POISONEDRAG_MODEL_NAME:-palm2}"
MODEL_CONFIG_PATH_RAW="${POISONEDRAG_MODEL_CONFIG_PATH:-model_configs/palm2_config.json}"
MODEL_CONFIG_PATH="$(sol_abs_path "${MODEL_CONFIG_PATH_RAW}")"
RUNTIME_MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH}"

if [[ ! -f "${MODEL_CONFIG_PATH}" ]]; then
  echo "ERROR: model config not found: ${MODEL_CONFIG_PATH}" >&2
  exit 2
fi

if [[ "${MODEL_NAME}" == "palm2" ]]; then
  API_KEY="${POISONEDRAG_GENAI_API_KEY:-${GENAI_API_KEY:-${GOOGLE_API_KEY:-}}}"

  if [[ -n "${API_KEY}" ]]; then
    RUNTIME_MODEL_CONFIG_PATH="${SOL_REPO_ROOT}/results/runtime_configs/palm2_runtime_${SLURM_JOB_ID:-manual}.json"
    "${PYTHON_BIN}" - "${MODEL_CONFIG_PATH}" "${RUNTIME_MODEL_CONFIG_PATH}" "${API_KEY}" <<'PY'
import json
import pathlib
import sys

src, dst, key = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.loads(pathlib.Path(src).read_text(encoding="utf-8"))
api_info = cfg.setdefault("api_key_info", {})
api_info["api_keys"] = [key]
api_info["api_key_use"] = 0
pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(dst).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
PY
  fi

  if grep -qi "Your api key here" "${RUNTIME_MODEL_CONFIG_PATH}"; then
    echo "ERROR: palm2 config still has placeholder API key." >&2
    echo "Set GENAI_API_KEY in your shell or update ${MODEL_CONFIG_PATH_RAW}" >&2
    exit 2
  fi
fi

IFS=' ' read -r -a DATASETS <<< "${POISONEDRAG_DATASETS:-nq hotpotqa msmarco}"

CMD=(
  "${PYTHON_BIN}" "${SOL_REPO_ROOT}/run_asr_simple.py"
  --datasets "${DATASETS[@]}"
  --num_questions "${POISONEDRAG_NUM_QUESTIONS:-10}"
  --num_repeats "${POISONEDRAG_NUM_REPEATS:-10}"
  --seed "${POISONEDRAG_SEED:-42}"
  --model_name "${MODEL_NAME}"
  --model_config_path "${RUNTIME_MODEL_CONFIG_PATH}"
  --eval_model_code "${POISONEDRAG_EVAL_MODEL_CODE:-contriever}"
  --top_k "${POISONEDRAG_TOP_K:-5}"
  --adv_per_query "${POISONEDRAG_ADV_PER_QUERY:-5}"
  --score_function "${POISONEDRAG_SCORE_FUNCTION:-dot}"
  --gpu_id "${POISONEDRAG_GPU_ID:-0}"
  --random_targets "${POISONEDRAG_RANDOM_TARGETS:-True}"
  --reuse_targets_per_repeat "${POISONEDRAG_REUSE_TARGETS_PER_REPEAT:-True}"
  --resume "${POISONEDRAG_RESUME:-True}"
  --timeout_sec "${POISONEDRAG_TIMEOUT_SEC:-0}"
  --query_results_dir "${POISONEDRAG_QUERY_RESULTS_DIR:-asr_eval}"
  --output_dir "${POISONEDRAG_OUTPUT_DIR:-results/asr_evaluation}"
  --python_exec "${PYTHON_BIN}"
)

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "Running ASR command:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"
