#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETUP=1
SUBMIT=1
FORWARDED_ARGS=()

print_help() {
  cat <<'EOF'
Usage:
  bash hpc/sol/quickstart.sh [--setup-only|--submit-only] [extra sbatch flags]

Examples:
  bash hpc/sol/quickstart.sh
  bash hpc/sol/quickstart.sh --setup-only
  bash hpc/sol/quickstart.sh --submit-only
  bash hpc/sol/quickstart.sh --submit-only --account grp_mylab
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --setup-only)
      SETUP=1
      SUBMIT=0
      shift
      ;;
    --submit-only)
      SETUP=0
      SUBMIT=1
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      FORWARDED_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "${ROOT_DIR}"

if [[ ! -f "${ROOT_DIR}/hpc/sol/sol.env" ]]; then
  cp "${ROOT_DIR}/hpc/sol/sol.env.example" "${ROOT_DIR}/hpc/sol/sol.env"
  echo "Created hpc/sol/sol.env from template. Edit it if you need custom resources/account."
fi

if [[ "${SETUP}" -eq 1 ]]; then
  echo "=== [1/2] Setting up Sol environment ==="
  bash hpc/sol/scripts/setup_env.sh
fi

if [[ "${SUBMIT}" -eq 1 ]]; then
  echo "=== [2/2] Submitting Slurm job ==="
  bash hpc/sol/submit.sh "${FORWARDED_ARGS[@]}"
  echo "Job submitted. Monitor with: myjobs"
fi
