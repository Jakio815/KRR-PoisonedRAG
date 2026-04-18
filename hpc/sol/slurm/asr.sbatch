#!/usr/bin/env bash
#SBATCH -J poisonedrag_asr
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-04:00:00
#SBATCH -p htc
#SBATCH -q public
#SBATCH -G 1
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"

bash hpc/sol/scripts/run_asr.sh
