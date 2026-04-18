# Sol Supercomputer Run Guide (ASU)

This folder makes the project runnable on ASU Sol with minimal manual steps.

## What This Adds

- `hpc/sol/scripts/setup_env.sh`: creates the Python environment and installs dependencies.
- `hpc/sol/scripts/run_asr.sh`: runs ASR evaluation with Sol-safe defaults.
- `hpc/sol/slurm/asr.sbatch`: batch job entry script.
- `hpc/sol/submit.sh`: submits the job with configurable resources.
- `hpc/sol/quickstart.sh`: one command to setup + submit.
- `hpc/sol/sol.env.example`: editable defaults for resources and evaluation settings.

## One-Time Setup Per Uploaded Folder

1. Upload project folder to Sol.
2. On Sol login node:

```bash
cd /path/to/project
bash hpc/sol/quickstart.sh --setup-only
```

If `hpc/sol/sol.env` does not exist, quickstart auto-creates it from `sol.env.example`.

3. Edit `hpc/sol/sol.env` for your account/resources:
- `POISONEDRAG_ACCOUNT` (optional, from `myaccounts`)
- `POISONEDRAG_PARTITION`, `POISONEDRAG_QOS`, `POISONEDRAG_TIME`
- `POISONEDRAG_MODEL_NAME` and `POISONEDRAG_MODEL_CONFIG_PATH`

4. If using `palm2`/Gemini API, export key (recommended):

```bash
export GENAI_API_KEY="your_real_key"
```

## Run (Setup + Submit)

```bash
bash hpc/sol/quickstart.sh
```

This command:
- creates/updates env at `POISONEDRAG_ENV_PATH` (default `.envs/poisonedrag-sol`)
- submits Slurm job

## Submit Again (No Setup)

```bash
bash hpc/sol/quickstart.sh --submit-only
```

## Monitor

```bash
myjobs
thisjob <jobid>
seff <jobid>
```

Logs:
- `logs/slurm/<jobname>-<jobid>.out`
- `logs/slurm/<jobname>-<jobid>.err`

Results:
- summary: `results/asr_evaluation/asr_summary.json` (or `POISONEDRAG_OUTPUT_DIR`)
- detailed runs: `results/asr_evaluation/runs/`
- per-query outputs: `results/query_results/<query_results_dir>/`

## Useful Modes

### Debug Mode (fast validation)

```bash
POISONEDRAG_NUM_QUESTIONS=1 \
POISONEDRAG_NUM_REPEATS=1 \
bash hpc/sol/submit.sh --qos debug --time 00:15:00
```

### Local GPU Model Instead of API

In `hpc/sol/sol.env`:
- set `POISONEDRAG_MODEL_NAME` to `vicuna7b` or `llama7b`
- set `POISONEDRAG_MODEL_CONFIG_PATH` accordingly

For low VRAM on Vicuna, set in `model_configs/vicuna7b_config.json`:
- `"load_8bit": "True"`

## Notes for Sol

- Scripts use `module load mamba/latest` and `source activate`, matching Sol Python guidance.
- Cache paths are redirected to `${SCRATCH}` (or fallback path) to avoid home quota pressure.
- `submit.sh` overrides the default SBATCH headers in `hpc/sol/slurm/asr.sbatch`.

## References

- ASU Sol Requesting Resources: https://docs.rc.asu.edu/requesting-resources/
- ASU Sol Partitions and QoS: https://docs.rc.asu.edu/partitions-and-qos/
- ASU Slurm SBATCH Scripts: https://docs.rc.asu.edu/slurm-sbatch/
- ASU Python Envs and Mamba: https://docs.rc.asu.edu/mamba/
