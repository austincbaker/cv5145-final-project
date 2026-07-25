#!/bin/bash
# =============================================================================
# austin_smoke_test.sh
# =============================================================================
# Runs `austin_smoke_test.py` for one or more models under `srun`, handling
# per-model dependency install (torch / transformers / extras) before the
# Python invocation. Writes per-model JSON results under
#   ./$USER/smoke_test_results/<SHORT_NAME>.json
# so the caller (me) can pull them via sshpass and inspect.
#
# Usage:
#   ./austin_smoke_test.sh --only "InternVideo2_5_Chat_8B,Ovis2.5-9B-Thinking"
#   ./austin_smoke_test.sh --only InternVL2.5-78B-AWQ
#   ./austin_smoke_test.sh --all            # all 5 non-from-source models
#
# Options:
#   --only LIST       comma-separated model short names (subset of the table).
#   --all             shortcut for "run every non-from-source row".
#   --timeout SEC     per-model srun timeout (default 1800 = 30 min).
#                     Weight download is the long tail; first-time loads can
#                     easily take 15 min.
#   --results-dir D   output dir (default ./$USER/smoke_test_results).
#   --dry-run         print the srun commands without executing.
#   -h | --help
# =============================================================================

set -euo pipefail

# Same table shape as austin_tmp_script.sh, but only the short columns we
# actually need here. From-source models (Qwen3-VL-*, Gemma-4) and GLM are
# deliberately excluded — they need a transformers install that can't be
# automated safely inside an srun one-shot.
#
# Columns: SHORT_NAME|HF_PATH|TRANSFORMERS|TORCH|TORCH_INDEX|TIER|EXTRA_PIPS
#
# TIER is resolved at submit time:
#   T1 -> --constraint="gmem48|gmem80"  --mem=48G   --cpus-per-task=8
#   T2 -> --constraint="gmem80"         --mem=128G  --cpus-per-task=16
# The constraint feature uses SLURM's OR operator `|` (pipe). This cluster
# tags nodes with `gmem48`, `gmem80`, etc. as features; a constraint of
# "gmem48|gmem80" matches a node with either tag.
MODELS=(
    "InternVL3-9B|OpenGVLab/InternVL3-9B|4.37.2|||T1|"
    "InternVideo2_5_Chat_8B|OpenGVLab/InternVideo2_5_Chat_8B|4.40.1|2.5.1||T1|"
    "Ovis2.5-9B-Thinking|AIDC-AI/Ovis2.5-9B|4.51.3|2.5.1||T1|"
    "Qwen2-VL-72B-Instruct-AWQ|Qwen/Qwen2-VL-72B-Instruct-AWQ|4.38.2|2.5.1|https://download.pytorch.org/whl/cu118|T2|autoawq"
    "InternVL2.5-78B-AWQ|OpenGVLab/InternVL2_5-78B-AWQ|4.49.0|||T2|autoawq"
)

# Tier-to-resource mapping
tier_constraint() { case "$1" in T1) echo "gmem48|gmem80" ;; T2) echo "gmem80" ;; esac; }
tier_mem()        { case "$1" in T1) echo "48G" ;; T2) echo "128G" ;; esac; }
tier_cpus()       { case "$1" in T1) echo "8" ;; T2) echo "16" ;; esac; }

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
ONLY_LIST=""
RUN_ALL=0
DRY_RUN=0
TIMEOUT_SEC=1800
RESULTS_DIR="./${USER}/smoke_test_results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)          ONLY_LIST="$2"; shift 2 ;;
        --all)           RUN_ALL=1; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --timeout)       TIMEOUT_SEC="$2"; shift 2 ;;
        --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
        -h|--help)       sed -n '2,/^# ==== *$/p' "$0" | sed 's/^# //; s/^#$//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "$RUN_ALL" == "0" && -z "$ONLY_LIST" ]]; then
    echo "ERROR: pass --all or --only LIST" >&2
    exit 1
fi

# Filter MODELS by --only (if set).
SELECTED=()
if [[ "$RUN_ALL" == "1" ]]; then
    SELECTED=("${MODELS[@]}")
else
    IFS=',' read -ra ONLY_ARR <<< "$ONLY_LIST"
    for row in "${MODELS[@]}"; do
        short="${row%%|*}"
        for name in "${ONLY_ARR[@]}"; do
            name="${name#"${name%%[![:space:]]*}"}"; name="${name%"${name##*[![:space:]]}"}"
            if [[ "$short" == "$name" ]]; then SELECTED+=("$row"); break; fi
        done
    done
fi

if [[ ${#SELECTED[@]} -eq 0 ]]; then
    echo "ERROR: no models matched --only. Available:" >&2
    for row in "${MODELS[@]}"; do echo "  ${row%%|*}" >&2; done
    exit 1
fi

mkdir -p "$RESULTS_DIR"
mkdir -p "./${USER}"   # srun --output parent must exist

echo "====================================================="
echo "austin_smoke_test.sh"
echo "====================================================="
echo "Models to test (${#SELECTED[@]}):"
for row in "${SELECTED[@]}"; do echo "  ${row%%|*}"; done
echo "Timeout per model: ${TIMEOUT_SEC}s"
echo "Results dir: $RESULTS_DIR"
[[ "$DRY_RUN" == "1" ]] && echo "DRY-RUN MODE"
echo

# -----------------------------------------------------------------------------
# Per-model: assemble an inline script, run it under srun.
# -----------------------------------------------------------------------------
run_one() {
    local row="$1"
    IFS='|' read -r short hf tf torch idx tier extra_pips <<< "$row"

    local constraint mem cpus
    constraint="$(tier_constraint "$tier")"
    mem="$(tier_mem "$tier")"
    cpus="$(tier_cpus "$tier")"

    local result_json="${RESULTS_DIR}/${short}.json"
    local log_file="${RESULTS_DIR}/${short}.log"

    # Assemble the inline payload run on the compute node. Deliberately uses
    # the same env-setup pattern as all_model_multi_gpu.sbatch so problems
    # surface in both paths. All log lines go to stderr of srun so we see
    # them in real time.
    local payload
    payload="$(cat <<EOF
set -e
echo "=== [$short] on \$(hostname) at \$(date) ==="

# Conda / modules
eval "\$('/home/au182598/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)" \\
    || source /home/au182598/miniconda3/etc/profile.d/conda.sh
module use ~/privatemodules 2>/dev/null || true
module load anaconda/25.11.1 2>/dev/null || true
conda activate base
module load cuda/12.6 2>/dev/null || true

# Python 3.13 compatibility: older transformers pins pull tokenizers
# versions whose pyo3 doesn't officially support 3.13, which triggers a
# failed source build. Setting this env var lets pyo3 fall back to the
# stable ABI and build anyway (the error message explicitly suggests it).
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

# Optional torch override
if [[ -n "${torch}" ]]; then
    CURRENT="\$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo MISSING)"
    if [[ "\$CURRENT" != "${torch}"* ]]; then
        echo "=== installing torch==${torch} ==="
        args=(torch=="${torch}")
        [[ -n "${idx}" ]] && args+=(--index-url "${idx}")
        pip install "\${args[@]}" || { echo "FATAL: torch install failed"; exit 10; }
    else
        echo "=== torch \$CURRENT already matches ${torch} ==="
    fi
fi

# Transformers override
if [[ -n "${tf}" ]]; then
    CURRENT="\$(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo MISSING)"
    if [[ "\$CURRENT" != "${tf}" ]]; then
        echo "=== installing transformers==${tf} ==="
        pip install "transformers==${tf}" || { echo "FATAL: transformers install failed"; exit 11; }
    else
        echo "=== transformers \$CURRENT already matches ${tf} ==="
    fi
fi

# Extra pips
if [[ -n "${extra_pips}" ]]; then
    echo "=== installing extras: ${extra_pips} ==="
    pip install ${extra_pips} || { echo "FATAL: extras install failed"; exit 12; }
fi

echo "=== running python smoke test ==="
python austin_smoke_test.py --model "${hf}" --output "${result_json}"
rc=\$?
echo "=== [$short] smoke test exit code: \$rc ==="
exit \$rc
EOF
)"

    # SLURM --constraint uses `|` as OR between features, `&` for AND.
    # Quote the whole value so bash doesn't interpret the pipe.
    local srun_cmd=(
        srun
        -p gpu
        --gres=gpu:1
        --cpus-per-task="$cpus"
        --mem="$mem"
        --constraint="$constraint"
        --time="$(( TIMEOUT_SEC / 60 )):00"
        --job-name="smoke-${short}"
        --output="${log_file}"
        bash -c "$payload"
    )

    echo "--- [$short] ---"
    echo "  hf: $hf (tier=$tier)"
    echo "  constraint: '$constraint'   mem: $mem   cpus: $cpus   timeout: ${TIMEOUT_SEC}s"
    echo "  result: $result_json"
    echo "  log:    $log_file"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  (dry-run; skipping)"
        return 0
    fi

    # Kick it off. srun blocks until the job ends or times out.
    if "${srun_cmd[@]}"; then
        echo "  -> srun exited 0 (check result JSON for stage status)"
    else
        echo "  -> srun exited non-zero (see log)" >&2
    fi
    echo
}

# -----------------------------------------------------------------------------
# Drive the loop sequentially. Parallel srun would hit your 4-GPU QOS cap and
# queue; sequential is simpler to diagnose.
# -----------------------------------------------------------------------------
for row in "${SELECTED[@]}"; do
    run_one "$row"
done

echo "====================================================="
echo "All models processed. Result JSONs in: $RESULTS_DIR"
echo "====================================================="
echo "Summary:"
for row in "${SELECTED[@]}"; do
    short="${row%%|*}"
    f="${RESULTS_DIR}/${short}.json"
    if [[ -f "$f" ]]; then
        status="$(python -c "import json; d=json.load(open('$f')); print(d.get('status','?'),d.get('stage','?'))" 2>/dev/null || echo "unparseable")"
        echo "  $short  --  $status"
    else
        echo "  $short  --  (no result file)"
    fi
done
