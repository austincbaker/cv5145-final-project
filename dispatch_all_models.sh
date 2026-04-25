#!/bin/bash
# =============================================================================
# dispatch_all_models.sh
# =============================================================================
# Submits all_model_multi_gpu.sbatch for each (model, question_file) pair with
# the right TORCH / TRANSFORMERS / TORCH_INDEX overrides per model.
#
# Usage:
#   ./dispatch_all_models.sh --questions GLOB [options]
#
# Required:
#   --questions GLOB       one or more question JSON files. Globs expand, so
#                          "generated_questions_freq_inv_part*of3.json" works
#                          for a split set.
#
# Options:
#   --only LIST            comma-separated subset of model short names to submit
#                          (see the table below). Default: all of them.
#   --dry-run              print the sbatch commands but don't submit.
#   --stagger-seconds N    pause between submits so the cluster doesn't see
#                          a sudden burst. Default 300 (5 minutes).
#   --allow-from-source    also submit the three models whose transformers
#                          must be installed from source (Qwen3-VL-8B-Instruct,
#                          Qwen3-VL-8B-Thinking, gemma-4-26B-A4B-it). You MUST
#                          first run the one-off setup command printed at the
#                          top of this script's output inside the target conda
#                          env. Without this flag those three are skipped.
#   --sbatch PATH          sbatch script to submit (default:
#                          all_model_multi_gpu.sbatch).
#   -h | --help            show this help.
#
# OUTPUT_DIR naming:
#   results_<question_label>_<model_short_name>
# where <question_label> is the question filename minus the .json extension
# (and the leading generated_questions_ prefix if present). The sbatch wraps
# this under ./$USER/ automatically.
#
# Models:
#   1. InternVL3-9B              OpenGVLab/InternVL3-9B
#   2. Qwen3-VL-8B-Instruct      Qwen/Qwen3-VL-8B-Instruct           [from-source]
#   3. InternVideo2_5_Chat_8B    OpenGVLab/InternVideo2_5_Chat_8B
#   4. Ovis2.5-9B-Thinking       AIDC-AI/Ovis2.5-9B                  (enable_thinking=True)
#   5. Qwen3-VL-8B-Thinking      Qwen/Qwen3-VL-8B-Thinking           [from-source]
#   6. gemma-4-26B-A4B-it        google/gemma-4-26B-A4B-it           [from-source]
#   7. Qwen2-VL-72B-Instruct-AWQ Qwen/Qwen2-VL-72B-Instruct-AWQ
#   8. InternVL2.5-78B-AWQ       OpenGVLab/InternVL2_5-78B-AWQ
#
# Deliberately excluded: GLM-4.6V-Flash. It requires transformers>=5.0.0rc0
# which will break every other model's conda env. Run that one by hand in a
# dedicated env.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Model table. Columns:
#   SHORT_NAME|HF_PATH|TRANSFORMERS|TORCH|TORCH_INDEX|FROM_SOURCE|GRES|MEM|CPUS|EXTRA_PIPS|CONDA_ENV|DEVICE_MAP
#
# TRANSFORMERS / TORCH / TORCH_INDEX empty string = don't export that var
# (sbatch falls back to its own model-name heuristic or leaves torch alone).
# FROM_SOURCE=1 means the user must install transformers from git ahead of time;
# gated by --allow-from-source.
# GRES / MEM / CPUS are passed to sbatch as --gres / --mem / --cpus-per-task
# overrides (sbatch-command-line overrides win over #SBATCH directives).
# EXTRA_PIPS is a space-separated list of additional pip packages to install
# before running (e.g. `autoawq` for AWQ-quantized models — without it the
# transformers loader silently drops AWQ weights and crashes at first
# inference with "NotImplementedError: Cannot copy out of meta tensor").
#
# Torch pins:
#   Cluster runs Python 3.13, which has no torch wheels before 2.5.0. The
#   original pins from model docs (2.2.1 / 2.4.0) fail to resolve on this
#   Python. Bumped to 2.5.1 for all models that need a torch override; the
#   API differences between 2.4 and 2.5 don't affect inference for these
#   models.
#
# Tiering (UCF CRCV Ampere classes, resolved via VRAM gres tag):
#   T1 (9 B bf16 models):  gpu:ampere:1,gpumem:48G  /  48G RAM  /  8 CPUs
#     - InternVL3-9B, Qwen3-VL-8B-*, InternVideo2_5_Chat_8B, Ovis2.5-9B
#     - 24 GB Ampere nodes are excluded; 48 GB and 80 GB both qualify.
#   T2 (70-80 B AWQ, 26 B MoE):  gpu:ampere:1,gpumem:80G  /  128G RAM  /  16 CPUs
#     - Qwen2-VL-72B-AWQ, InternVL2.5-78B-AWQ, gemma-4-26B-A4B-it
#     - AWQ loading spikes past 80 GB system RAM; gemma MoE must hold all
#       25.2 B experts resident despite 3.8 B active/step.
# -----------------------------------------------------------------------------
# Pins updated from smoke-test findings on 2026-04-24 (Python 3.12 vlm_py312 env):
#   - T1 models (9B bf16) all work at transformers 4.51.3; no torch pin needed
#     since the env ships 2.5.1 by default.
#   - Qwen2-VL-72B-AWQ: class exists in transformers >= 4.45; preprocessor
#     schema changed >=4.48 so 4.45.2 is the sweet spot. autoawq 0.2.7.post3
#     avoids the qwen3 import requirement of newer autoawq versions.
#   - InternVL2.5-78B-AWQ: pinned at 4.51.3 (consistent with T1). autoawq
#     same pin as Qwen2-VL to avoid qwen3 module dependency.
#
# CONDA_ENV column:
#   Transformers versions are pinned per-env to avoid pip-install races when
#   jobs with different transformers pins run concurrently on NFS-shared
#   conda envs. Model-to-env mapping:
#     vlm_py312         -- default; transformers 4.51.3 baseline for T1 + InternVL family
#     vlm_py312_tf4451  -- transformers 4.45.2 + autoawq 0.2.7.post3 for Qwen2-VL-AWQ
#   Create envs once on the login node before first submit:
#     conda create -n vlm_py312_tf4451 --clone vlm_py312 -y
#     conda activate vlm_py312_tf4451 && pip install transformers==4.45.2 autoawq==0.2.7.post3
#     conda activate vlm_py312 && pip install transformers==4.51.3
MODELS=(
    "InternVL3-9B|OpenGVLab/InternVL3-9B|4.51.3|||0|gpu:ampere:1,gpumem:48G|48G|8||vlm_py312|"
    "Qwen3-VL-8B-Instruct|Qwen/Qwen3-VL-8B-Instruct||||1|gpu:ampere:1,gpumem:48G|48G|8||vlm_py312|"
    "InternVideo2_5_Chat_8B|OpenGVLab/InternVideo2_5_Chat_8B|4.51.3|||0|gpu:ampere:1,gpumem:48G|48G|8||vlm_py312|"
    "Ovis2.5-9B-Thinking|AIDC-AI/Ovis2.5-9B|4.51.3|||0|gpu:ampere:1,gpumem:48G|48G|8||vlm_py312|"
    "Qwen3-VL-8B-Thinking|Qwen/Qwen3-VL-8B-Thinking||||1|gpu:ampere:1,gpumem:48G|48G|8||vlm_py312|"
    # "gemma-4-26B-A4B-it|google/gemma-4-26B-A4B-it||||1|gpu:ampere:1,gpumem:80G|128G|16||vlm_py312|"
    "Qwen2-VL-72B-Instruct-AWQ|Qwen/Qwen2-VL-72B-Instruct-AWQ|4.45.2|||0|gpu:ampere:2,gpumem:80G|128G|16|autoawq==0.2.7.post3|vlm_py312_tf4451|auto"
    # InternVL2.5-78B-AWQ: commented out pending loader fix -- AWQ dispatch
    # isn't firing via trust_remote_code path, model loads as dense bf16 and
    # either OOMs at ~78 GB or returns gibberish. Revisit after resolving.
    # "InternVL2.5-78B-AWQ|OpenGVLab/InternVL2_5-78B-AWQ|4.51.3|||0|gpu:ampere:1,gpumem:80G|128G|16|autoawq==0.2.7.post3|vlm_py312|"
)

# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------
QUESTIONS_GLOB=""
ONLY_LIST=""
DRY_RUN=0
STAGGER_SECONDS=300
ALLOW_FROM_SOURCE=0
SBATCH_SCRIPT="all_model_multi_gpu.sbatch"

usage() {
    sed -n '2,/^# ==== *$/p' "$0" | sed 's/^# //; s/^#$//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --questions)         QUESTIONS_GLOB="$2"; shift 2 ;;
        --only)              ONLY_LIST="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=1; shift ;;
        --stagger-seconds)   STAGGER_SECONDS="$2"; shift 2 ;;
        --allow-from-source) ALLOW_FROM_SOURCE=1; shift ;;
        --sbatch)            SBATCH_SCRIPT="$2"; shift 2 ;;
        -h|--help)           usage 0 ;;
        *)                   echo "Unknown argument: $1" >&2; usage 1 ;;
    esac
done

if [[ -z "$QUESTIONS_GLOB" ]]; then
    echo "ERROR: --questions GLOB is required" >&2
    usage 1
fi

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "ERROR: sbatch script not found: $SBATCH_SCRIPT" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Expand question-file glob
# -----------------------------------------------------------------------------
# Split glob on whitespace so the caller can pass multiple patterns if they
# want, e.g. --questions "a.json b.json".
shopt -s nullglob
QUESTION_FILES=()
for pattern in $QUESTIONS_GLOB; do
    for f in $pattern; do
        [[ -f "$f" ]] && QUESTION_FILES+=("$f")
    done
done
shopt -u nullglob

if [[ ${#QUESTION_FILES[@]} -eq 0 ]]; then
    echo "ERROR: no files matched: $QUESTIONS_GLOB" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Filter models by --only
# -----------------------------------------------------------------------------
if [[ -n "$ONLY_LIST" ]]; then
    IFS=',' read -ra ONLY_ARR <<< "$ONLY_LIST"
    FILTERED=()
    for row in "${MODELS[@]}"; do
        short="${row%%|*}"
        for name in "${ONLY_ARR[@]}"; do
            # trim whitespace
            name="${name#"${name%%[![:space:]]*}"}"; name="${name%"${name##*[![:space:]]}"}"
            if [[ "$short" == "$name" ]]; then
                FILTERED+=("$row"); break
            fi
        done
    done
    if [[ ${#FILTERED[@]} -eq 0 ]]; then
        echo "ERROR: --only produced no matches. Available short names:" >&2
        for row in "${MODELS[@]}"; do echo "  ${row%%|*}" >&2; done
        exit 1
    fi
    MODELS=("${FILTERED[@]}")
fi

# -----------------------------------------------------------------------------
# Split out from-source models; handle --allow-from-source gate.
# -----------------------------------------------------------------------------
SUBMITTABLE=()
SKIPPED_FROM_SOURCE=()
for row in "${MODELS[@]}"; do
    IFS='|' read -r short hf tf torch idx fromsrc gres mem cpus extra_pips conda_env device_map <<< "$row"
    if [[ "$fromsrc" == "1" && "$ALLOW_FROM_SOURCE" != "1" ]]; then
        SKIPPED_FROM_SOURCE+=("$short")
    else
        SUBMITTABLE+=("$row")
    fi
done

# -----------------------------------------------------------------------------
# Preamble
# -----------------------------------------------------------------------------
echo "============================================================"
echo "dispatch_all_models.sh"
echo "============================================================"
echo "Question files (${#QUESTION_FILES[@]}):"
for f in "${QUESTION_FILES[@]}"; do echo "  $f"; done
echo
echo "Models to submit (${#SUBMITTABLE[@]}):"
for row in "${SUBMITTABLE[@]}"; do
    IFS='|' read -r short hf tf torch idx fromsrc gres mem cpus extra_pips conda_env device_map <<< "$row"
    tag=""
    [[ "$fromsrc" == "1" ]] && tag=" [from-source; transformers pin skipped]"
    echo "  $short -> $hf$tag"
    printf '      resources: --gres=%s --mem=%s --cpus-per-task=%s\n' "$gres" "$mem" "$cpus"
    [[ -n "$conda_env"  ]] && printf '      conda_env:  %s\n' "$conda_env"
    [[ -n "$extra_pips" ]] && printf '      extra_pips: %s\n' "$extra_pips"
done

if [[ ${#SKIPPED_FROM_SOURCE[@]} -gt 0 ]]; then
    echo
    echo "Skipped (transformers must be installed from source):"
    for s in "${SKIPPED_FROM_SOURCE[@]}"; do echo "  $s"; done
    echo
    echo "To include them, first run this inside your target conda env:"
    echo "    pip install git+https://github.com/huggingface/transformers"
    echo "then re-run this script with --allow-from-source."
fi

TOTAL_JOBS=$(( ${#SUBMITTABLE[@]} * ${#QUESTION_FILES[@]} ))
echo
echo "Total jobs that will be submitted: $TOTAL_JOBS"
echo "Stagger between submits: ${STAGGER_SECONDS}s"
[[ "$DRY_RUN" == "1" ]] && echo "DRY-RUN MODE: no jobs will be submitted."
echo "============================================================"
echo

# Warn about GLM exclusion
echo "NOTE: GLM-4.6V-Flash is deliberately excluded from this batch."
echo "      It requires transformers>=5.0.0rc0 which is incompatible with"
echo "      every other model here. Run it manually in a dedicated env."
echo

# -----------------------------------------------------------------------------
# Submit loop
# -----------------------------------------------------------------------------
question_label() {
    local path="$1"
    local base
    base="$(basename "$path" .json)"
    # Strip conventional prefix so labels stay short.
    echo "${base#generated_questions_}"
}

job_count=0
for qfile in "${QUESTION_FILES[@]}"; do
    qlabel="$(question_label "$qfile")"
    for row in "${SUBMITTABLE[@]}"; do
        IFS='|' read -r short hf tf torch idx fromsrc gres mem cpus extra_pips conda_env device_map <<< "$row"
        job_count=$(( job_count + 1 ))

        output_dir="results_${qlabel}_${short}"

        # Build the --export list. ALL first, then required vars, then
        # conditional overrides (only when non-empty).
        exports="ALL,MODEL=${hf},QUESTIONS_FILE=${qfile},OUTPUT_DIR=${output_dir}"
        [[ -n "$tf"         ]] && exports+=",TRANSFORMERS=${tf}"
        [[ -n "$torch"      ]] && exports+=",TORCH=${torch}"
        [[ -n "$idx"        ]] && exports+=",TORCH_INDEX=${idx}"
        [[ -n "$extra_pips" ]] && exports+=",EXTRA_PIPS=${extra_pips}"
        [[ -n "$conda_env"  ]] && exports+=",CONDA_ENV=${conda_env}"
        [[ -n "$device_map" ]] && exports+=",DEVICE_MAP=${device_map}"

        # Per-model resource overrides. sbatch command-line flags override
        # the #SBATCH directives baked into all_model_multi_gpu.sbatch, so
        # we steer each job to the right VRAM tier without editing the
        # sbatch file.
        cmd=(sbatch "--export=${exports}")
        [[ -n "$gres" ]] && cmd+=("--gres=${gres}")
        [[ -n "$mem"  ]] && cmd+=("--mem=${mem}")
        [[ -n "$cpus" ]] && cmd+=("--cpus-per-task=${cpus}")
        cmd+=("$SBATCH_SCRIPT")

        echo "----- [${job_count}/${TOTAL_JOBS}] ${short} on ${qlabel} -----"
        printf '  %s' "${cmd[@]}"; echo
        echo "  -> OUTPUT_DIR=./\$USER/${output_dir}"

        if [[ "$DRY_RUN" == "1" ]]; then
            echo "  (dry-run; skipping)"
        else
            if "${cmd[@]}"; then
                :
            else
                echo "  ERROR: sbatch returned non-zero. Continuing with remaining jobs." >&2
            fi
        fi

        # Stagger only between submits, not after the final one.
        if [[ "$job_count" -lt "$TOTAL_JOBS" && "$DRY_RUN" != "1" ]]; then
            echo "  ...sleeping ${STAGGER_SECONDS}s before next submit"
            sleep "$STAGGER_SECONDS"
        fi
    done
done

echo
echo "============================================================"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete. ${TOTAL_JOBS} jobs would have been submitted."
else
    echo "All ${TOTAL_JOBS} submit attempts complete. Check squeue -u \$USER."
fi
echo "Results will land under ./\$USER/results_<qlabel>_<model>/"
echo "============================================================"
