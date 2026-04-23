#!/bin/bash
# Usage:
#   ./eval_model.sh --file generated_questions.json --model qwen-72b
#   ./eval_model.sh --model qwen-72b          (on-the-fly question generation)
#   ./eval_model.sh some_script.sbatch        (legacy mode)

SBATCH_SCRIPT="all_model_multi_gpu.sbatch"
QUESTIONS_FILE=""
MODEL=""

# Legacy mode: first arg is not a flag, treat as sbatch script path
if [[ $# -ge 1 && "$1" != --* ]]; then
    dos2unix "$1" && sbatch "$1" && sleep 2 && ./monitor_job.sh $(squeue --me -h -o %i | head -n 1)
    exit $?
fi

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --file)
            QUESTIONS_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --file <questions.json> --model <model-shortcut> --gpus <number of GPUs>"
            echo "       $0 <sbatch-script>  (legacy mode)"
            exit 1
            ;;
    esac
done

# Build --export string for sbatch
EXPORT_PARTS=()
if [[ -n "$MODEL" ]]; then
    EXPORT_PARTS+=("MODEL=$MODEL")
fi
if [[ -n "$QUESTIONS_FILE" ]]; then
    EXPORT_PARTS+=("QUESTIONS_FILE=$QUESTIONS_FILE")
fi
if [[ -n "$NUM_GPUS" ]]; then
    EXPORT_PARTS+=("NUM_GPUS=$NUM_GPUS")
fi

EXPORT_ARGS=""
if [[ ${#EXPORT_PARTS[@]} -gt 0 ]]; then
    EXPORT_ARGS="--export=$(IFS=,; echo "${EXPORT_PARTS[*]}")"
fi


# echo "Checking if results file already exists..."
# if [[ -d results_$MODEL ]]; then 
#     echo "Directory already exists, moving results_$MODEL to results_$(date +%s)_$MODEL"
#     mv results_$MODEL results_$(date +%s)_$MODEL
# fi

dos2unix "$SBATCH_SCRIPT" 2>/dev/null
sbatch $EXPORT_ARGS "$SBATCH_SCRIPT" && \
    sleep 2 && \
    ./monitor_job.sh $(squeue --me -h -o %i | head -n 1)
