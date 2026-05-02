#!/bin/bash
#SBATCH --job-name=social_grade
#SBATCH --output=freeform_eval/social_results/grade_%j.out
#SBATCH --error=freeform_eval/social_results/grade_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

echo "=============================================="
echo "Social Appropriateness Grading with gpt-oss-20b"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================================="

. ~/miniconda3/etc/profile.d/conda.sh

# Create or activate gpt-oss environment
if conda env list | grep -q "gptoss"; then
    echo "Using existing gptoss environment"
    conda activate gptoss
else
    echo "Creating gptoss environment..."
    conda create -n gptoss python=3.12 -y
    conda activate gptoss
    pip install openai vllm --extra-index-url https://download.pytorch.org/whl/cu124
fi

cd ~/aggressive_behavior_project
export PYTHONPATH="$PWD:$PYTHONPATH"

# Start vLLM server in background
echo "Starting vLLM server with gpt-oss-20b..."
vllm serve openai/gpt-oss-20b --port 8000 --max-model-len 4096 &
VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM server..."
for attempt in $(seq 1 60); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "vLLM server ready (attempt $attempt)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM server died during startup"
        exit 1
    fi
    sleep 5
done

if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "vLLM server failed to start after 5 minutes"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Grade each model's responses
MODELS=("InternVL2.5-78B-AWQ" "InternVL2.5-8B" "Qwen3-VL-8B" "Ovis2.5-9B-Thinking" "Ovis2.5-9B")

for MODEL in "${MODELS[@]}"; do
    RESP="freeform_eval/social_results/social_responses_${MODEL}.json"
    OUT="freeform_eval/social_results/social_grades_${MODEL}.json"

    if [ ! -f "$RESP" ]; then
        echo "SKIP $MODEL: $RESP not found"
        continue
    fi

    echo ""
    echo "=== Grading $MODEL ==="
    python freeform_eval/grade_social_appropriateness.py \
        --responses "$RESP" \
        --judge vllm \
        --judge-url http://localhost:8000/v1 \
        --judge-model openai/gpt-oss-20b \
        -o "$OUT"
done

# Stop vLLM server
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo ""
echo "=== Summary ==="
for MODEL in "${MODELS[@]}"; do
    OUT="freeform_eval/social_results/social_grades_${MODEL}.json"
    if [ -f "$OUT" ]; then
        python3 -c "
import json
with open('$OUT') as f:
    d = json.load(f)
a = d['aggregate']
print(f'  ${MODEL}: {a[\"overall_accuracy\"]:.1f}% overall, {a[\"aggressive_accuracy\"]:.1f}% aggressive, {a[\"non_aggressive_accuracy\"]:.1f}% non-aggressive')
"
    fi
done

echo ""
echo "Grading complete: $(date)"
