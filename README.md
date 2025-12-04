# Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate questions from annotations
python main.py annotations.json -n 100

# Evaluate a model
python -m prompt_generator.evaluation.run_evaluation \
    annotations.json videos/ -n 50 -o ./results
```

## Project Structure

```
prompt_generator/
├── generator.py              # Question generation
├── templates.py              # 18 question types (10 standard + 8 compound)
├── answer_bank.py            # Distractor pool builder
├── cli.py                    # CLI interface
└── evaluation/
    ├── model_loader.py       # Ovis model wrapper
    ├── video_processor.py    # Frame extraction
    ├── evaluator.py          # Evaluation pipeline
    └── run_evaluation.py     # Evaluation CLI
```

## Question Types

**Standard Questions (10)**
- Primary action, aggressor ID, victim recognition
- Bystander detection, scene location
- Interaction summary, event sequence
- Social appropriateness, role counts

**Compound Questions (8)**
- Aggressor + location
- Action + victim count, action + location
- Aggressor + victim, bystander + location
- Count combinations, action + roles

## SLURM Usage

```bash
# Edit run_eval.sbatch with your settings
sbatch run_eval.sbatch

# Monitor job
./monitor_job.sh <job_id>
```

## Annotation Format

```json
{
  "video_name": "video001.mp4",
  "action": "push",
  "aggressor": "person in blue",
  "victim": "person in red",
  "environment": "parking lot",
  "bystander": "person in green"
}
```

## Commands

```bash
# Generate all questions for dataset
python main.py annotations.json --all-questions

# Evaluate with specific model
python -m prompt_generator.evaluation.run_evaluation \
    annotations.json videos/ \
    -m "AIDC-AI/Ovis2.5-9B" \
    -f 8 \
    --all-questions

# Check setup
python check-setup.py
```

## Output

Questions and results saved as timestamped JSON files with accuracy metrics by question type.