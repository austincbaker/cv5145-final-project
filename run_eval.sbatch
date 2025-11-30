#!/bin/bash
#SBATCH --job-name=bullying_model_eval_baker
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=gpu80
#SBATCH --mem=64G
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err
# module load anaconda/anaconda-2024.10
# conda activate ovis2_5
# module load cuda/cuda-12.4.0
pip install -r requirements.txt
python -m prompt_generator.evaluation.run_evaluation correct_annotations.json videos/ --all-questions -o ./results