#!/bin/bash
#SBATCH --job-name=bullying_model_eval_baker
#SBATCH --time=4:00:00   
#SBATCH --nodelist=evc104             
#SBATCH --partition=highgpu         
#SBATCH --gres=gpu:1                  
#SBATCH --cpus-per-task=6             
#SBATCH --mem=64G                     
#SBATCH --constraint=h100             
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err

# Run your Python script
python -m prompt_generator.evaluation.run_evaluation correct_annotations.json videos/ --all-questions -o ./results

