  ┌─────┬───────────┬────────┬───────────────────────┬───────────────────────┐                       
  │  #  │ Aggressor │ Victim │        Action         │         Notes         │                       
  ├─────┼───────────┼────────┼───────────────────────┼───────────────────────┤                       
  │ 1   │ A         │ B      │ aggressive talking    │ role-reversal of GT   │                       
  ├─────┼───────────┼────────┼───────────────────────┼───────────────────────┤
  │ 2   │ A         │ A      │ choke                 │ wrong_action on (A,B) │                       
  ├─────┼───────────┼────────┼───────────────────────┼───────────────────────┤                       
  │ 3   │ B         │ A      │ hair grabbing         │ self-directed         │                       
  ├─────┼───────────┼────────┼───────────────────────┼───────────────────────┤                       
  │ 4   │ B         │ A      │ aggressive talking    │ ✓ GT                  │
  ├─────┼───────────┼────────┼───────────────────────┼───────────────────────┤                       
  │ 5   │ A         │ B      │ choke                 │ wrong_action on (A,B) │
  ├─────┼───────────┼────────┼───────────────────────┼───────────────────────┤                       
  │ 6   │ B         │ A      │ choke                 │ self-directed         │
  ├─────┼───────────┼────────┼───────────────────────┼───────────────────────┤
  │ 7   │ A         │ B      │ hair grabbing         │ wrong_action on (A,B) │
  ├─────┼───────────┼────────┼───────────────────────┼───────────────────────┤                       
  │ 8   │ B         │ B      │ hair grabbing         │ self-directed         │
  └─────┴───────────┴────────┴───────────────────────┴───────────────────────┘      


python generate_questions_local.py annotations.json --hardness-profile frequency_inverted -o generated_questions_freq_inverted.json && \
python make_frameless_questions.py generated_questions_freq_inverted.json && \
sbatch --export=ALL,MODEL=OpenGVLab/InternVL3_5-8B,QUESTIONS_FILE=generated_questions_freq_inverted_text_only.json text_only_eval.sbatch


python generate_questions_local.py annotations.json  --hardness-profile frequency_inverted --sample 100 --seed 42 -o generated_questions_freq_inv_sample_70pct.json && \
python make_frameless_questions.py generated_questions_freq_inv_sample_70pct.json -o generated_questions_freq_inv_sample_70pct.json && \
sbatch --export=ALL,MODEL=OpenGVLab/InternVL3_5-8B,QUESTIONS_FILE=generated_questions.json,TORCH=<TORCH_VERSION>,TRANSFORMERS=<TRANSFORMERS_VERSION> text_only_eval.sbatch
sbatch --export=ALL,MODEL=$MODEL,QUESTIONS_FILE=generated_questions_freq_inv_part${i}of3.json,OUTPUT_DIR=./results_freq_inv_part${i}_Qwen2_5_VL_72B_final,,TORCH=<TORCH_VERSION>,TRANSFORMERS=<TRANSFORMERS_VERSION> all_model_multi_gpu.sbatch

MODEL=OpenGVLab/InternVL3_5-8B
for i in 1 2 3; do                                          
    sbatch --export=ALL,\
  MODEL=$MODEL,\
  QUESTIONS_FILE=generated_questions_freq_inv_part${i}of3.json,\
  OUTPUT_DIR=./results_freq_inv_part${i} \
      all_model_multi_gpu.sbatch                                
  done    
python generate_questions_local.py annotations.json -hardness-profile frequency_inverted --split 3 -o generated_questions_freq_inv.json

results_freq_inv_part2_InternVL3_5_8b_final
python combine_eval_results.py results_freq_inv_part1_InternVL3_5_8b_final/evaluation_*.json results_freq_inv_part2_InternVL3_5_8b_final/evaluation_*.json results_freq_inv_part3_InternVL3_5_8b_final/evaluation_*.json -o results_combined_freq_inv.json

InternVL3-9B
Qwen3-VL-8B-Instruct
InternVideo2_5_Chat_8B
Ovis2.5-9B-Thinking
Qwen3-VL-8B-Thinking
gemma-4-26B-A4B-it
GLM-4.6V-Flash
Qwen2-VL-72B-Instruct-AWQ
InternVL2.5-78B-AWQ

{
  "group_1": ["video1", "video_2"]
}

1. just get it working for the first 2-3 models and document what remains               
2. skip those and only attempt the 5 whose dependencies we have version pins for                                                                                                                                                           
3. dump full tracebacks to files                                                                                                   
4. Can I allocate srun resources aggressively: sure                                               

srun -p gpu --gres=gpu:1 --cpus-per-task=6 --mem-per-cpu=8G -C gmem80 --pty bash

srun -p gpu --qos=group3i-1  --gres=gpu:1  --constraint="gmem48|gmem80"  --cpus-per-task=8  --mem=48G  --pty bash

eval "$('/home/au182598/miniconda3/bin/conda' 'shell.bash' 'hook')"                          
  module use ~/privatemodules 2>/dev/null                                                      
  module load anaconda/25.11.1 2>/dev/null                                                     
  conda activate base                                                                          
  module load cuda/12.6 2>/dev/null

Counting
Social Appropriateness
Model cheating by Frequency