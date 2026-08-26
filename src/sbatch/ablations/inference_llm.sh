#!/bin/bash
#SBATCH -c 32                   # 32 CPUs
#SBATCH --mem=64g               # 64 GB RAM
#SBATCH --gres=gpu:rtxa5000:1   # 1 GPU (A5000)
#SBATCH --time=2-00:00:00       # 3 days
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=huge-long
#SBATCH --output=logs/sbatch_eval_llm_%j.out
#SBATCH --error=logs/sbatch_eval_llm_%j.err

# Modify inference script via configs/task1_model_inference.yaml
export HOME=/fs/nexus-projects/health_sim_ai
cd /fs/nexus-projects/health_sim_ai
source venvs/llm/bin/activate
cd src
python inference_pred_llm.py
# python inference_rec_llm.py