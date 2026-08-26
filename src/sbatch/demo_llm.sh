#!/bin/bash
#SBATCH -c 32                   # 32 CPUs
#SBATCH --mem=64g               # 64 GB RAM
#SBATCH --gres=gpu:rtxa6000:1   # 1 GPU (A6000)
#SBATCH --time=5-00:00:00       # 8 days
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=huge-long

export HOME=/fs/nexus-projects/health_sim_ai
cd /fs/nexus-projects/health_sim_ai
source venvs/llm/bin/activate
cd src
python demo_task1.py
# python inference_pred_llm.py
# python inference_rec_llm.py