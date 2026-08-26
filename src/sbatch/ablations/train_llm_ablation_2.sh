#!/bin/bash
#SBATCH -c 32                   # 32 CPUs
#SBATCH --mem=128g              # 128 GB RAM
#SBATCH --gres=gpu:rtxa5000:1   # 1 GPU (A5000)
#SBATCH --time=2-00:00:00       # 3 days
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=huge-long
#SBATCH --output=logs/sbatch_train_llm_%j.out
#SBATCH --error=logs/sbatch_train_llm_%j.err

export HOME=/fs/nexus-projects/health_sim_ai
export TMPDIR=$HOME/tmp
cd /fs/nexus-projects/health_sim_ai
source venvs/llm/bin/activate
cd src
# llama, gemma, qwen
python train_pred_llm.py \
    --model gemma \
    --visual_stimuli True \
    --n_epochs 1 \
    --test_style threatening \
    --ablation noBig5_noFacet_noLocus # noDemo, noBig5, noFacet, noLocus, noBig5_noFacet_noLocus