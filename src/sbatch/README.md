# SLURM Scripts

These shell scripts are cluster launch examples for the VLM-enabled Attribute-Aware Response Prediction pipeline. They request GPUs and other resources, then activate an environment and run a Python script from `src/`.

| Script | Action |
| --- | --- |
| `train_llm.sh` | VLM training with heldout "neutral" test style |
| `train_llm_copy1.sh` | VLM training with heldout "efficacy" test style |
| `train_llm_copy2.sh` | VLM training with heldout "threatening" test style |
| `inference_llm.sh` | VLM inference and response logging |
| `demo_llm.sh` | Gradio app demo launch |
| `ablations/ablations_create_data.sh` | Dataset-generation commands for trait ablations |
| `ablations/train_llm_ablation_1.sh`-`3.sh` | Ablation training examples |
| `ablations/inference_llm.sh` | Ablation inference example |

### Guidelines
1) These scripts contain hard-coded `/fs/nexus-projects/health_sim_ai` paths, environment location `venvs/llm`, and SLURM settings. 
2) Update the variables before launching a job on your machine. 
3) Output and error destinations under `logs/` are generated artifacts and are intentionally not part of the source instructions.
4) The ablation training scripts pass `--ablation` values such as `noBig5_noFacet_noLocus`; see the root README for all accepted training arguments.