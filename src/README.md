This `src/` directory contains:
1) `./data/`: the raw dataset; processed SFT dataset splits are also saved here.
2) `./configs/`: configs for training and inference process.
3) `./sbatch/`: bash scripts for training, inference, Gradio demo launch, ablation scripts, and logs.
4) Implementation of training, inference, and simple Gradio app interface.
5) `./eval/`: save each model's attribute-aware Likert-scale responses to visual health posters for all samples in the dataset.

## Core implementations

| File | Purpose |
| --- | --- |
| `create_dataset_task1.py` | Reads the two data CSVs and creates SFT data splits for all communication strategy held-outs. |
| `train_pred_llm.py` | SFT a vision-language model with Unsloth and TRL. |
| `inference_pred_llm.py` | Loads YAML configuration and writes model responses to a configured CSV. |
| `evaluate_pred_llm.py` | Evaluation-related module; no executable entry point is documented in this copy. |
| `demo_task1.py` | Loads the standard config and constructs a Gradio demo. |
| `demo_task1_sph.py` | Loads the School of Public Health's requested config version and constructs a Gradio demo. |
| `utils.py` | Shared image, YAML, conversation, dataset, and trait-processing helpers. |

## Helpers

`utils.py` provides these helpers:

| Function | Purpose |
| --- | --- |
| `convert_to_base64(pil_image)` | Returns a PNG image encoded as a Base64 string. |
| `load_config(filepath)` | Loads YAML and returns the parsed configuration, or `None` on a missing/invalid file. |
| `convert_to_conversation(sample, use_image=True)` | Converts one sample to a `{"messages": ...}` conversation. |
| `convert_to_model_trainer_dataset(dataframe, use_image=True)` | Converts dataframe rows to trainer conversations. |
| `process_trait_info(demo_info, persona_score, locus, include_demo=True, demo_full=True, include_big5=True, include_facet=True, include_locus=True, train_mode=True)` | Filters demographic, personality, and locus information; returns the three processed values. |

+ `create_dataset_task1.py` exposes `create_instruction(...)`, `create_answers(sample)`, and `parse_args()`.

+ `train_pred_llm.py` accepts `--model`, `--visual_stimuli`, `--n_epochs`, `--test_style`, `--partial_traits`, `--sph_traits`, `--ablation`, and `--resume_from_checkpoint`. See the root README for supported values.

## Usage

```bash
# 1) create (train/test) splits for all held-out strategies
python create_dataset_task1.py --include_demo --demo_full --include_big5 --include_facet --include_locus
# 2) train selected VLM architecture with selected held-out communication strategy
python train_pred_llm.py --model gemma --visual_stimuli True --n_epochs 1 --test_style neutral
# 3) run inference; update model variants in `configs/task1_model_inference.yaml`
python inference_pred_llm.py
```

Inference and demos use YAML files in `configs/`. Update model, data, and export paths before running them.