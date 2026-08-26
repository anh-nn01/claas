# 🌱 CLAAS: Conditional LVLM-enabled Attribute-Aware response Simulation for Public Health Education   

[![arXiV](https://img.shields.io/badge/arxiv-link-red)](google.com) 
[![Website](https://img.shields.io/badge/website-link-purple)](google.com) 
[![Dataset](https://img.shields.io/badge/dataset-huggingface-yellow)](google.com) 

<!-- ![Paper Picture Highlight](assets/to_be_updated.png) -->

## I. HPC allocation
```bash
# -> standard
srun --pty --mem=32gb --time=3-00:00:00 --gres=gpu:rtxa6000:1 bash
# -> more memory
srun --pty --qos=medium --mem=64gb --time=2-00:00:00 --gres=gpu:rtxa5000:1 bash
# -> most memory
srun --pty --qos=high --mem=128gb --time=1-00:00:00 --gres=gpu:rtxa5000:1 bash
# -> more cpu + more time
srun --pty --qos=huge-long --cpus-per-gpu=32 --mem=32gb --time=8-00:00:00 --gres=gpu:rtxa6000:1 bash
```

<!-- ## Ollama installation without `sudo`
0. Set `HOME` directory to project directory:
    ```bash
    pwd
    export HOME=<your_project_path>
    ```
1. Install `ollama-linux-amd64` here: https://github.com/ollama/ollama/releases <br>
    or 
    ```bash
    curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
    ```
2.  Create Ollama binary
    ```bash
    mkdir ollama
    tar -C ./ollama -xzf ollama-linux-amd64.tgz
    ```
## Ollama deployment
1. ***In another terminal (with `NODE_NAME`):*** Start Ollama 
    ```bash
    ./ollama/bin/ollama serve
    ```
2. Pull **LLaMA-3.2-Vision 11b**: (6gb)
    ```bash
    ./ollama/bin/ollama pull llama3.2-vision
    ```
    Pull **Gemma-3 Multimodal 12b**: (8.1Gb)
    ```bash
    ./ollama/bin/ollama pull gemma3:12b
    ```
    Pull **Gemma-3 Multimodal 12b-it-q8_0**: (13Gb)
    ```bash
    ./ollama/bin/ollama pull gemma3:12b-it-q8_0
    ```
3. Execute Ollama: 
    ```
    ssh <username>@<nodename> # e.g: ssh anhu@tron33
    ./ollama/bin/ollama run llama3.2-vision
    ``` -->

## II. Installation Guide
```bash
export HOME=/fs/nexus-projects/health_sim_ai # replace your HOME directory here
python3.12 venv -m venvs/llm
source ./venvs/llm/bin/activate
pip install -r requirements.txt
```

## III. Usage: overall pipeline (after installation)
### Step 0. Env activation
```bash
export HOME=/fs/nexus-projects/health_sim_ai # replace your HOME directory here
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR
source ./venvs/llm/bin/activate
cd src
```
---
### Step 1. SFT Task dataset split and creation (Please read)

The **purpose** of this step is to generate **training- and testing-ready datasets** for *trait-aware alignment* using supervised fine-tuning (SFT).

### (a) Dataset Structure
Each dataset contains the following core fields:
- **`Input`**: Carefully crafted prompts with properly parsed attributes  
  *(e.g., demographic traits, personality traits, visual stimuli)*  
- **`Target`**: Processed ground-truth responses  
  *(e.g., numerical values for Likert-scale items, short text labels for impact calibration)*

### (b) Prerequisites
- The cleaned, pre-screened, and pre-formatted source dataset **must** be available at:  
  `data/survey_responses_screened.csv`
- **Note**: If the datasets listed below already exist in the `data/` directory, you do **not** need to re-run the commands in this step.


    Trait-aware Aligment Sets:
    ### Task 1: *Trait-Aware Response Prediction*

    **Holdout by poster type**

    1. **Neutral Holdout**
    - `task1_it_train_holdout_neutral.csv`  
        *(Training)*: Traits + responses to **["threatening", "self-efficacy"]** posters  
    - `task1_it_test_holdout_neutral.csv`  
        *(Testing)*: Traits + responses to **["informational / neutral"]** posters  

    2. **Self-Efficacy Holdout**
    - `task1_it_train_holdout_efficacy.csv`  
        *(Training)*: Traits + responses to **["threatening", "informational / neutral"]** posters  
    - `task1_it_test_holdout_efficacy.csv`  
        *(Testing)*: Traits + responses to **["self-efficacy"]** posters  

    3. **Threatening Holdout**
    - `task1_it_train_holdout_threatening.csv`  
        *(Training)*: Traits + responses to **["self-efficacy", "informational / neutral"]** posters  
    - `task1_it_test_holdout_threatening.csv`  
        *(Testing)*: Traits + responses to **["threatening"]** posters  

    ---

    <!-- ### Task 2: *Trait-Aware Communication Strategy Impact Calibration*

    **Holdout by topic**

    - `task2_it_train_holdout_NutriHeart.csv`  
    *(Training)*: Traits + impact scores for posters from **all topics except ["Heart Disease", "Nutrition"]**  
    - `task2_it_test_holdout_NutriHeart.csv`  
    *(Testing)*: Traits + impact scores for posters from **["Heart Disease", "Nutrition"]** -->

    ---

    <!-- * **Task 1:** *Trait-Aware Response Prediction*
        1. `mcq_it_train_v3a`: (training) individuals' traits & their responses to `['threatening', 'self-efficacy']` posters
        2. `mcq_it_test_v3a`: (testing) individials' trait & their responses to `['informational / neutral']` posters 
        ---
        3. `mcq_it_train_v3b`: (training) traits & responses to `['threatening', 'informational / neutral']` posters
        4. `mcq_it_test_v3b`: (testing) traits & responses to `['self-efficacy']` posters
        ---
        5. `mcq_it_train_v3c`: (training) traits & responses to `['self-efficacy', 'informational / neutral']` posters
        6. `mcq_it_test_v3c`: (testing) trait & responses to `['threatening']` posters
    
    * **Task 2:** *Trait-Aware Communication Strategy Impact Caliberation*
        1. `mcq_it_train_recommender_v4_NutriHeart.csv`: (training) individuals' traits & impact scores of posters in *ALL topics except `["Heart Disease", "Nutrition"]`* on them.
        2. `mcq_it_test__recommender_v4_NutriHeart.csv`: (testing) individuals' traits & impact scores of posters in *`["Heart Disease", "Nutrition"]`* on them. -->

* Train/Test set format: columns `["instruction", "answer", "set"]`


### (c) Dataset Generation Commands
Execute these commands to craft the dataset(s) described above
```bash
#####################################
# create dataset(s) for Task 1
#####################################
# a) full traits (in the paper)
python create_dataset_task1.py --demo_full --include_big5 --include_facet --include_locus
# b) partial traits: 9 selected demographics + Big5 (pilot test)
python create_dataset_task1.py --include_big5

# # create dataset(s) for Task 2
# python create_dataset_task2.py
```
<!-- ```bash
# create dataset(s) for Task 1
python create_dataset_v3.py
# create dataset(s) for Task 2
python create_dataset_v4.py
``` -->


### Step 2. Model Training
* Execute one of these 2 commands to train model for each specific task
    ```bash
    ########################################################################
    # Task 1: VLM-enabled trait-aware response prediction
    #       --model: [gemma, llama, qwen]
    #       --visual_stimuli: [True, False] 
    #                       # ablation study on visual impacts
    #       --n_epochs: int # our experiment: 1
    #       --test_style: [neutral, efficacy, threatening, train_on_all] 
    #                       # styles of unseen test posters
    #       --partial_traits: [True, False]
    #                       # use full (all demographics + Big5 + locus of control) 
                            # or partial traits (9 demographics + Big5)
    #
    #   * "train_on_all": train on all available data
    #   * due to implementation issue, please use gemma 
    #       when set visual_stimuli=False
    # e.g. train multimodal Gemma with unseen "Information/Neutral" posters
    ########################################################################
    python train_pred_llm.py \
        --model gemma \ # pixtral is not working due to version issues
        --visual_stimuli True \
        --n_epochs 1 \
        --test_style neutral
    
    ########################################################################
    # Task 2: LLM-enabled trait-aware style impact caliberation
    #       --model: [gemma, llama]
    #       --n_epochs: int # number of training epochs; our experiment: 7
    #       --resample: [True, False] # our experiment: False
    #                   # resample positive impacts <=> rebalance dataset
    #                   # 1) include very positive samples only (score=9)
    #                   # 2) duplicate very negative samples (score=1) 
    #                   #    => increase under-represented data
    #
    ########################################################################
    # python train_rec_llm.py
    ```
* Model training details:
    1. Load pre-processed train/test set in Step 1 (compatible to UnSloth implementation)
    2. Train/Test set are already processed with Instruction Tuning prompts (with embedded trait data & questions), plus the target response
    3. Train model with LoRA using UnSloth
        * configs are initialized with `configs/trainer_config.py`

### Step 3. Model Inference
* Execute one of these two commands to evaluate on each specific task (inference step)
```bash
########################################################################
# Task 1: VLM-enabled trait-aware response prediction
#   * NOTE: all arguments, including model weights, export path,
#           saving name, etc. should be modified in
#           `configs/task1_model_inference.yaml` 
########################################################################
python inference_pred_llm.py # args in `configs/task1_model_inference.yaml`

#######    [NOTE] ablation study    ########
# To set ablation study, use `configs/task1_model_inference_ablat.yaml`
#######         [END NOTE]          ########

# ########################################################################
# # Task 2: LLM-enabled trait-aware style impact caliberation
# #       * NOTE: all arguments, including model weights, export path,
# #          saving name, etc. should be modified in
# #          `configs/task2_model_inference.yaml`
# ########################################################################
# python inference_rec_llm.py
```

* The evaluation results are stored in `src/evals/`

* Model:s <br>
    * Each model's responses are recorded in an unique columns, formatted as dictionary `{"Q1": int, "Q2": int, ...}` <br>
    * The models are as follow:
    1. `gemma3-12b (zero-shot)`
    2. `gemma3-4b (zero-shot)`
    3. `gemma3-12b (FT)`
    4. `gemma3-4b (FT)`
        ***
    5. `Llama3.2-11B-Vision (zero-shot)`
    6. `Llama3.2-11B-Vision (FT)`
        ***
    7. `gemma3-4b (no vision, FT)` 
    8. `gemma3-4b (no trait, FT)`
    9. `gemma3-4b (no trait, zero-shot)`
        ***
    10. `Qwen2.5-VL-7B (zero-shot)`
    11. `Qwen2.5-VL-7B (FT)`
        ***



### Step 4. Model Evaluation / Performance Analysis
* `healthai_model_analysis.ipynb`
    1. Evaluate model performance
    2. Compute random baseline on-the-fly
        * `policy-random-uniform`: random baseline with uniform sampler
        * `policy-random-priors`: random baseline guided by prior distributions