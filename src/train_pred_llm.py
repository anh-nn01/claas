"""
    Instruction Tuning of VLM for Trait-conditioned Response Prediction
"""

import os
import torch    # type: ignore
from unsloth import FastVisionModel, FastModel # type: ignore
from unsloth import is_bf16_supported   # type: ignore
from unsloth.trainer import UnslothVisionDataCollator   # type: ignore
from unsloth.chat_templates import get_chat_template    # type: ignore
from unsloth.chat_templates import standardize_data_formats # type: ignore
from unsloth.chat_templates import standardize_sharegpt # type: ignore
from unsloth.chat_templates import train_on_responses_only  # type: ignore
from unsloth import is_bf16_supported   # type: ignore
from trl import SFTTrainer, SFTConfig   # type: ignore
from transformers import TrainingArguments  # type: ignore
from configs.trainer_config import get_config
from datasets import Dataset # type: ignore

import torch    # type: ignore
from datasets import load_dataset, load_from_disk   # type: ignore
from PIL import Image   # type: ignore
import pandas as pd # type: ignore
from PIL import Image # type: ignore

from utils import convert_to_model_trainer_dataset
from termcolor import colored # type: ignore
from dotenv import load_dotenv # type:ignore

import wandb # type: ignore
from transformers.integrations import WandbCallback # type: ignore
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")



""""""""" PARSE CMD-LINE INPUT HYPERPARAMETERS """""""""
# model = "gemma"         # gemma, llama, qwen, or llava
# visual_stimuli = True   # True: original setup; False: ablation study on no visual stimuli
# n_epochs = 1            # training epochs
# test_style = "neutral" # neutral, efficacy, threatening
parser = argparse.ArgumentParser(description="Training arguments")
# Add argument model type
parser.add_argument(
    "--model",
    choices=["gemma", "llama", "qwen"],
    default="gemma",
    help="Model to use: gemma, llama, qwen (default: gemma)"
)
# Add argument visual stimuli (for ablation)
def str2bool(v):
    return v.lower() in ('true', '1', 'yes')
parser.add_argument(
    "--visual_stimuli",
    type=str2bool,
    default=True,
    help="Include visual stimuli: True or False (default: True)"
)
parser.set_defaults(visual_stimuli=True)
# Add argument n_epochs
parser.add_argument(
    "--n_epochs",
    type=int,
    default=1,
    help="Number of training epochs (default: 1)"
)
# Add argument test_style => specify train/test splits for task 1
parser.add_argument(
    "--test_style",
    choices=["neutral", "efficacy", "threatening", "train_on_all"],
    default="neutral",
    help="Test style: neutral, efficacy, or threatening (default: neutral)"
)
# Add argument demo_full => specify train/test splits for task 1
parser.add_argument(
    "--partial_traits",
    type=str2bool,
    default=False,
    help="Whether to use partial traits (only selected 9 demographic + Big5 traits)"
)
# Add argument demo_full => specify train/test splits for task 1
parser.add_argument(
    "--sph_traits",
    type=str2bool,
    default=False,
    help="Whether to use SPH-requested traits (only selected 3 demographic: [Gender, Age, Race/Ethnicity])"
)
# Add argument ablations => ablation names
parser.add_argument(
    "--ablation",
    choices=["", "noDemo", "noBig5", "noFacet", "noLocus", "noBig5_noFacet_noLocus"],
    default="",
    help="Ablation on which trait group: noDemo, noBig5, noFacet, noLocus (default: '')"
)
# Add argument demo_full => specify train/test splits for task 1
parser.add_argument(
    "--resume_from_checkpoint",
    type=str2bool,
    default=False,
    help="Continue fine-tuning partial traits from FT models on full traits (use only when partial_traits=True)"
)
# Parse arguments
args = parser.parse_args()
# Assign to variables
model = args.model
visual_stimuli = args.visual_stimuli
n_epochs = args.n_epochs
test_style = args.test_style
ablation = args.ablation
partial_traits = args.partial_traits
sph_traits = args.sph_traits
# sph-requested traits overrides partial_traits
if sph_traits:
    partial_traits = True # set partial_traits to True as well
# always set to False for now: no FT resume
resume_from_checkpoint = False # args.resume_from_checkpoint
"""======================================="""





"""================================= 
    1. Load model configs

    Model & Training settings
    (model, lora, seed, etc)
================================="""
# load model cfg
model_cfg = get_config(model, visual_stimuli=visual_stimuli)
# update number of training epochs
model_cfg.num_train_epochs = n_epochs

#########################################################
# update dataset paths based on:
# + test_style
# + trait selection settings
#########################################################
if test_style == "train_on_all":
    print("Training on all styles (deploy).")
    if not (partial_traits or sph_traits) and (ablation == ""):
        print("Using FULL traits (all demographics + personality + locus).")
        model_cfg.train_path = os.path.abspath(f"../data/task1_it_train_all_styles.csv") # training dataset path
        model_cfg.test_path = os.path.abspath(f"../data/task1_it_test_all_styles.csv")  # testing dataset path
    else:
        if ablation != "":
            print(f"Ablation on {ablation} traits.")
            model_cfg.train_path = os.path.abspath(f"../data/task1_it_train_all_styles_{ablation}.csv") # training dataset path
            model_cfg.test_path = os.path.abspath(f"../data/task1_it_test_all_styles_{ablation}.csv")  # testing dataset path
        elif sph_traits:
            print("Using SPH-requested traits (only 3 selected demographics: Gender, Race/Ethnicity, Age).")
            model_cfg.train_path = os.path.abspath(f"../data/task1_it_train_all_styles_5sensitiveDemo_noBig5_noFacet_noLocus.csv") # training dataset path
            model_cfg.test_path = os.path.abspath(f"../data/task1_it_test_all_styles_5sensitiveDemo_noBig5_noFacet_noLocus.csv")  # testing dataset path
        else:
            print("Using partial traits (only 9 selected demographics + Big5 traits)")
            model_cfg.train_path = os.path.abspath(f"../data/task1_it_train_all_styles_partialDemo_noFacet_noLocus.csv") # training dataset path
            model_cfg.test_path = os.path.abspath(f"../data/task1_it_test_all_styles_partialDemo_noFacet_noLocus.csv")  # testing dataset path
else:
    if not (partial_traits or sph_traits) and (ablation == ""):
        print("Using FULL traits (all demographics + personality + locus).")
        model_cfg.train_path = os.path.abspath(f"../data/task1_it_train_holdout_{test_style}.csv") # training dataset path
        model_cfg.test_path = os.path.abspath(f"../data/task1_it_test_holdout_{test_style}.csv")  # testing dataset path
    else:
        if ablation != "":
            print(f"Ablation on {ablation} traits.")
            model_cfg.train_path = os.path.abspath(f"../data/task1_it_train_holdout_{test_style}_{ablation}.csv") # training dataset path
            model_cfg.test_path = os.path.abspath(f"../data/task1_it_test_holdout_{test_style}_{ablation}.csv")  # testing dataset path
        elif sph_traits:
            print("Using SPH-requested traits (only 3 selected demographics:  Gender, Race/Ethnicity, Age).")
            model_cfg.train_path = os.path.abspath(f"../data/task1_it_train_holdout_{test_style}_5sensitiveDemo_noBig5_noFacet_noLocus.csv") # training dataset path
            model_cfg.test_path = os.path.abspath(f"../data/task1_it_test_holdout_{test_style}_5sensitiveDemo_noBig5_noFacet_noLocus.csv")  # testing dataset path
        else:
            print("Using partial traits (only 9 selected demographics + Big5 traits)")
            model_cfg.train_path = os.path.abspath(f"../data/task1_it_train_holdout_{test_style}_partialDemo_noFacet_noLocus.csv") # training dataset path
            model_cfg.test_path = os.path.abspath(f"../data/task1_it_test_holdout_{test_style}_partialDemo_noFacet_noLocus.csv")  # testing dataset path



        
assert os.path.exists(model_cfg.train_path), f"Train path {model_cfg.train_path} does not exist! Please create one!"
assert os.path.exists(model_cfg.test_path), f"Test path {model_cfg.test_path} does not exist! Please create one!"

###################################################
# model checkpoint name based on configs:
#   + model_name 
#   + "task1" 
#   + epochs/steps 
#   + dataversion 
#   + novision (optional; ablation only)
###################################################
save_name = f"{model_cfg.model_name}_task1"
if model_cfg.num_train_epochs > 0:
    save_name += f"_{model_cfg.num_train_epochs}_epochs"
else:
    save_name += f"_{model_cfg.max_steps}_steps"

save_name += f"_test_{test_style}"


# ablation: notrait
if ablation != "":
    save_name += f"_ablation_{ablation}"
# ablation: novision
if not model_cfg.visual_stimuli:
    save_name += "_novision"

# partial traits: selected traits + Big5 only
resume_ckpt_path = None
if partial_traits:
    save_name += "_partialTraits"
    if sph_traits:
        save_name += "_sphTraits"
        
    # continue from FT model on full traits
    if resume_from_checkpoint:
        save_name += "_resumeFromCheckpoint"
        # load FT model on full trait (Stage 1)
        resume_ckpt_path = model_cfg.model_name + f"_task1_1_epochs_test_{test_style}"


print()
print(colored('MODEL: ' + model_cfg.model_name, 'green', attrs=["bold"]))
print(colored('Train data path :', 'green', attrs=["bold"]), model_cfg.train_path)
print(colored('Lora rank       :', 'green', attrs=["bold"]), model_cfg.r)
print(colored('Max seq length  :', 'green', attrs=["bold"]), model_cfg.max_seq_length)
print(colored('Vision tuning   :', 'green', attrs=["bold"]), model_cfg.finetune_vision_layers)
print(colored('Language tuning :', 'green', attrs=["bold"]), model_cfg.finetune_language_layers)
print(colored('Attention tuning:', 'green', attrs=["bold"]), model_cfg.finetune_attention_modules)
print(colored('Num epochs      :', 'green', attrs=["bold"]), model_cfg.num_train_epochs)
print(colored('Wandb project   :', 'green', attrs=["bold"]), model_cfg.wandb_project)
print(colored('Visual stimuli  :', 'green', attrs=["bold"]), model_cfg.visual_stimuli)
print(colored('Partial Traits  :', 'green', attrs=["bold"]), partial_traits)
print(colored('SPH-selected Traits :', 'green', attrs=["bold"]), sph_traits)
print(colored('Saving path     :', 'green', attrs=["bold"]), save_name)
print()
print()


""" Wanb initialization"""
wandb.init(
    project=model_cfg.wandb_project,
    name=model_cfg.wandb_run_name or model_cfg.get_save_name(),
    tags=model_cfg.wandb_tags,
    config=model_cfg  # Log all hyperparameters
)



"""================================= 
    2. Initialize model

    a) load pretrained model
    b) initialize peft (lora)
    c) Enable model training

    * Model defined differently
      based on config `visual_stimuli`
    => multimodal   : FastVisionModel
    => language-only: FastModel
================================="""
if model_cfg.visual_stimuli:
    """ NOTE: ORIGINAL SETUP WITH VISUAL STIMULI"""
    # load pretrained weights
    model, tokenizer = FastVisionModel.from_pretrained(
        model_cfg.model_name,
        load_in_4bit = model_cfg.load_in_4bit, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )
    # initialize peft (lora) model: efficient training
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = model_cfg.finetune_vision_layers,
        finetune_language_layers   = model_cfg.finetune_language_layers,
        finetune_attention_modules = model_cfg.finetune_attention_modules,
        finetune_mlp_modules       = model_cfg.finetune_mlp_modules,

        r = model_cfg.r,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = model_cfg.lora_alpha,  # Recommended alpha == r at least
        lora_dropout = model_cfg.lora_dropout,
        bias = model_cfg.bias,
        random_state = model_cfg.random_state,
        use_rslora = model_cfg.use_rslora,  # support rank stabilized LoRA
        loftq_config = model_cfg.loftq_config, # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )
    # enable model training (gradient graph)
    FastVisionModel.for_training(model) # Enable for training!
    # print(model)
else:
    """ NOTE: ABLATION STUDY ON NO VISUAL STIMULI"""
    # load pretrained weights
    model, tokenizer = FastModel.from_pretrained(
        model_cfg.model_name,
        load_in_4bit = model_cfg.load_in_4bit, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )
    # initialize peft (lora) model: efficient training
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = model_cfg.finetune_vision_layers,
        finetune_language_layers   = model_cfg.finetune_language_layers,
        finetune_attention_modules = model_cfg.finetune_attention_modules,
        finetune_mlp_modules       = model_cfg.finetune_mlp_modules,

        r = model_cfg.r,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = model_cfg.lora_alpha,  # Recommended alpha == r at least
        lora_dropout = model_cfg.lora_dropout,
        bias = model_cfg.bias,
        random_state = model_cfg.random_state,
        use_rslora = model_cfg.use_rslora,  # support rank stabilized LoRA
        loftq_config = model_cfg.loftq_config, # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )
    # enable model training (gradient graph)
    FastModel.for_training(model) # Enable for training!




"""=================================
    3. Initialize dataset

    a) load original CSV dataset
    b) convert to 
       trainer-compatible dataset
================================="""
# load raw dataset
df_train = pd.read_csv(model_cfg.train_path)
df_test = pd.read_csv(model_cfg.test_path)
print(colored('Train/Test Quantity:', 'green'), len(df_train), ';', len(df_test))
# load into trainer-compatible dataset format
if model_cfg.visual_stimuli:
    """ NOTE: ORIGINAL SETUP WITH VISUAL STIMULI"""
    train_dataset = convert_to_model_trainer_dataset(df_train[:])
    test_dataset = convert_to_model_trainer_dataset(df_test[:])
else:
    """ NOTE: ABLATION STUDY ON NO VISUAL STIMULI"""
    train_dataset = convert_to_model_trainer_dataset(df_train[:], use_image=False)
    test_dataset = convert_to_model_trainer_dataset(df_test[:], use_image=False)
# print(train_dataset[0])

def count_words(text):
    words = text.split()
    return len(words)
print(colored(f'Advice: Please make sure max_seq_length={model_cfg.max_seq_length} is sufficient \
to capture context of {count_words(df_train.loc[0, 'instruction'])} words AND Image!', 'yellow'))


del df_train, df_test # save memory

# convert tokenizer to gemma template
if "gemma" in model_cfg.model_name:
    # gemma-specific tokenizer chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3",
    )
    # gemma-specific data format standardization
    train_dataset = standardize_data_formats(train_dataset)
    print(train_dataset[99])
    
# additional processing for non-visual stimuli (text only)
if not model_cfg.visual_stimuli:
    """ NOTE: ABLATION STUDY ON NO VISUAL STIMULI"""
    if "gemma" in model_cfg.model_name:
        # copied from gemma3(4b) notebook Unsloth tutorial
        def apply_chat_template(examples):
            texts = tokenizer.apply_chat_template(examples["messages"])
            return { "text" : texts }
        train_dataset = Dataset.from_list(train_dataset)
        train_dataset = train_dataset.map(apply_chat_template, batched = True)
        print(type(train_dataset))
    # elif "Llama" in model_cfg.model_name:
    #     # llama-specific tokenizer chat template
    #     tokenizer = get_chat_template(
    #         tokenizer,
    #         chat_template = "llama-3.1",
    #     )
    #     # copied from `Llama3.2_(1B_and_3B)-Conversational.ipynb` Unsloth tutorial
    #     def formatting_prompts_func(examples):
    #         convos = examples["messages"]
    #         texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    #         return { "text" : texts, }
    #     # data format standardization
    #     train_dataset = Dataset.from_list(train_dataset)
    #     train_dataset = standardize_sharegpt(train_dataset)
    #     train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)
    #     print(type(train_dataset))
    else:
        pass

"""=================================
    4. Initialize model trainer
================================="""
""" Training configurations """
callbacks = [WandbCallback] if model_cfg.report_to == "wandb" else []


################################################
### Pixtral requires fixed image size        ###
### Others (Gemma, Llama, Qwen): flexible    ###
################################################
if model == "pixtral":
    data_collator = UnslothVisionDataCollator(model, tokenizer, resize=512) # Must use!
else:
    data_collator = UnslothVisionDataCollator(model, tokenizer) # Must use!

if model_cfg.visual_stimuli:
    """ NOTE: ORIGINAL SETUP WITH VISUAL STIMULI"""
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = data_collator, # Must use!
        train_dataset = train_dataset,
        args = SFTConfig(
            per_device_train_batch_size = model_cfg.per_device_train_batch_size,
            gradient_accumulation_steps = model_cfg.gradient_accumulation_steps,
            warmup_steps = model_cfg.gradient_accumulation_steps,
            # max_steps = model_cfg.max_steps,
            num_train_epochs = model_cfg.num_train_epochs, # Set this instead of max_steps for full training runs
            learning_rate = model_cfg.learning_rate,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_steps = model_cfg.logging_steps,
            optim = model_cfg.optim,
            weight_decay = model_cfg.weight_decay,
            lr_scheduler_type = model_cfg.lr_scheduler_type,
            seed = model_cfg.seed,
            output_dir = model_cfg.output_dir,
            report_to = model_cfg.report_to,     # For Weights and Biases

            # for resuming from checkpoints
            save_steps = 1000,
            resume_from_checkpoint = resume_ckpt_path,

            # # You MUST put the below items for vision finetuning:
            # remove_unused_columns = False,
            # dataset_text_field = "",
            # dataset_kwargs = {"skip_prepare_dataset": True},
            # dataset_num_proc = 4,
            # max_seq_length = 2048,
        ),
    )
else:
    """ NOTE: ABLATION STUDY ON NO VISUAL STIMULI"""
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        # data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
        train_dataset = train_dataset,
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = model_cfg.per_device_train_batch_size,
            gradient_accumulation_steps = model_cfg.gradient_accumulation_steps,
            warmup_steps = model_cfg.gradient_accumulation_steps,
            # max_steps = model_cfg.max_steps,
            num_train_epochs = model_cfg.num_train_epochs, # Set this instead of max_steps for full training runs
            learning_rate = model_cfg.learning_rate,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_steps = model_cfg.logging_steps,
            optim = model_cfg.optim,
            weight_decay = model_cfg.weight_decay,
            lr_scheduler_type = model_cfg.lr_scheduler_type,
            seed = model_cfg.seed,
            output_dir = model_cfg.output_dir,
            report_to = model_cfg.report_to,     # For Weights and Biases
        ),
    )
    # only train on the assistant outputs and ignore the loss on the user's inputs. 
    # This helps increase accuracy of finetunes
    if "gemma" in model_cfg.model_name:
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<start_of_turn>user\n",
            response_part = "<start_of_turn>model\n",
        )
    elif "Llama" in model_cfg.model_name:
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
            response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    else:
        pass



"""=================================
    5. Display memory stats
================================="""
# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


"""=================================
    6. Model training
================================="""
trainer_stats = trainer.train(resume_from_checkpoint)

wandb.log({
    "final/train_loss": trainer_stats.metrics["train_loss"],
    "final/train_runtime": trainer_stats.metrics["train_runtime"],
    "final/train_samples_per_second": trainer_stats.metrics["train_samples_per_second"],
})

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


model.save_pretrained(save_name)
tokenizer.save_pretrained(save_name)