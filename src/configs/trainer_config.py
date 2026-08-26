import os
import sys
import torch # type: ignore
from dataclasses import dataclass # type: ignore
from typing import Optional, Dict

@dataclass
class BaseTrainingConfig:
	model_name: str = "unsloth/Pixtral-12B-2409"
	load_in_4bit: bool = True #True
	load_in_8bit: bool = False #True
	use_gradient_checkpointing: bool = True
	
	full_finetuning: bool = False
	visual_stimuli: bool = True # ablation study on visual stimuli's impacts

	# LoRA parameters
	finetune_vision_layers: bool = False
	finetune_language_layers: bool = True
	finetune_attention_modules: bool = True
	finetune_mlp_modules: bool = True
	r: int = 8
	lora_alpha: int = 8
	lora_dropout: float = 0
	bias: str = "none"
	random_state: int = 99
	use_rslora: bool = False
	loftq_config: Optional[Dict] = None

	#######################
	###  Dataset paths  ###
	#######################
	train_path = None # training dataset path; specify at training script
	test_path = None  # testing dataset path; specify at training script
	
	# Training parameters
	per_device_train_batch_size: int = 1
	gradient_accumulation_steps: int = 4
	warmup_steps: int = 5
	max_steps: int = 2000 #500 #250, 
	num_train_epochs = 1 # if num_train_epochs > 0, max_steps=None
	learning_rate: float = 2e-4
	logging_steps: int = 1
	optim: str = "paged_adamw_8bit"
	weight_decay: float = 0.01
	lr_scheduler_type: str = "linear"
	seed: int = 99
	report_to: str =  "none" #"wandb"
	wandb_project: str = "health_sim_ai"
	wandb_run_name : Optional[str] = None
	wandb_tags: Optional[list] = None
	output_dir: str = "./output_directory"
	# You MUST put the below items for vision finetuning:
	remove_unused_columns = False,
	dataset_text_field = "text", #text
	dataset_kwargs = {"skip_prepare_dataset": True},
	dataset_num_proc = 4,
	max_seq_length = 2048 # 1024 # 512 # 256, #512, #1024 #2048, => for MCQs
	# dtype = None #torch.bfloat16,
	
	def get_save_name(self):
		"""Generate descriptive save name based on config parameters"""
		model_shortname = self.model_name.split('/')[-1].lower()
				
		params = [
			f"max_seq{self.max_seq_length}",
			f"bs{self.per_device_train_batch_size}",
			f"ga{self.gradient_accumulation_steps}",
			f"lr{self.learning_rate:.1e}",
			f"steps{self.max_steps}",
			f"epochs{self.num_train_epochs}",
			f"r{self.r}",
			f"alpha{self.lora_alpha}",
			
		]
		
		if self.finetune_vision_layers:
			params.append("vision")
		if self.finetune_language_layers:
			params.append("lang")
		if self.finetune_attention_modules:
			params.append("attn")
		if self.finetune_mlp_modules:
			params.append("mlp")
			
		param_str = "_".join(params)
		return f"{model_shortname}_{param_str}"
	
def get_config(model_type: str = "pixtral", **kwargs):
	config_class = MODEL_CONFIGS.get(model_type.lower(), GemmaConfig)
	return config_class(**kwargs)

@dataclass
class QwenConfig(BaseTrainingConfig):
	"""Configuration for Qwen models"""
	model_name: str = "unsloth/Qwen2-VL-7B-Instruct"
	per_device_train_batch_size: int = 2
	gradient_accumulation_steps: int = 4
	finetune_attention_modules: bool = True
	output_dir = "./output_directory"

@dataclass
class Qwen3VLConfig(BaseTrainingConfig):
	"""Configuration for Qwen3-VL models"""
	model_name: str = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
	per_device_train_batch_size: int = 2
	gradient_accumulation_steps: int = 4
	finetune_attention_modules: bool = True
	output_dir = "./output_directory"

@dataclass
class LlavaConfig(BaseTrainingConfig):
	"""Configuration for Llava models"""
	model_name: str = "unsloth/llava-v1.6-mistral-7b-hf" # -bnb-4bit
	per_device_train_batch_size: int = 2
	gradient_accumulation_steps: int = 4
	finetune_attention_modules: bool = True
	output_dir = "./output_directory"

@dataclass
class PixtralConfig(BaseTrainingConfig):
	"""Configuration for Pixtral models"""
	model_name: str = "unsloth/Pixtral-12B-2409-bnb-4bit"
	per_device_train_batch_size: int = 1
	gradient_accumulation_steps: int = 4
	finetune_attention_modules: bool = True
	output_dir = "./output_directory"

@dataclass
class LlamaConfig(BaseTrainingConfig):
	""" Configuration for LLaMA models 
		=> use pretrained 11b multimodal model
			or pretrained 8b language-only model (implementation purpose)
	"""
	def __post_init__(self):
		# define model from `self.visual_stimuli`` in super class
		if self.visual_stimuli:
			# Multimodal VLM: Llama-3.2
			self.model_name: str = "unsloth/Llama-3.2-11B-Vision-Instruct" #"unsloth/Llama-3.2-11B-Vision-bnb-4bit"
		else:
			# Language-only LLM: Llama-3.1
			self.model_name: str = "unsloth/Llama-3.1-8B-Instruct" # zero-shot model

	finetune_attention_modules: bool = True
	per_device_train_batch_size: int = 2
	gradient_accumulation_steps: int = 4
	# r = 16, 
	# lora_alpha = 16
	# num_train_epochs: int = 2
	learning_rate: float = 3e-4  
	output_dir = "./output_directory"
	# output_dir: str = "./output_directory/llama3.2-recommender-outputs" # "./output_directory/llama3.2-style-outputs"
	# output_dir: str = "./output_directory/llama3.1-recommender-outputs" # "./output_directory/llama3.2-style-outputs"

@dataclass
class GemmaConfig(BaseTrainingConfig):
	""" Configuration for Gemma models 
		=> use base pretrained model (pt) quantized to 4bit
	"""
	model_name: str = "unsloth/gemma-3-12b-it" # "unsloth/gemma-3-12b-pt-unsloth-bnb-4bit" #"unsloth/gemma-3-4b-it" #"unsloth/gemma-3-4b-it-unsloth-bnb-4bit" #"unsloth/gemma-3-4b-it" #"unsloth/gemma-7b"
	# model_name: str = "unsloth/gemma-3-4b-it" # "unsloth/gemma-3-12b-pt-unsloth-bnb-4bit" #"unsloth/gemma-3-4b-it" #"unsloth/gemma-3-4b-it-unsloth-bnb-4bit" #"unsloth/gemma-3-4b-it" #"unsloth/gemma-7b"
	per_device_train_batch_size: int = 1
	gradient_accumulation_steps: int = 4
	finetune_attention_modules: bool = True
	num_train_epochs: int = 1 # 5
	learning_rate: float = 3e-4  
	output_dir = "./output_directory"
	# output_dir: str = "./output_directory/gemma3-12b-recommender-outputs" 
	# output_dir: str = "./output_directory/gemma3-4b-recommender-outputs" # "./output_directory/gemma3-4b-style-outputs" # "./output_directory/gemma3-4b-style_test_threat-outputs" # "./output_directory/gemma3-4b-style_test_neutral-outputs" # "./output_directory/gemma3-outputs"
	
# Dictionary mapping for easy access
MODEL_CONFIGS = {
	"llama": LlamaConfig,
	"gemma": GemmaConfig,
	"qwen": QwenConfig,
	"llava": LlavaConfig,
	"qwen3vl": Qwen3VLConfig,
	# "pixtral": PixtralConfig,
}