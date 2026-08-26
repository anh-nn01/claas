"""
	Instruction Tuning of LLM for Trait-conditioned Style Impact Caliberation
"""

import ollama   # type: ignore
import yaml     # type: ignore
import pandas as pd # type: ignore
import os
from PIL import Image # type: ignore

import torch    # type: ignore
from langchain_community.chat_models import ChatOllama # type: ignore
from langchain_core.messages import SystemMessage, HumanMessage # type: ignore
from langchain_ollama import OllamaEmbeddings # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from pydantic import BaseModel # format LLM output as JSON # type: ignore
from unsloth import FastVisionModel, FastModel, FastLanguageModel # type: ignore
from transformers import TextStreamer # type: ignore
from unsloth.chat_templates import get_chat_template    # type: ignore
from unsloth.chat_templates import standardize_sharegpt # type: ignore

from utils import convert_to_base64, load_config, process_trait_info # type: ignore
from tqdm import tqdm   # type: ignore
from termcolor import colored # type: ignore

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""##########################################################################
	Trait-aware VLM/LLM response simulation to survey question
	1 inference for each question for robustness 
		(LLM forgets to answer some question very frequently)

	Args:
		+ sample: dataframe row with individual information 
				(demographic, personality, etc.)
				  survey questions (Q1-Q15) of the given `poster_id`
				  individual answers to the survey questions (A1-A15)
		+ cfgs  : model settings (e.g: seed, temperature)
		+ prompts: set of fixed prompts (SYSTEM, INSTRUCTION, etc.)
		+ task: (1) 'sim' 		=> community simulation
				(2) 'strategy' 	=> optimal communication strategy classification
				(3) 'classify'  => classify communication strategy of an image
		+ verbose: print out the prompts
	return: 
		+ simulated response by VLM or LLM
##########################################################################"""
def llm_simulate(model, sample, cfgs, prompts, task='sim', verbose=False):
	"""===============================
		1a. Get personality
			+ demographic data
	==============================="""
	# Personality & Demographic information
	demo_info = sample["Demographic"] # demographic information
	persona_score = sample["Persona_score"] # personality scores
	# persona_qa = sample["Persona_QA"] # personality survey questions & answers
	locus = sample["Locus"] # locus of control survey
	"""===============================
		1b. Get image data (encoded)
	==============================="""
	# Poster visual media information
	poster_id = sample["Poster_id"] # poster id
	image_path = os.path.abspath(f"../stimuli/{poster_id}.png")  # IMPORTANT: must be absolute path
	if cfgs["vision"]:
		image = Image.open(image_path)
		image_b64 = convert_to_base64(image)
	else:
		# ablation study: image (poster) removal 
		image = None # Image.new('RGB', (24,24))

	######################################################################################
	# 1*. modify trait info based on trait selection setings
	#       demo_full: wheter include full demographic traits or only selected ones
	#       include_big5, include_facet, include_locus: include big5 / facet / locus of control traits or not
	#     format: <trait>: <value> if available; else <trait>: [Not specified]
	######################################################################################
	demo_info, persona_score, locus = process_trait_info(
		demo_info, persona_score, locus,
		cfgs['include_demo'], cfgs['demo_full'], 
		cfgs['include_big5'],
		cfgs['include_facet'], 
		cfgs['include_locus']
	)


	"""===========================================
		1c. SYSTEM & SIM PROMPT CONSUTRUCTION
	==========================================="""
	if task == 'sim':
		""" Community simulation prompt"""
		# system prompt
		SYSTEM_PROMPT = prompts["SYSTEM_SIM"]
		# information for role-playing
		SIM_PROMPT = ""
		if not cfgs["trait"]:
			# ablation study: trait-awareness removal
			pass
		else:
			SIM_PROMPT += f"You are: {demo_info}\n"
			SIM_PROMPT += f"Your personality test shows you have (min score = 0; max score = 5): {persona_score}\n"
			# SIM_PROMPT += f"Your answers to the personality tests was: {persona_qa}\n"
			if locus is not None:
				SIM_PROMPT += f"You also have {locus}\n"
		# situation description
		SIM_PROMPT += prompts["SIMULATION_SIM"]
		# # questions
		# SIM_PROMPT += "The questions are marked starting from Q1 as followed:\n"
	elif task == 'strategy':
		raise NotImplementedError
	elif task == 'classify':
		raise NotImplementedError
	else:
		raise ValueError(f'No task named {task}!')
	
	
	"""===============================
		2. Construct each question
			into instruction
	==============================="""
	answers_json = {}
	""" Iterate through each question"""
	for i in range(1,16,1):
		# 1. intialize USER PROMPT with SIMULATION PROMPT with full demographic+personality data
		USER_PROMPT = SIM_PROMPT 
		# 2. specific survey question
		col = f"Q{i}"
		if not pd.isna(sample[col]):
			question = sample[col].replace("\n", " ")
			USER_PROMPT += f"Question: {question}\n\n" # specific question
			# instruction prompt to answer in proper format
			if "type in" in question.lower():
				USER_PROMPT += prompts['INSTRUCTION_FREE']
			else:
				USER_PROMPT += prompts['INSTRUCTION_MCQ']
		else:
			continue
		
		""""===============================
			3. MODEL INITIALIZATION
				AND INFERENCE STEP
		"==============================="""
		if cfgs["infer_engine"] == "ollama":
			#############################
			# Ollama inference engine
			#############################
			# Contruct LLM message
			messages = [
				SystemMessage(content=SYSTEM_PROMPT),
				HumanMessage(content=[
					{"type": "text", "text": USER_PROMPT}, # USER_PROMPT
					{"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}",} # IMAGE
				])
			]
			# Model inference
			answer = model.invoke(messages).content
			# all_answers += f"{col}: {answer}"
			answers_json[col] = answer
		elif cfgs["infer_engine"] == "unsloth":
			#############################
			# Unsloth model inference
			#############################
			
			
			if cfgs["vision"]:
				""" WITH VISUAL STIMULI"""
				# Contruct LLM message
				messages = [
					{"role": "user", "content": [
						{"type": "image"},
						{"type": "text", "text": SYSTEM_PROMPT + USER_PROMPT}
					]}
				]
				input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
				inputs = tokenizer(
					image.convert("RGB"),
					input_text,
					add_special_tokens = False,
					return_tensors = "pt",
				).to(device)
			else:
				""" WITHOUT VISUAL STIMULI"""
				# Contruct LLM message
				messages = [
					{"role": "user", "content": SYSTEM_PROMPT + USER_PROMPT}
				]
				# Text tokenization processing
				if "gemma" in cfgs["model"]:
					input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
					inputs = tokenizer(
						input_text,
						add_special_tokens = False,
						return_tensors = "pt",
					).to(device)
				# elif "Llama" in cfgs["model"]:
				# 	messages = standardize_sharegpt(messages)
				# 	inputs = tokenizer.apply_chat_template(
				# 		messages,
				# 		tokenize = True,
				# 		add_generation_prompt = True, # Must add for generation
				# 		return_tensors = "pt",
				# 	).to(device)
				else:
					raise NotImplementedError('Model inference not implemented.')
			# Model inference
			# text_streamer = TextStreamer(tokenizer, skip_prompt = True)
			gen_tokens = model.generate(
				**inputs, 
				max_new_tokens = 64,
				use_cache = True,
				min_p = 0.1, 
				do_sample=cfgs["stochastic"],
			)
			outs = tokenizer.batch_decode(gen_tokens[:, inputs.input_ids.shape[1]:])[0]
			answer = outs.replace(tokenizer.eos_token, "")
			answer = answer.replace("<end_of_turn>", "")
			answers_json[col] = answer
		else:
			raise ValueError(f'No inference engine {cfgs["infer_engine"]}')

		if verbose:
			# print('USER PROMPT:\n', USER_PROMPT)
			print(question)
			print('Model:', answer)
			print('True :', sample['A'+col[1:]])
			print('='*35)
			# print()
	
	return answers_json
	


"""###########################################################################
	Evaluate a given model (specified in model_cfgs)
	on posters with given test_style

	Args:
		+ cfgs			: specify model type (e.g. gemma or llama), 
						  data source, and export paths
		+ prompts		: set of prompts

	Outputs:
		=> save model in cfgs["export_path"] (CSV file)
			+ if cfgs["export_path"] not exists, initialize it with cfgs["data_path"] 
				=> original survey data with ground-truth responses
			+ add column "<model>:<version>": store AI-simulated responses
			+ support concurrent evaluation on different jobs
##########################################################################"""
if __name__ == '__main__':
	"""==========================================
		1. load model settings & prompts format
	=========================================="""
	######################################
	# Load model configs & prompts
	######################################
	### Original model config ###
	model_cfg = "./configs/task1_model_inference.yaml"
	### Ablation model config ###
	# model_cfg = "./configs/task1_model_inference_ablat.yaml"
	prompt_cfg = "./configs/prompts.yaml"
	cfgs = load_config(model_cfg)
	prompts = load_config(prompt_cfg)

	"""==========================================
		2. Evaluate model defined in configs
	=========================================="""
	print(colored('MODEL USE:', 'green'), cfgs["model"])
	# print(prompts['SYSTEM'])
	# print(prompts['INSTRUCTION'])
	
	"""===============================
		2. load dataset, if specified
	==============================="""
	data_path = os.path.expandvars(cfgs["data_path"])
	export_path = os.path.expandvars(cfgs["export_path"])
	print(colored('Eval data path:', 'blue'), colored(data_path, 'yellow'))
	print(colored('Export path   :', 'blue'), colored(export_path, 'yellow'))

	if not os.path.exists(export_path):
		# initialize <export_path> with <data_path>, if not exist
		# 	=> original survey data with ground-truth responses
		df = pd.read_csv(data_path)
		df.to_csv(export_path, index=False)
	else:
		# export_path is data_path + LLM responses
		# => just continue filling out new responses to export_path
		df = pd.read_csv(export_path)

	"""===============================
		3. Initialize model
	==============================="""
	if cfgs["infer_engine"] == "ollama":
		model = ChatOllama(
			model=cfgs["model"], 
			temperature=cfgs["temperature"],
			seed=cfgs["seed"],
		)
	elif cfgs["infer_engine"] == "unsloth":
		if cfgs["vision"]:
			# WITH VISUAL STIMULI
			model, tokenizer = FastVisionModel.from_pretrained(
				model_name=cfgs["model"],
				load_in_4bit=True,
			)
			FastVisionModel.for_inference(model)
			if "gemma" in cfgs["model"]:
				# gemma-specific tokenizer chat template
				tokenizer = get_chat_template(
					tokenizer,
					chat_template = "gemma-3",
				)
		else:
			# WITHOUT VISUAL STIMULI
			model, tokenizer = FastModel.from_pretrained(
				model_name=cfgs["model"],
				load_in_4bit=True,
			)
			# Initialize model tokenizer & inference pipeline
			if "gemma" in cfgs["model"]:
				# gemma-specific tokenizer chat template
				tokenizer = get_chat_template(tokenizer, chat_template = "gemma-3")
			elif "Llama" in cfgs["model"]:
				# Llama-specific tokenizer chat template
				tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")
			# set inference mode
			FastModel.for_inference(model)
	else:
		raise ValueError(f'No inference engine {cfgs["infer_engine"]}')
	
	"""=============================================
		4. LLM response simulation for each sample
			MODEL = model_name + version
			e.g. MODEL = "gemma3" + ":zeroshot"
	============================================="""
	# model name version
	MODEL = cfgs["model"] + cfgs["version"]
	# append names for ablation study versions
	#   1) no vision
	# 	2) no trait
	if ("vision" in cfgs) and (not cfgs["vision"]):
		MODEL = MODEL + "-novision"
	if ("trait" in cfgs) and (not cfgs["trait"]):
		MODEL = MODEL + "-notrait"
		
	if MODEL not in df.columns:
		# add column of LLM answers, if MODEL not already evaluated
		df[MODEL] = None 
	else:
		print(colored(f'WARNING: {MODEL} already in the dataset! Please verify if you want to proceed!', 'yellow'))
		
	for idx in tqdm(df.index[:], desc=f'Eval {MODEL}'):
		# sample = df.loc[idx]
		# answers_json = llm_simulate(model, sample, cfgs, prompts, verbose=False)
		# print(answers_json)

		#################################################
		# only update model responses if NAN
		#################################################
		if df.loc[idx, MODEL] is None or pd.isna(df.loc[idx, MODEL]):
			sample = df.loc[idx]
			answers_json = llm_simulate(model, sample, cfgs, prompts, verbose=False)
			df.loc[idx, MODEL] = [answers_json]

		if idx%100 == 0:
			# save partial outputs
			### reload file at export_path to identify any new models/columns 
			### => avoid overwritting models by other jobs
			df_temp = pd.read_csv(export_path) # NOTE: for concurrent inference of different models
			concurrent_models = [col for col in df_temp.columns if (col not in df.columns)]
			# to be safe: reload all unsloth models (if presented)
			# => because in our code, each model is named "unsloth/<model_name>:<version>"
			concurrent_models = concurrent_models + [col for col in df_temp.columns if ('unsloth' in col)]
			concurrent_models = concurrent_models + [col for col in df_temp.columns if ('novision' in col)]
			concurrent_models = concurrent_models + [col for col in df_temp.columns if ('notrait' in col)]
			concurrent_models = concurrent_models + [col for col in df_temp.columns if ('zeroshot' in col)]
			for col in concurrent_models:
				if col != MODEL: # only update concurrent models 
					df[col] = df_temp[col]
			del df_temp
			
			df.to_csv(export_path, index=False)
			print(colored('Save intermediate evaluations!', 'green'))

		""" Print out answers"""
		# sample = df.loc[idx]
		# answers_json = llm_simulate(model, sample, cfgs, prompts)
		# for col in answers_json:
		# 	print('Model', col, answers_json[col])
		# # True survey
		# for i in range(1,14,1):
		# 	print('True Q'+str(i)+':', sample[f'A{i}'])
		# print('='*99)
	
	##############################################
	# save partial outputs for concurrency 
	# of different model evals 
	# (e.g: zeroshot & FT on different jobs)
	# 	=> avoid overwritting saved responses 
	# 		of concurrent jobs
	##############################################
	df_temp = pd.read_csv(export_path) # NOTE: for concurrent inference of different models
	# 1. set of all models (columns) not in current df
	# 		=> in case other jobs have saved a new column for a new model
	# 		=> avoid overwritting saved responses of concurrent jobs
	concurrent_models = [col for col in df_temp.columns if (col not in df.columns)]
	# 2. to be safe: set of all unsloth models (if presented)
	#		=> because in our code, each model is named "unsloth/<model_name>:<version>"
	concurrent_models = concurrent_models + [col for col in df_temp.columns if ('unsloth' in col)]
	concurrent_models = concurrent_models + [col for col in df_temp.columns if ('novision' in col)]
	concurrent_models = concurrent_models + [col for col in df_temp.columns if ('notrait' in col)]
	concurrent_models = concurrent_models + [col for col in df_temp.columns if ('zeroshot' in col)]
	# 3. copied over all model responses that is currently NOT in df 
	# 		=> avoid overwritting
	for col in concurrent_models:
		if col != MODEL: # only update concurrent models 
			df[col] = df_temp[col]
	del df_temp
	df.to_csv(export_path, index=False)



# if __name__ == "__main__":
# 	# """"""""" PARSE CMD-LINE INPUT HYPERPARAMETERS """""""""
# 	# model = "gemma"         # gemma or llama
# 	# zeroshot = False		# False or True
# 	# visual_stimuli = True   # True: multimodal VLM; 
# 	# 						# False: language-only LLm => ablation study on "no vision"
# 	# trait = True 			# True: full trait-aware model;
# 	# 						# False: ablation study on "no trait"
# 	# version = "FT"			# zeroshot or FT
# 	# # test_style = "neutral" # neutral, efficacy, threatening
# 	# """======================================="""

# 	"""==========================================
# 		1. load model settings & prompts format
# 	=========================================="""
# 	######################################
# 	# Load model configs & prompts
# 	######################################
# 	model_cfg = "./configs/task1_model_inference.yaml"
# 	prompt_cfg = "./configs/prompts.yaml"
# 	cfgs = load_config(model_cfg)
# 	prompts = load_config(prompt_cfg)

# 	"""==========================================
# 		2. Evaluate model defined in configs
# 	=========================================="""
# 	eval_model(cfgs, prompts)

# 	# ######################################
# 	# # 1b. Load media's communication style classification
# 	# # 	Format: {
# 	# # 		<Poster_id> : <style>
# 	# # 	}
# 	# ######################################
# 	# style_path = '../data/media_style_posters.csv'
# 	# style_path = os.path.abspath(style_path)
# 	# df_styles = pd.read_csv(style_path)
# 	# media_styles = {}
# 	# for idx in df_styles.index:
# 	# 	poster_id = df_styles.loc[idx, 'Poster_id']
# 	# 	style = df_styles.loc[idx, 'Strategy']
# 	# 	media_styles[poster_id] = style
# 	# 	print(f'\t{idx+1}. Poster {poster_id}: {style}.')
# 	# print('#'*35)

# 	# """==========================================
# 	# 	2. Evaluate Model on different test styles
# 	# =========================================="""
# 	# # Poster sets by styles
# 	# posters_neutral = [
# 	# 	poster for poster in media_styles \
# 	# 	if media_styles[poster] == "Informational / Educational / Neutral"
# 	# ]
# 	# posters_efficacy = [
# 	# 	poster for poster in media_styles \
# 	# 	if media_styles[poster] == "Self-Efficacy"
# 	# ]
# 	# posters_threaten = [
# 	# 	poster for poster in media_styles \
# 	# 	if media_styles[poster] == "Threatening / Fear-driven"
# 	# ]
# 	# df = pd.read_csv(os.path.expandvars(cfgs["data_path"]))
# 	# assert len(posters_neutral) + len(posters_efficacy) + len(posters_threaten) == len(df["Poster_id"].unique())
# 	# # # Poster sets by styles in dict
# 	# # dict_test_posters = {}

# 	# #################################################
# 	# # evaluate models on different poster sets
# 	# #################################################
# 	# # ablation study for model naming 
# 	# if not visual_stimuli:
# 	# 	cfgs["vision"] = False 	# in eval_model(): append "-novision" to model name
# 	# if not trait:
# 	# 	cfgs["trait"] = False 	# in eval_model(): append "-notrait" to model name

# 	# ### 1. eval on "neutral" style ###
# 	# eval_model(cfgs, prompts, test_posters=posters_neutral)
# 	# ### 2. eval on "efficacy" style ###
# 	# eval_model(cfgs, prompts, test_posters=posters_efficacy)
# 	# ### 3. eval on "threaten" style ###
# 	# eval_model(cfgs, prompts, test_posters=posters_threaten)


