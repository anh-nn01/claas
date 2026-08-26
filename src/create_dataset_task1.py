"""
	Create Vision-Language Tuning dataset (train/test) for 
	"Trait-aware Response Prediction; Generalize on Unseen Communication Strategies"

	Train/Test: split by visual stimuli / media style
				(Thretening, Self-efficacy, Informational/Neutral)
	=> split by Communication Strategy
	=> generalize to unseen Community Strategy

	v3a:
		train : ['threatening', 'self-efficacy'] 
		test  : ['informational / neutral']
	v3b:
		train : ['threatening', 'informational / neutral'] 
		test  : ['self-efficacy']
	v3c:
		train : ['self-efficacy', 'informational / neutral'] 
		test  : ['threatening']
"""

import os
import re
import yaml # type: ignore
from PIL import Image # type: ignore
import pandas as pd # type: ignore

import io # type: ignore
import random
random.seed(1)
import pyarrow as pa # type: ignore
import pyarrow.parquet as pq # type: ignore

from utils import convert_to_base64, load_config, process_trait_info # type: ignore
from tqdm import tqdm   # type: ignore
from termcolor import colored
import argparse

"""=========================================================
	Create instruction based on sample data & prompts

	Args;
		+ sample: data points with personality/demographic data & MCQ Q&A
		+ prompts: prompt cfg template
		+ demo_full: 
			True: use full available dempgraphic traits
			False: only use (
				Gender, Age, Current Profession, 
				Race/Ethnicity, Religious/Cultural Group, Political Affiliation,
				Highest Education, Annual Household Income, Family Status
			)
		+ big5: 
			True: include big5 traits
			False: exclude big5 traits
		+ facet:
			True: include facet score traits
			False: exclude facet score traits
		+ locus:
			True: include locus of control traits
			Fals: exclude locus of control traits

	return:
		+ all training instructions for current sample
		  (dictionary of instruction for all survey questions)
========================================================="""
def create_instruction(
		sample, prompts, 
		include_demo=True, demo_full=True, 
		include_big5=True, 
		include_facet=True, 
		include_locus=True
	):
	INSTRUCTION_DICT = {}

	###########################################
	###    Instruction prompt construction  ###
	###########################################
	# 1. Personality & Demographic information
	demo_info = sample["Demographic"] # demographic information
	demo_info = "" if pd.isna(demo_info) else demo_info
	persona_score = sample["Persona_score"] # personality scores
	# persona_qa = sample["Persona_QA"] # personality survey questions & answers
	locus = sample["Locus"] # locus of control survey
	# ************************************************
	# return None if 
	#   demo_info and persona_score are both None
	# ************************************************
	if demo_info == "" and (persona_score is None or pd.isna(persona_score)):
		return None
	
	######################################################################################
	# 1*. modify trait info based on trait selection setings
	#       demo_full: wheter include full demographic traits or only selected ones
	#       include_big5, include_facet, include_locus: include big5 / facet / locus of control traits or not
	#     format: <trait>: <value> if available; else <trait>: [Not specified]
	######################################################################################
	demo_info, persona_score, locus = process_trait_info(
		demo_info, persona_score, locus,
		include_demo, demo_full, 
		include_big5, 
		include_facet, 
		include_locus
	)

	# 2. Personality emulation prompt
	#   (a) system prompt
	SYSTEM_PROMPT = prompts["SYSTEM_SIM"]
	#   (b) information for role-playing
	SIM_PROMPT = ""
	if include_demo:
		SIM_PROMPT += f"You are: {demo_info}\n"
	# if at least 1 personality trait is included
	if include_big5 or include_facet or include_locus:
		SIM_PROMPT += f"Your personality test shows you have (min score = 0; max score = 5): {persona_score}\n"
	# SIM_PROMPT += f"Your answers to the personality tests was: {persona_qa}\n"
	if locus is not None:
		SIM_PROMPT += f"You also have {locus}\n"
	#   (c) situation description
	SIM_PROMPT += prompts["SIMULATION_SIM"]
	#   (d) questions
	# SIM_PROMPT += "The survey question is:\n"
	# loop through questions
	for i in range(1,16,1):
		# Instruction prompt
		INSTRUCTION = SYSTEM_PROMPT + SIM_PROMPT
		# specific survey question
		q_num = f"Q{i}"
		if not pd.isna(sample[q_num]):
			question = sample[q_num].replace("\n", " ")
			# SIM_PROMPT += f"Q{i}: {question}\n" # specific question
			USER_PROMPT = f"Question: {question}\n" # specific question
			# instruction prompt to answer in proper format
			if "type in" in question.lower():
				# USER_PROMPT += prompts['INSTRUCTION_FREE']
				continue # ignore free-text questions
			else:
				USER_PROMPT += prompts['INSTRUCTION_MCQ']
		else:
			continue
		# instruction for current question
		INSTRUCTION += USER_PROMPT
		INSTRUCTION_DICT[q_num] = INSTRUCTION
	# #   (e) instruction
	# SIM_PROMPT += (
	#     "Your personality, locus of control, and demographic traits influence your emotions/reactions. "
	#     "Answer the survey questions authentically, as if you are completing a real online survey. "
	# )

	# 3. Instruction prompt
	# INSTRUCTION = SYSTEM_PROMPT + '' + SIM_PROMPT

	return INSTRUCTION_DICT


"""=========================================================
	Create MCQ answers based on sample data

	Args;
		+ sample: data points with personality/demographic data & MCQ Q&A
	return:
		+ all training answers for current sample
		(dictionary of instruction for all survey questions)
========================================================="""
def create_answers(sample):
	ANS_DICT = {}
	# loop through questions
	for i in range(1,16,1):
		# specific survey question
		q_num, a_num = f"Q{i}", f"A{i}"
		
		# ignore free-response survey questions
		if (pd.isna(sample[q_num])) or ("type in" in sample[q_num].lower()):
			continue

		# only include MCQs
		try:
			# MCQs: answer must be integer value
			answer = int(sample[a_num])
			ANS_DICT[q_num] = answer
		except:
			pass # ignore free responses

	# return str(ANS_DICT)#.replace('\'', '\"')
	return ANS_DICT



def parse_args():
	parser = argparse.ArgumentParser(description="Dataset Generation Configuration")
	
	# Use 'store_true' for boolean flags. 
	# If the flag is present, it becomes True. If absent, it stays False.
	parser.add_argument('--include_demo', action='store_true', default=False, help="Include demographic traits; override demo_full")
	parser.add_argument('--demo_full', action='store_true', default=False, help="Use all demographic traits")
	parser.add_argument('--include_big5', action='store_true', default=False, help="Include Big5 personality traits")
	parser.add_argument('--include_facet', action='store_true', default=False, help="Include facet-level traits")
	parser.add_argument('--include_locus', action='store_true', default=False, help="Include locus of control")
	
	return parser.parse_args()


"""
	Save the training dataset in the following format:
		+ Column 1: "image_name" => health poster as visual stimuli
		+ Column 2: "instruction => emulation prompt + personality/demographic information + MCQ
		+ Column 3: "answer"     => true answer for MCQ
		+ Column 4: "set"        => ["train", "test"]
"""
if __name__ == "__main__":
	######################################
	# 0. survey data path and prompt settings
	######################################
	# data_path = '../data/survey_responses.csv'
	data_path = '../data/survey_responses_screened.csv'
	style_path = '../data/media_style_posters.csv'
	prompt_cfg = './configs/prompts.yaml'
	prompts = load_config(prompt_cfg)
	SYSTEM_PROMPT = prompts["SYSTEM_SIM"]
	######################################
	# * which trait to be included
	######################################
	args = parse_args()
	include_demo = args.include_demo
	demo_full = args.demo_full  # default: True; false -> only selected demographic traits
	include_big5 = args.include_big5        # default: True
	include_facet = args.include_facet      # default: True
	include_locus = args.include_locus      # default: True

	######################################
	# 1a. Load survey data
	######################################
	data_path = os.path.abspath(data_path)
	df = pd.read_csv(data_path)
	######################################
	# 1b. Load media's communication style classification
	######################################
	style_path = os.path.abspath(style_path)
	df_styles = pd.read_csv(style_path)
	media_styles = {}
	for idx in df_styles.index:
		poster_id = df_styles.loc[idx, 'Poster_id']
		style = df_styles.loc[idx, 'Strategy']
		media_styles[poster_id] = style
		print(f'\tPoster {poster_id}: {style}.')

	######################################
	###     define train/test splits   ###
	######################################
	# test: "neutral"
	train_styles_1 = ['Self-Efficacy', 'Threatening / Fear-driven']
	# test: "efficacy"
	train_styles_2 = ['Informational / Educational / Neutral', 'Threatening / Fear-driven']     
	# test: "threatening"  
	train_styles_3 = ['Self-Efficacy', 'Informational / Educational / Neutral']  
	# test: "" => train on all styles (deploy)
	train_styles_all = ['Self-Efficacy', 'Informational / Educational / Neutral', 'Threatening / Fear-driven']
	
	""" =========================================================
			loop through 3 different train/test splits
	========================================================="""
	for train_styles in [train_styles_1, train_styles_2, train_styles_3, train_styles_all, train_styles_all]:

		######################################
		# create train/test dataset paths 
		# based on current split
		######################################
		if 'Self-Efficacy' not in train_styles:
			assert 'Threatening / Fear-driven' in train_styles and 'Informational / Educational / Neutral' in train_styles
			test_style = 'efficacy'
		elif 'Threatening / Fear-driven' not in train_styles:
			assert 'Self-Efficacy' in train_styles and 'Informational / Educational / Neutral' in train_styles
			test_style = 'threatening'
		elif 'Informational / Educational / Neutral' not in train_styles:
			assert 'Self-Efficacy' in train_styles and 'Threatening / Fear-driven' in train_styles
			test_style = 'neutral'
		else:
			print("All styles for training.")
			test_style = 'train_on_all' # deploy on all styles
			# raise ValueError(f"Undefined test style for {train_styles}.")

		#################################################
		# define dataset paths based on current split
		#################################################
		if test_style == 'train_on_all':
			train_it_path = os.path.abspath(f"../data/task1_it_train_all_styles")
			test_it_path = os.path.abspath(f"../data/task1_it_test_all_styles")
		else:
			train_it_path = os.path.abspath(f"../data/task1_it_train_holdout_{test_style}")
			test_it_path = os.path.abspath(f"../data/task1_it_test_holdout_{test_style}")
		
		if not include_demo:
			# ablation: no demographic traits
			train_it_path += "_noDemo"
			test_it_path += "_noDemo"
		if include_demo and (not demo_full):
			# ablation: partial demographic traits
			train_it_path += "_partialDemo"
			test_it_path += "_partialDemo"
		if not include_big5:
			# ablation: no big5 traits
			train_it_path += "_noBig5"
			test_it_path += "_noBig5"
		if not include_facet:
			# ablation: no facet traits
			train_it_path += "_noFacet"
			test_it_path += "_noFacet"
		if not include_locus:
			# ablation: no locus of control traits
			train_it_path += "_noLocus"
			test_it_path += "_noLocus"
		
		train_it_path += ".csv"
		test_it_path += ".csv"
		print(colored(f'TASK 1: Creating train/test sets ...', 'green'))
		print(colored(f"Testing on {test_style.upper()} posters.", "green"))

		######################################
		# 2. Split train/test set based on poster styles
		######################################
		poster_ids = df['Poster_id'].unique()
		train_poster_ids = [pid for pid in poster_ids if media_styles[pid] in train_styles]
		test_poster_ids = [pid for pid in poster_ids if media_styles[pid] not in train_styles]
		print(colored('Traing poster ids:\n', "blue"), train_poster_ids)
		print(colored('\tTraining Total =', "blue"), len(train_poster_ids))
		print(colored('Test poster ids:\n', "blue"), test_poster_ids)
		print(colored('\tTesting Total =', "blue"), len(test_poster_ids))

		######################################
		# 3. Create train/test 
		#    instruction tuning dataset
		######################################
		df_it_train = pd.DataFrame(columns=["image_name", "style", "instruction", "answer", "set"])
		df_it_test = pd.DataFrame(columns=["image_name", "style", "instruction", "answer", "set"])

		for idx in tqdm(df.index[:], desc='Contructing dataset'):
			sample = df.loc[idx]
			# (a) Poster visual media information
			poster_id = sample["Poster_id"] # poster id
			style = media_styles[poster_id]
			# (b) Construct instruction for role-playing & MCQs
			INSTRUCTION_DICT = create_instruction(
				sample, prompts, include_demo, demo_full, include_big5, include_facet, include_locus
			)
			# (b*) skip responses that miss BOTH demo_info and persona_score
			if INSTRUCTION_DICT is None:
				continue
			# (c) Construct MCQs answer format from GT
			ANS_dict = create_answers(sample)
			# (d) Train/test set
			mode = "train" if (poster_id in train_poster_ids) else "test"

			# (e) add data
			new_data = {
				"image_name": [poster_id] * len(ANS_dict), 
				"style": [style] * len(ANS_dict), 
				"instruction": [instruction for instruction in INSTRUCTION_DICT.values()], 
				"answer": [ans for ans in ANS_dict.values()], 
				"set": [mode] * len(ANS_dict)
			}
			if mode == "train":
				df_it_train = pd.concat([df_it_train, pd.DataFrame(new_data)], ignore_index=True)
			elif mode == "test":
				df_it_test = pd.concat([df_it_test, pd.DataFrame(new_data)], ignore_index=True)
			else:
				raise ValueError(f"{mode} set is not defined.")        
			
			# if idx == 0:
			#     # print(instruction["Q7"])
			#     # print(answers["Q7"])
			#     print([ans for ans in ANS_dict.values()])

		df_it_train.to_csv(train_it_path, index=False)
		df_it_test.to_csv(test_it_path, index=False)
		print("\t", "="*50)
		print()






