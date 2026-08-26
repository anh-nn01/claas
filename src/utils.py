import os
import base64
from io import BytesIO
import yaml # type: ignore
from PIL import Image # type: ignore
from tqdm import tqdm # type: ignore
import re
import pandas as pd
import numpy as np
np.random.seed(42)


"""======================================================
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
======================================================"""
def convert_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")  # You can change the format if needed
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_b64 

"""======================================================
        Load config files
======================================================"""
def load_config(filepath):
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{filepath}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{filepath}': {e}")
        return None
    


""" ======================================================
    Convert individual-poster pair row to LLM conversation format
    
    Args:
        + sample: a row of individual-poster response
    return: conversation format
======================================================"""
def convert_to_conversation(sample, use_image=True):
    poster_id = sample["image_name"]
    poster_path = os.path.abspath(f"../stimuli/{poster_id}.png")

    if use_image:
        # Multimodal Vision-Language Model
        image = Image.open(poster_path)
        conversation = [
            { "role": "user",
            "content" : [
                {"type" : "text",  "text"  : sample["instruction"]},
                {"type" : "image", "image" : image} ]
            },
            { "role" : "assistant",
            "content" : [
                {"type" : "text",  "text"  : str(sample["answer"])} ]
            },
        ]
    else:
        # Text-only Language Model
        conversation = [
            { "role": "user", "content" : sample["instruction"]},
            { "role" : "assistant", "content" : str(sample["answer"])},
        ]
    return { "messages" : conversation }

"""
    Convert dataframe to SFT-compatible trainer dataset

    Args:
        + dataframe: original dataframe
    return: SFT Trainer-compatible dataset
"""
def convert_to_model_trainer_dataset(dataframe, use_image=True):
    trainer_dataset = []
    for idx in tqdm(dataframe.index, desc="dataset conversion"):
        sample = dataframe.loc[idx]
        trainer_dataset.append(convert_to_conversation(sample, use_image))

    return trainer_dataset


"""
    Process and filter persona components (demographics, personality scores, and locus)
    based on user selection flags.

    Args:
        demo_info (str): Raw demographic information text.
        persona_score (str): Text block containing personality trait information.
        locus (str): Text or variable for locus of control information.
        demo_full (bool): If False, only include selected demographic fields.
        include_big5 (bool): Whether to include Big-Five personality scores.
        include_facet (bool): Whether to include facet-level trait scores.
        include_locus (bool): Whether to include locus information.
        train_mode (bool): 
            True, use training mode processing: use sensitive traits only.
            False, use evaluation mode processing: include all specified traits.

    Returns:
        tuple: (demo_info, persona_score, locus)
"""
def process_trait_info(
        demo_info, persona_score, locus,
        include_demo=True, demo_full=True, 
        include_big5=True,
        include_facet=True, 
        include_locus=True,
        train_mode=True,
    ):

    # ==============================
    # 1. Demographic filtering
    # ==============================
    if (demo_full is not None) and (not demo_full):
        fields = [
            "Gender", "Age", "Current Profession", "Race/Ethnicity",
            "Religious/Cultural Group", "Political Affiliation",
            "Highest Education", "Annual Household Income", "Family Status"
        ]

        # """ ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # [SPH Request] Most sensitive to ["Gender", "Age", and "Race/Ethnicity".]
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
        # if train_mode:
        #     # 90% of the time, only include the 3 most sensitive fields
        #     fields_2 = ["Gender", "Age", "Race/Ethnicity", "Current Profession", "Annual Household Income",]
        #     # use_top3_sensitive = True if (np.random.rand() <= 0.9) else False
        #     use_top5_sensitive = True if (np.random.rand() <= 0.9) else False
        # else:
        #     # include all user-specified fields during evaluation
        #     # use_top3_sensitive = False
        #     use_top5_sensitive = False


        pattern = r"^(.*?):\s*(.*)$"
        if (not include_demo) or pd.isna(demo_info) or demo_info is None:
            demo_info = ""
            
        demo_info = demo_info.replace("Demographics:", "").replace("Demographics", "")
        matches = re.findall(pattern, demo_info, flags=re.MULTILINE)

        demo_dict = {k.strip(): v.strip() for k, v in matches if (k.strip().replace(':', '') in fields)}

        demo_info = "Demographics:\n"
        for k in fields:
            # demo trait must be available
            if k in demo_dict:
                demo_info += f"{k}: {demo_dict[k]}\n"
            else:
                demo_info += f"{k}: [Not specified]\n"
            # # we are not using the top 3 sensitive filtering OR trait k must be in top-3 sensitive
            # trait_k_included = (not use_top5_sensitive) or (k in fields_2)
            # # demo trait must be available + selected
            # if k in demo_dict and trait_k_included:
            #     demo_info += f"{k}: {demo_dict[k]}\n"
            # else:
            #     demo_info += f"{k}: [Not specified]\n"

    # ================================================
    # 2. Personality trait filtering
    #   => if not available, set to [Not specified]
    # ================================================
    if not include_big5 and (persona_score is not None) and (not pd.isna(persona_score)):
        # exclude any big5 traits in the data
        persona_score = re.sub(
            r"Big-Five Trait Scores:.*?(?=(Facet Scores:|Demographics:|$))",
            "",
            persona_score,
            flags=re.DOTALL
        )
    else:
        # include big5 trait
        if (persona_score is None) or (pd.isna(persona_score)):
            persona_score = ""
        if "Big-Five Trait Scores" not in persona_score:
            persona_score = "\nBig-Five Trait Scores: [Not specified]\n" + persona_score

    if not include_facet and (persona_score is not None) and (not pd.isna(persona_score)):
        # exclude any facet traits in the data
        persona_score = re.sub(
            r"Facet Scores:.*?(?=(Demographics:|$))",
            "",
            persona_score,
            flags=re.DOTALL
        )
    else:
        # include facet trait
        if (persona_score is None) or (pd.isna(persona_score)):
            persona_score = ""
        if "Facet Scores" not in persona_score:
            persona_score = persona_score + "\nFacet Scores: [Not specified]\n"

    # ==============================
    # 3. Locus inclusion
    # ==============================
    if not include_locus:
        locus = None

    return demo_info, persona_score, locus