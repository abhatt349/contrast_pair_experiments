# %%
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional, Dict
import open_ended_prompting as open_ended_prompting
from open_ended_prompting import zero_shot, few_shot
from claude_scoring import evaluate_json
import os
import torch as t
import importlib
from tqdm import tqdm
import time
import random
import psutil
import GPUtil
import sys
import matplotlib.pyplot as plt
from einops import reduce, repeat, rearrange
import argparse
# %%
def convert_test_set_to_casing(test_filename, new_filename):
    with open(test_filename, 'r') as f:
        test = json.load(f)
    for example in test:
        question, remainder = example['question'].split('(A)')
        response_a, response_b = remainder.split('(B)')
        if example['answer_matching_behavior'] == '(A)':
            example['question'] = f"{question}(A){response_a.upper()}(B){response_b.lower()}"
        else:
            example['question'] = f"{question}(A){response_a.lower()}(B){response_b.upper()}"
    with open(new_filename, 'w') as f:
        json.dump(test, f, indent=4)
    
def convert_test_set_to_pure_casing(test_filename, new_filename):
    with open(test_filename, 'r') as f:
        test = json.load(f)
    for example in test:
        question, remainder = example['question'].split('(A)')
        response_a, response_b = remainder.split('(B)')
        response = random.choice([response_a, response_b])
        if example['answer_matching_behavior'] == '(A)':
            example['question'] = f"{question}(A){response.upper()}(B){response.lower()}"
        else:
            example['question'] = f"{question}(A){response.lower()}(B){response.upper()}"
    with open(new_filename, 'w') as f:
        json.dump(test, f, indent=4)
    
# %%

convert_test_set_to_casing('preprocessed_data/mcq_test_dataset.json', 'preprocessed_data/casing_test_dataset.json')
convert_test_set_to_casing('preprocessed_data/mcq_examples_full.json', 'preprocessed_data/casing_examples_full.json')

# %%

convert_test_set_to_pure_casing('preprocessed_data/mcq_test_dataset.json', 'preprocessed_data/pure_casing_test_dataset.json')
convert_test_set_to_pure_casing('preprocessed_data/mcq_examples_full.json', 'preprocessed_data/pure_casing_examples_full.json')

# %%

convert_test_set_to_pure_casing('preprocessed_data/mcq_test_dataset_2.json', 'preprocessed_data/pure_casing_test_dataset_2.json')

# %%
