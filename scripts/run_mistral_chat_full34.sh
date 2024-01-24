#!/bin/bash

# FILEPATH: /home/aryan/quick_experiments/act_add/contrast_pairs_experiment/scripts/run_123.sh

# Activate the virtual environment
source venv/bin/activate

# Run the Python script
python mcq_experiments_mistral.py --test_filename "preprocessed_data/pure_casing_test_dataset.json" --examples_filename "preprocessed_data/pure_casing_examples_full.json" --model "mistralai/Mistral-7B-Instruct-v0.1" --subfolder mistral_chat_full --run_str mistral_chat_full1 --n_examples 3 4 --samples 40 40 --seed_start 1000

python mcq_experiments_mistral.py --test_filename "preprocessed_data/pure_casing_test_dataset.json" --examples_filename "preprocessed_data/pure_casing_examples_full.json" --model "mistralai/Mistral-7B-Instruct-v0.1" --subfolder mistral_chat_full --run_str mistral_chat_full_oneprompt1 --n_examples 3 4 --samples 40 40 --seed_start 1000 --oneprompt