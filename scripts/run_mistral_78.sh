#!/bin/bash

# FILEPATH: /home/aryan/quick_experiments/act_add/contrast_pairs_experiment/scripts/run_123.sh

# Activate the virtual environment
source venv/bin/activate

# Run the Python script
python mcq_experiments_mistral.py --test_filename "preprocessed_data/pure_casing_test_dataset_2.json" --examples_filename "preprocessed_data/pure_casing_examples_full.json" --subfolder mistral_test2 --run_str mistralfullset2_78 --n_examples 7 8  --samples 20 20 --seed_start 1000 --oneprompt