#!/bin/bash

# FILEPATH: /home/aryan/quick_experiments/act_add/contrast_pairs_experiment/scripts/run_123.sh

# Activate the virtual environment
source venv/bin/activate

# Run the Python script
python mcq_experiments_mistral.py --test_filename "preprocessed_data/pure_casing_test_dataset.json" --examples_filename "preprocessed_data/pure_casing_examples_full.json" --subfolder mistral_cheat --run_str mistral_cheat --n_examples 0 1 2 3 4 5 --samples 1 10 10 10 10 10 --seed_start 1000 --oneprompt --sysprompt "Please always choose the answer choice corresponding to the uppercase answer"