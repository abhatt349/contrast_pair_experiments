#!/bin/bash

# FILEPATH: /home/aryan/quick_experiments/act_add/contrast_pairs_experiment/scripts/run_123.sh

# Activate the virtual environment
source venv/bin/activate

# Run the Python script
python mcq_experiments_mistral.py --test_filename "preprocessed_data/pure_casing_flipped_test_dataset.json" --examples_filename "preprocessed_data/pure_casing_flipped_examples_full.json" --subfolder mistral_flipped --run_str mistralflippedset_012 --n_examples 0 1 2 --samples 1 30 30 --seed_start 1000 --oneprompt