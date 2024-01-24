#!/bin/bash

# FILEPATH: /home/aryan/quick_experiments/act_add/contrast_pairs_experiment/scripts/run_123.sh

# Activate the virtual environment
source venv/bin/activate

# Run the Python script
python mcq_experiments.py --subfolder casing --test_filename "preprocessed_data/casing_test_dataset.json" --examples_filename "preprocessed_data/casing_examples_full.json" --run_str sanity8 --n_examples 8 --samples 40 --seed_start 400