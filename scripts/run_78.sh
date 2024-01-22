#!/bin/bash

# FILEPATH: /home/aryan/quick_experiments/act_add/contrast_pairs_experiment/scripts/run_123.sh

# Activate the virtual environment
source venv/bin/activate

# Run the Python script
python mcq_experiments.py --subfolder 8 --run_str bird2 --n_examples 7 8 --samples 30 30 --seed_start 400 --oneprompt