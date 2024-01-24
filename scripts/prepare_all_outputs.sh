#!/bin/bash

# FILEPATH: /home/aryan/quick_experiments/act_add/contrast_pairs_experiment/scripts/prepare_all_outputs.sh

# Activate the virtual environment
source venv/bin/activate

# Make the directory ./results if it doesn't exist
mkdir -p ./results/tmp

# ---------------------Sycophancy---------------------
#            ----------Llama-2-7b-chat----------
# standard
python mcq_experiments_mistral.py --test_filename "preprocessed_data/mcq_test_dataset.json" --examples_filename "preprocessed_data/mcq_examples_full.json" --model "meta-llama/Llama-2-7b-chat-hf" --subfolder llama_chat_sycophancy --run_str llama_chat_sycophancy --n_examples 0 1 2 3 4 5 6 7 8 9 10 11 12  --samples 1 60 60 50 50 40 40 40 40 30 30 30 30 --seed_start 1832

# oneprompt
python mcq_experiments_mistral.py --test_filename "preprocessed_data/mcq_test_dataset.json" --examples_filename "preprocessed_data/mcq_examples_full.json" --model "meta-llama/Llama-2-7b-chat-hf" --subfolder llama_chat_sycophancy --run_str llama_chat_sycophancy_op --n_examples 0 1 2 3 4 5 6 7 8 9 10 11 12  --samples 1 60 60 50 50 40 40 40 40 30 30 30 30 --seed_start 1832 --oneprompt

# ---------------------Uppercase+Sycophancy---------------------
#            ----------Llama-2-7b-chat----------
# 0-shot
python mcq_experiments_mistral.py --test_filename "preprocessed_data/casing_test_dataset.json" --examples_filename "preprocessed_data/casing_examples_full.json" --model "meta-llama/Llama-2-7b-chat-hf" --subfolder llama_chat_U+S --run_str llama_chat_U+S_0shot --n_examples 0 --samples 1 --seed_start 1832

# standard
python mcq_experiments_mistral.py --test_filename "preprocessed_data/casing_test_dataset.json" --examples_filename "preprocessed_data/casing_examples_full.json" --model "meta-llama/Llama-2-7b-chat-hf" --subfolder llama_chat_U+S --run_str llama_chat_U+S --n_examples 1 2 3 4 5 6 7 8  --samples 40 --seed_start 1832

# oneprompt
python mcq_experiments_mistral.py --test_filename "preprocessed_data/casing_test_dataset.json" --examples_filename "preprocessed_data/casing_examples_full.json" --model "meta-llama/Llama-2-7b-chat-hf" --subfolder llama_chat_U+S --run_str llama_chat_U+S_op --n_examples 1 2 3 4 5 6 7 8  --samples 40 --seed_start 1832 --oneprompt

# ---------------------Pure Uppercase---------------------
#            ----------Mistral-7b----------
python mcq_experiments_mistral.py --test_filename "preprocessed_data/pure_casing_test_dataset.json" --examples_filename "preprocessed_data/pure_casing_examples_full.json" --model "mistralai/Mistral-7B-v0.1" --subfolder mistral_PU --run_str mistral_PU_0shot --n_examples 0 --samples 1 --seed_start 1832

python mcq_experiments_mistral.py --test_filename "preprocessed_data/pure_casing_test_dataset.json" --examples_filename "preprocessed_data/pure_casing_examples_full.json" --model "mistralai/Mistral-7B-v0.1" --subfolder mistral_PU --run_str mistral_PU_op --n_examples 1 2 3 4 5 6 7 8 --samples 40 --seed_start 1832 --oneprompt

#          ----------Mistral-7b-Instruct----------
python mcq_experiments_mistral.py --test_filename "preprocessed_data/pure_casing_test_dataset.json" --examples_filename "preprocessed_data/pure_casing_examples_full.json" --model "mistralai/Mistral-7B-Instruct-v0.1" --subfolder mistral_chat_PU --run_str mistral_chat_PU_0shot --n_examples 0 --samples 1 --seed_start 1832

python mcq_experiments_mistral.py --test_filename "preprocessed_data/pure_casing_test_dataset.json" --examples_filename "preprocessed_data/pure_casing_examples_full.json" --model "mistralai/Mistral-7B-Instruct-v0.1" --subfolder mistral_chat_PU --run_str mistral_chat_PU --n_examples 1 2 3 4 5 6 7 8 --samples 40 --seed_start 1832

python mcq_experiments_mistral.py --test_filename "preprocessed_data/pure_casing_test_dataset.json" --examples_filename "preprocessed_data/pure_casing_examples_full.json" --model "mistralai/Mistral-7B-Instruct-v0.1" --subfolder mistral_chat_PU --run_str mistral_chat_PU_op --n_examples 1 2 3 4 5 6 7 8 --samples 40 --seed_start 1832 --oneprompt