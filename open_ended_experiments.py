# %%
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional, Dict
import contrast_pairs_experiment.open_ended_prompting as open_ended_prompting
from contrast_pairs_experiment.open_ended_prompting import zero_shot, few_shot
from claude_scoring import evaluate_json
import os
import torch as t
import importlib
DEVICE = 'cuda:4' if t.cuda.is_available() else 'cpu'


def extract_qa_pair(example: Dict[str, str]):
    """
    For the old json format, like {
                                    "question": "[question] (A) [response A] (B) [response B]",
                                    "answer_matching_behavior": "(A)",
                                    "answer_not_matching_behavior": "(B)"
                                  }
    """
    question, remainder = example['question'].split('(A)')
    response_a, response_b = remainder.split('(B)')

    question = question.strip()
    response_a = response_a.strip()
    response_b = response_b.strip()

    if example['answer_matching_behavior'] == '(A)':
        answer = response_a
    else:
        answer = response_b
    
    return (question, answer)

def format_examples(examples_filename):
    with open(examples_filename, "r") as f:
        examples = json.load(f)
    
    examples = [extract_qa_pair(example) for example in examples]
    return examples

def format_examples_new(examples_filename, flipped=False):
    with open(examples_filename, "r") as f:
        examples = json.load(f)
    
    choice_str = "answer_matching_behavior" if flipped else "answer_not_matching_behavior"
    examples = [(example['question'].strip(), example[example[choice_str]].strip()) for example in examples]
    return examples

def reformat_test_dataset(test_filename, new_filename):
    """
    For the old json format, like {
                                    "question": "[question] (A) [response A] (B) [response B]",
                                    "answer_matching_behavior": "(A)",
                                    "answer_not_matching_behavior": "(B)"
                                  }

    Reformat to the new json format, like {
                                            "question": "[question]",
                                            "(A)": "[response A]",
                                            "(B)": "[response B]",
                                            "answer_matching_behavior": "(A)",
                                            "answer_not_matching_behavior": "(B)"
                                        }
    """

    with open(test_filename, "r") as f:
        test_data = json.load(f)

    new_data = []
    for example in test_data:
        question, remainder = example['question'].split('(A)')
        response_a, response_b = remainder.split('(B)')
        new_data.append({
            'question': question.strip(),
            '(A)': response_a.strip(),
            '(B)': response_b.strip(),
            'answer_matching_behavior': example['answer_matching_behavior'],
            'answer_not_matching_behavior': example['answer_not_matching_behavior']
        })

    with open(new_filename, "w") as f:
        json.dump(new_data, f, indent=4)


def run_tests_zero_shot(model, tokenizer, test_filename, run_str = ""):

    # test zero-shot model for a baseline
    zero_shot_output_file = os.path.join("results", f"zero_shot_output_{run_str}.json")
    print(f'Testing zero_shot on {test_filename}')
    zero_shot(model, tokenizer, test_filename, zero_shot_output_file)

    print(f'Evaluating results in {zero_shot_output_file}')
    zero_shot_scored_data = evaluate_json(zero_shot_output_file, question_key='question', answer_key='response')
    with open(os.path.join("results", f"zero_shot_scored_{run_str}.json"), "w") as f:
        json.dump(zero_shot_scored_data, f, indent=4)

    # make sure the evaluator model had mostly valid scores:
    zero_shot_invalid_scores = sum([1 if item['score'] == 'Claude API error' else 0 for item in zero_shot_scored_data]) / len(zero_shot_scored_data)
    print('zero-shot invalid scores:', zero_shot_invalid_scores)
    
    zero_shot_score = sum([item['score'] if isinstance(item['score'], float) else 0 for item in zero_shot_scored_data]) / len(zero_shot_scored_data)

    print('zero-shot score:', zero_shot_score)

    return zero_shot_score


def run_tests_few_shot(model, tokenizer, test_filename, examples: List[Tuple[str, str]], run_str = ""):

    # test few-shot model
    few_shot_output_file = os.path.join("results", f"few_shot_output_{run_str}.json")
    # examples = [(example['question'], example['response']) for example in examples]
    print(f'Testing few_shot on {test_filename}')
    few_shot(model, tokenizer, test_filename, few_shot_output_file, examples) # type: ignore

    print(f'Evaluating results in {few_shot_output_file}')
    few_shot_scored_data = evaluate_json(few_shot_output_file, question_key='question', answer_key='response')
    with open(os.path.join("results", f"few_shot_scored_{run_str}.json"), "w") as f:
        json.dump(few_shot_scored_data, f, indent=4)
    
    # make sure the evaluator model had mostly valid scores:
    few_shot_invalid_scores = sum([1 if item['score'] == 'Claude API error' else 0 for item in few_shot_scored_data]) / len(few_shot_scored_data)
    print('few-shot invalid scores:', few_shot_invalid_scores)
    
    few_shot_score = sum([item['score'] if isinstance(item['score'], float) else 0 for item in few_shot_scored_data]) / len(few_shot_scored_data)

    print('few-shot score:', few_shot_score)

    return few_shot_score


def run_tests(model, tokenizer, test_filename, examples: List[Tuple[str, str]], run_str = ""):

    zero_shot_score = run_tests_zero_shot(model, tokenizer, test_filename, run_str)
    few_shot_score = run_tests_few_shot(model, tokenizer, test_filename, examples, run_str)

    return zero_shot_score, few_shot_score


def main():
    # test on the test set
    test_filename = os.path.join("preprocessed_data", "test_dataset_small.json")
    examples_filename = os.path.join("preprocessed_data", "examples.json")

    print(f"Using examples from {examples_filename}")
    examples = format_examples_new(examples_filename)

    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf').to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

    run_tests(model, tokenizer, test_filename, examples)




# %%
test_filename = os.path.join("preprocessed_data", "test_dataset.json")
examples_filename = os.path.join("preprocessed_data", "examples.json")

examples = format_examples_new(examples_filename)
flipped_examples = format_examples_new(examples_filename, flipped=True)
# %%
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf').to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
# %%

importlib.reload(open_ended_prompting)
run_tests_few_shot(model, tokenizer, test_filename, flipped_examples, run_str = "flipped")
# run_tests(model, tokenizer, test_filename, examples)

# %%

with open('results/zero_shot_scored.json', 'r') as f:
    zero_shot_scored_data = json.load(f)
with open('results/few_shot_scored.json', 'r') as f:
    few_shot_scored_data = json.load(f)

# %%

zero_shot_score = sum([item['score'] if isinstance(item['score'], float) else 0 for item in zero_shot_scored_data]) / len(zero_shot_scored_data)
few_shot_score = sum([item['score'] if isinstance(item['score'], float) else 0 for item in few_shot_scored_data]) / len(few_shot_scored_data)

print('zero-shot score:', zero_shot_score)
print('few-shot score:', few_shot_score)
# %%















# %%
if __name__ == "__main__":
    main()