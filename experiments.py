import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional, Dict
from prompting_from_scratch import zero_shot, few_shot
from claude_scoring import evaluate_json
import os


def extract_qa_pair(example: Dict[str, str]):
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

def reformat_test_dataset(test_filename, new_filename):
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


def run_tests(model, tokenizer, test_filename, examples: List[Tuple[str, str]]):

    # test zero-shot model for a baseline
    zero_shot_output_file = os.path.join("results", f"zero_shot_output.json")
    print(f'Testing zero_shot on {test_filename}')
    zero_shot(model, tokenizer, test_filename, zero_shot_output_file)

    print(f'Evaluating results in {zero_shot_output_file}')
    zero_shot_scored_data = evaluate_json(zero_shot_output_file, question_key='question', answer_key='response')
    with open(os.path.join("results", "zero_shot_scored.json"), "w") as f:
        json.dump(zero_shot_scored_data, f, indent=4)

    # test few-shot model
    few_shot_output_file = os.path.join("results", f"few_shot_output.json")
    # examples = [(example['question'], example['response']) for example in examples]
    print(f'Testing few_shot on {test_filename}')
    few_shot(model, tokenizer, test_filename, few_shot_output_file, examples) # type: ignore

    print(f'Evaluating results in {zero_shot_output_file}')
    few_shot_scored_data = evaluate_json(few_shot_output_file, question_key='question', answer_key='response')
    with open(os.path.join("results", "few_shot_scored.json"), "w") as f:
        json.dump(few_shot_scored_data, f, indent=4)
    
    # make sure the evaluator model had mostly valid scores:
    zero_shot_invalid_scores = sum([1 if item['score'] == 'Claude API error' else 0 for item in zero_shot_scored_data]) / len(zero_shot_scored_data)
    few_shot_invalid_scores = sum([1 if item['score'] == 'Claude API error' else 0 for item in few_shot_scored_data]) / len(few_shot_scored_data)
    print('zero-shot invalid scores:', zero_shot_invalid_scores)
    print('few-shot invalid scores:', few_shot_invalid_scores)
    
    zero_shot_accuracy = sum([1 if item['score'] == '1.0' else 0 for item in zero_shot_scored_data]) / len(zero_shot_scored_data)
    few_shot_accuracy = sum([1 if item['score'] == '1.0' else 0 for item in few_shot_scored_data]) / len(few_shot_scored_data)

    print('zero-shot accuracy:', zero_shot_accuracy)
    print('few-shot accuracy:', few_shot_accuracy)

    return zero_shot_accuracy, few_shot_accuracy










def main():
    # test on the test set

    test_filename = os.path.join("preprocessed_data", "test_dataset_small.json")
    examples_filename = os.path.join("preprocessed_data", "examples.json")

    print(f"Using examples from {examples_filename}")
    examples = format_examples(examples_filename)

    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

    run_tests(model, tokenizer, test_filename, examples)
