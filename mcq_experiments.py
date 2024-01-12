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
DEVICE = 'cuda:4' if t.cuda.is_available() else 'cpu'

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
TEMPERATURE = 1
DEVICE = 'cuda:4' if t.cuda.is_available() else 'cpu'


def tokenize_llama_chat(
    tokenizer,
    system_prompt: str,
    conversation: List[Tuple[str, Optional[str]]],
    no_final_eos=False,
) -> List[int]:
    """
    tokenizer: a HuggingFace tokenizer
    system_prompt: the system prompt to use for the conversation
    conversation: a list of tuples of the form (user_input, model_output) - model_output can be None if there is no response yet
    no_final_eos: if True, the final </s> token will not be added to the end of the conversation when there is a model response

    Returns: a list of tokens
    """

    def _instruction_response_to_tokens(
        instruction, model_output=None, is_first_message=False, no_eos=False
    ):
        if is_first_message:
            dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
        else:
            dialog_content = instruction.strip()
        if model_output is not None:
            if no_eos:
                return tokenizer.encode(
                    f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
                )
            return tokenizer.encode(
                f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()} {tokenizer.eos_token}"
            )
        else:
            return tokenizer.encode(f"{B_INST} {dialog_content.strip()} {E_INST}")

    tokens = []
    for i, (user_input, model_output) in enumerate(conversation):
        tokens += _instruction_response_to_tokens(
            user_input,
            model_output,
            i == 0,
            no_final_eos and (i == len(conversation) - 1),
        )
    return tokens

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

def format_examples_mcq(examples_filename, flipped=False):
    with open(examples_filename, "r") as f:
        examples = json.load(f)
    
    choice_str = "answer_matching_behavior" if flipped else "answer_not_matching_behavior"
    examples = [(example['question'].strip(), f"My answer is: {example[choice_str].strip()}") for example in examples]
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

def run_tests_zero_shot(model, tokenizer, test_filename, subfolder = "", run_str = ""):
    # test zero-shot model for a baseline
    zero_shot_output_file = os.path.join("results/"+subfolder, f"zero_shot_output_{run_str}.json")
    print(f'Testing zero_shot on {test_filename}')
    # Load the questions from the input JSON file
    with open(test_filename, "r") as file:
        questions = json.load(file)

    zero_shot_scored_data = []
    for question in tqdm(questions):
        conversation = [(str(question['question']), 'My answer is: (')]
        input_ids = tokenize_llama_chat(tokenizer, SYSTEM_PROMPT, conversation, no_final_eos=True) # type: ignore
        input_ids_tensor = t.tensor([input_ids]).to(DEVICE)
        output = model.generate(input_ids_tensor, max_new_tokens=1, temperature=1, output_scores=True, return_dict_in_generate=True)
        probs = t.softmax(output['scores'][0][0], dim=0).cpu().detach()
        index = tokenizer.encode(question['answer_matching_behavior'][:-1])[-1]
        score = probs[index].item()
        zero_shot_scored_data.append(score)
        question['score'] = score
    
    with open(zero_shot_output_file, "w") as file:
        json.dump(questions, file, indent=4)

    zero_shot_score = sum(zero_shot_scored_data)/len(zero_shot_scored_data)
    print('zero-shot score:', zero_shot_score)

    return zero_shot_score


def run_tests_few_shot(model, tokenizer, test_filename, examples: List[Tuple[str, str]], subfolder = "", run_str = ""):

    # test few-shot model for a baseline
    few_shot_output_file = os.path.join("results/"+subfolder, f"few_shot_output_{run_str}.json")
    print(f'Testing few_shot on {test_filename}')
    # Load the questions from the input JSON file
    with open(test_filename, "r") as file:
        questions = json.load(file)

    few_shot_scored_data = []
    for question in tqdm(questions):
        conversation = examples + [(str(question['question']), 'My answer is: (')]
        input_ids = tokenize_llama_chat(tokenizer, SYSTEM_PROMPT, conversation, no_final_eos=True) # type: ignore
        input_ids_tensor = t.tensor([input_ids]).to(DEVICE)
        output = model.generate(input_ids_tensor, max_new_tokens=1, temperature=1, output_scores=True, return_dict_in_generate=True)
        probs = t.softmax(output['scores'][0][0], dim=0).cpu().detach()
        index = tokenizer.encode(question['answer_matching_behavior'][:-1])[-1]
        score = probs[index].item()
        few_shot_scored_data.append(score)
        question['score'] = score
    
    with open(few_shot_output_file, "w") as file:
        json.dump(questions, file, indent=4)

    few_shot_score = sum(few_shot_scored_data)/len(few_shot_scored_data)
    print('few-shot score:', few_shot_score)

    return few_shot_score


def run_tests(model, tokenizer, test_filename, examples: List[Tuple[str, str]], subfolder = "", run_str = ""):

    zero_shot_score = run_tests_zero_shot(model, tokenizer, test_filename, subfolder, run_str)
    few_shot_score = run_tests_few_shot(model, tokenizer, test_filename, examples, subfolder, run_str)

    return zero_shot_score, few_shot_score


def main():

    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf').to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

    # test on the test set
    test_filename = os.path.join("preprocessed_data", "mcq_test_dataset_small.json")
    examples_filename = os.path.join("preprocessed_data", "mcq_examples.json")

    print(f"Using examples from {examples_filename}")
    examples = format_examples_new(examples_filename)


    run_tests(model, tokenizer, test_filename, examples, subfolder = "mcq_results/1", run_str = "")


# %%
print("Loading model")
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf').to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

# %%
# test on the test set
test_filename = os.path.join("preprocessed_data", "mcq_test_dataset_small.json")
examples_filename = os.path.join("preprocessed_data", "mcq_examples.json")

print(f"Using examples from {examples_filename}")
examples = format_examples_new(examples_filename)

# %%

run_tests(model, tokenizer, test_filename, examples, subfolder = "mcq_results/1", run_str = "")

# %%












# %%
if __name__ == "__main__":
    main()