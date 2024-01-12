import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional
from tqdm import tqdm
import torch as t

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
TEMPERATURE = 0.01
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


def zero_shot(model, tokenizer, input_filename, output_filename):
    # Load the questions from the input JSON file
    with open(input_filename, "r") as file:
        questions = json.load(file)

    # Generate responses for each question
    responses = []
    for question in tqdm(questions):
        conversation = [(str(question['question']), 'My answer is: (')]
        input_ids = tokenize_llama_chat(tokenizer, SYSTEM_PROMPT, conversation, no_final_eos=True) # type: ignore
        input_ids_tensor = t.tensor([input_ids]).to(DEVICE)
        output = model.generate(input_ids_tensor, max_new_tokens=100, temperature=TEMPERATURE)
        response = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[-1].strip()
        responses.append({"question": question["question"], "response": response})

    # Save the responses along with the questions to the output JSON file
    with open(output_filename, "w") as file:
        json.dump(responses, file, indent=4)

def few_shot(model, tokenizer, input_filename, output_filename, examples: List[Tuple[str, Optional[str]]]):
    # Load the questions from the input JSON file
    with open(input_filename, "r") as file:
        questions = json.load(file)

    # Generate responses for each question
    responses = []
    for question in tqdm(questions):
        conversation = examples + [(str(question['question']), 'My answer is: (')]
        input_ids = tokenize_llama_chat(tokenizer, SYSTEM_PROMPT, conversation, no_final_eos=True)
        input_ids_tensor = t.tensor([input_ids]).to(DEVICE)
        output = model.generate(input_ids_tensor, max_new_tokens=100, temperature=TEMPERATURE)
        response = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[-1].strip()
        responses.append({"question": question["question"], "response": response})

    # Save the responses along with the questions to the output JSON file
    with open(output_filename, "w") as file:
        json.dump(responses, file, indent=4)

