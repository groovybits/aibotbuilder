#!/usr/local/bin/python3

## Rewrite of v1 evolving the way we work
##
## Christi Kennedy (C) March 2023
##
## To train GPT off the output json run:
##
##  openai tools fine_tunes.prepare_data -f training_data.json
##  openai api fine_tunes.create -t training_data_prepared.jsonl -m davinci
##  openai api fine_tunes.follow -i <YOUR_FINE_TUNE_JOB_ID>
##
##  Read: https://platform.openai.com/docs/guides/fine-tuning


import argparse
import fpdf
import json
import openai
import os
import tiktoken
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sentencepiece
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_summary_t5(text, max_chars, tokenizer, model):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=max_chars, truncation=True)
    outputs = model.generate(inputs, max_length=max_chars, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0])
    return summary

def parallel_summarize_t5(msg, max_chars, tokenizer, model):
    return generate_summary_t5(msg, max_chars, tokenizer, model)

def generate_summary_gpt(text, max_chars, model_name):
    prompt = f"Please summarize the following text within {max_chars} characters:\n\n{text}\n\nSummary:"
    response = openai.Completion.create(engine=model_name, prompt=prompt, max_tokens=max_chars, n=1, stop=None, temperature=0.5)
    summary = response.choices[0].text.strip()
    return summary

def save_training_data(input_output_pairs, output_file):
    output_data = []
    with open(output_file, "w") as f:
        for pair in input_output_pairs:
            json_line = {"prompt": pair["input"], "completion": pair["output"]}
            f.write(json.dumps(json_line) + "\n")
            output_data.append(json_line)
    return output_data

def write_messages_to_pdf(text, output_file):
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(output_file)

def load_facebook_data(folder):
    messages = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    if "messages" in data:
                        for message in data["messages"]:
                            if "content" in message:
                                messages.append(message["content"])

    return messages

def create_training_data(messages, your_name):
    input_output_pairs = []

    for i in range(1, len(messages), 2):
        input_msg = f"User: {messages[i-1]}\nAI (as {your_name}):"
        output_msg = f"{messages[i]}"
        input_output_pairs.append({"input": input_msg, "output": output_msg})

    return input_output_pairs

def fine_tune_codex(input_output_pairs, model_name):
    pass

def chatbot_qa(prompt, model_name):
    response = openai.Completion.create(
        engine=model_name,
        prompt=prompt,
        temperature=0.5,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].text.strip()

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Codex with Facebook Messenger data")
    parser.add_argument("--api_key", required=True, help="Your OpenAI API key")
    parser.add_argument("--your_name", required=True, help="Your name for the chatbot")
    parser.add_argument("--folder", required=True, help="Path to the folder containing Facebook data")
    parser.add_argument("--gpt_summarizer_model", default="", help="Name of the GPT summarization model to use")
    parser.add_argument("--gpt_fine_tuned_model", default="", help="Name of the GPT fine-tuned model to use")
    parser.add_argument("--output", default="training_data.json", help="Output file for the training data")
    parser.add_argument("--t5_summarizer_model", default="", help="Summarization model for the training data: use t5-small to enable")
    parser.add_argument("--personality", default="Use the tuning input to craft a personality of the persons answers to questions",
                        help="General personality of your AI bot")
    parser.add_argument("--question", default="Tell me about yourself?", help="Question to ask your AI bot")
    parser.add_argument("--max_chars", type=int, default=50, help="Maximum number of characters for summary")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    openai.api_key = args.api_key
    your_name = args.your_name
    folder = args.folder
    output_file = args.output
    max_chars = args.max_chars

    messages = load_facebook_data(folder)

    # T5
    if args.gpt_summarizer_model == "":
        if args.t5_summarizer_model == "":
            ## no summaries
            summarized_messages = messages
        else:
            tokenizer = T5Tokenizer.from_pretrained(args.t5_summarizer_model, model_max_length=max_chars)
            model = T5ForConditionalGeneration.from_pretrained(args.t5_summarizer_model)
            with ProcessPoolExecutor() as executor:
                summarized_messages = list(tqdm(executor.map(parallel_summarize_t5,
                                                             messages, [max_chars] * len(messages),
                                                             [tokenizer] * len(messages), [model] * len(messages)),
                                                total=len(messages), desc="Summarizing messages"))
    else:
        # chatGPT
        summarized_messages = [generate_summary_gpt(msg, max_chars, args.gpt_summarizer_model) for msg in messages]

    input_output_pairs = create_training_data(summarized_messages, your_name)
    input_output_pairs_final = save_training_data(input_output_pairs, output_file)
    print(f"Summarized training data saved to {output_file}")

    cost_per_token = 0.030  # Replace with the current cost per token for the DaVinci model
    total_tokens = 0
    for pair in input_output_pairs_final:
        print(pair)
        prompt_tokens = num_tokens_from_string(pair["prompt"], "gpt2")
        completion_tokens = num_tokens_from_string(pair["completion"], "gpt2")
        total_tokens += prompt_tokens + completion_tokens
    total_cost = (float(total_tokens) / 1000) * float(cost_per_token)

    print("Total tokens in the output file: %s will cost $%02.02f" % (total_tokens, total_cost))

    ## Only use for chatGPT
    if args.gpt_fine_tuned_model == "":
        sys.exit(0)

    # Test the chatbot with a custom prompt
    fine_tune_codex(input_output_pairs, args.gpt_fine_tuned_model)

    personality_description = args.personality
    question = args.question
    custom_prompt = f"{personality_description}User: {question}\nAI (as {your_name}):"
    response = chatbot_qa(custom_prompt, args.gpt_fine_tuned_model)
    print("\n---\n" + response + "\n---\n")

