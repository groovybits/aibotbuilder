#!/usr/local/bin/python3

## Read in FB messages and train GPT-3 to respond to them
## Create role playing game of your FB messages friends as NPCs
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
import re
import textwrap
import tiktoken
import sys

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def write_messages_to_pdf(text, output_file):
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(output_file)

def load_facebook_data(folder):
    conversations = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    if "messages" in data:
                        conversation = []
                        for message in data["messages"]:
                            if "content" in message and "sender_name" in message:
                                conversation.append({"sender": message["sender_name"], "content": message["content"]})
                        conversations.append(conversation)

    return conversations

def split_long_message(message, max_length):
    wrapped_lines = textwrap.wrap(message, max_length)
    return wrapped_lines

def remove_non_text_and_urls(input_string):
    # Remove URLs
    input_string = re.sub(r'http\S+|www\S+', '', input_string)

    # Remove non-text characters
    cleaned_string = ''.join(char for char in input_string if char.isalnum() or char.isspace())

    return cleaned_string

def create_training_data(conversations, your_name, max_length=1000):
    input_output_pairs = []

    for conversation in conversations:
        input_msg, output_msg = "", ""
        previous_sender = None

        for message in conversation:
            current_sender = message["sender"]
            content = message["content"]

            if len(content) > max_length:
                content_lines = split_long_message(content, max_length)
            else:
                content_lines = [content]

            for content_line in content_lines:
                if previous_sender != current_sender:
                    if input_msg and output_msg:
                        input_msg = remove_non_text_and_urls(input_msg)
                        output_msg = remove_non_text_and_urls(output_msg)
                        input_output_pairs.append({"input": previous_sender + ": " + input_msg, "output": current_sender + ": " + output_msg})
                        input_msg, output_msg = "", ""

                    if current_sender != your_name:
                        if input_msg:
                            input_msg += " "
                        input_msg += content_line
                    else:
                        if output_msg:
                            output_msg += " "
                        output_msg += content_line

                else:
                    if current_sender != your_name:
                        input_msg += f" {content_line}"
                    else:
                        output_msg += f" {content_line}"

                previous_sender = current_sender

        if input_msg and output_msg:
            input_output_pairs.append({"input": input_msg, "output": output_msg})

    return input_output_pairs

def save_training_data(input_output_pairs, output_file):
    output_data = []
    with open(output_file, "w") as f:
        for pair in input_output_pairs:
            json_line = {"prompt": f"{pair['input']}\n\n###\n\n", "completion": f" {pair['output']}\nEND"}
            f.write(json.dumps(json_line) + "\n")
            output_data.append(json_line)
    return output_data

def chatbot_qa(prompt, model_name):
    response = openai.Completion.create(
        model=model_name,
        prompt=prompt,
        temperature=0.8,
        max_tokens=100,
        top_p=1,
        stop=["END"],
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    return response.choices[0].text.strip()

def output_to_json(summarized_messages, output_file):
    with open(output_file, "w") as f:
        for conversation in summarized_messages:
            f.write(json.dumps(conversation) + "\n")


def parse_args():
    default_prompt = ""

    parser = argparse.ArgumentParser(description="Fine-tune GPT Models with Facebook Messenger data")
    parser.add_argument("--api_key", required=False, default="", help="Your OpenAI API key")
    parser.add_argument("--your_name", required=True, help="Your name for the chatbot")
    parser.add_argument("--folder", required=False, default="./", help="Path to the folder containing Facebook data")
    parser.add_argument("--gpt_fine_tuned_model", default="", help="Name of the GPT fine-tuned model to use")
    parser.add_argument("--output", default="training_data.json", help="Output file for the training data")
    parser.add_argument("--personality", default=default_prompt,
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

    ## Only use for chatGPT
    if args.gpt_fine_tuned_model != "":
        personality_description = args.personality
        question = args.question
        custom_prompt = f"{personality_description}User: {question}\n\n###\n\n"

        response = chatbot_qa(custom_prompt, args.gpt_fine_tuned_model)
        print("\n---\n" + response + "\n---\n")
        sys.exit(0)

    messages = load_facebook_data(folder)
    output_to_json(messages, output_file)

    input_output_pairs = create_training_data(messages, your_name)
    input_output_pairs_final = save_training_data(input_output_pairs, output_file)
    print(f"Summarized training data saved to {output_file}")

    cost_per_token = 0.003  # Replace with the current cost per token for the DaVinci model
    total_tokens = 0
    for pair in input_output_pairs_final:
        prompt_tokens = num_tokens_from_string(pair["prompt"], "gpt2")
        completion_tokens = num_tokens_from_string(pair["completion"], "gpt2")
        total_tokens += prompt_tokens + completion_tokens
    total_cost = (float(total_tokens) / 250) * float(cost_per_token)

    print("Total tokens in the output file: %s will cost $%02.02f" % (total_tokens, total_cost))
    sys.exit(0)

