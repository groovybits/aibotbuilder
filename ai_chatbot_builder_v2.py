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
import json
import openai
import os
import sys
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def load_facebook_data(folder, your_name, target_user=None):
    conversations = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    if "messages" in data:
                        conversation = []
                        for message in data["messages"]:
                            if "content" in message and "sender_name" in message and message["sender_name"] != "Facebook user" and (message["sender_name"] == your_name or (target_user is None or message["sender_name"] == target_user)):
                                conversation.append({"sender": message["sender_name"], "content": message["content"]})
                        conversations.append(conversation)

    return conversations

def create_training_data(conversations, your_name):
    input_output_pairs = []

    for conversation in conversations:
        for i in range(0, len(conversation) - 1, 2):
            prompt = conversation[i + 1]["sender"] + ": " + conversation[i + 1]["content"]
            completion = conversation[i]["sender"] + ": " + conversation[i]["content"]
            input_output_pairs.append({"input": prompt.strip(), "output": completion.strip()})

    return input_output_pairs

def save_training_data(input_output_pairs, output_file):
    output_data = []
    with open(output_file, "w") as f:
        for pair in input_output_pairs:
            json_line = {"prompt": f"{pair['input']}\n\n###\n\n", "completion": f"{pair['output']}\n"}
            f.write(json.dumps(json_line) + "\n")
            output_data.append(json_line)
    return output_data

def parse_args():
    parser = argparse.ArgumentParser(description="Create Training Data from Facebook Messenger")
    parser.add_argument("--your_name", required=True, help="Your name for the chatbot")
    parser.add_argument("--target_user", required=False, default=None, help="Target user to include in the training data (optional)")
    parser.add_argument("--folder", required=False, default="./", help="Path to the folder containing Facebook data")
    parser.add_argument("--output", default="training_data.json", help="Output file for the training data")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    your_name = args.your_name
    target_user = args.target_user
    folder = args.folder
    output_file = args.output

    messages = load_facebook_data(folder, your_name, target_user)
    input_output_pairs = create_training_data(messages, your_name)
    input_output_pairs_final = save_training_data(input_output_pairs, output_file)
    print(f"Training data saved to {output_file}")

    #cost_per_token = 0.0300 # davinci
    #cost_per_token = 0.0030 # curie
    cost_per_token = 0.0006 # babbage
    #cost_per_token = 0.0004 # ada

    total_tokens = 0
    for pair in input_output_pairs_final:
        prompt_tokens = num_tokens_from_string(pair["prompt"], "gpt2")
        completion_tokens = num_tokens_from_string(pair["completion"], "gpt2")
        total_tokens += prompt_tokens + completion_tokens
    total_cost = (float(total_tokens) / 250) * float(cost_per_token)

    print("Total tokens in the output file: %s will cost $%02.02f" % (total_tokens, total_cost))
    sys.exit(0)

