#!/usr/local/bin/python3

import os
import json
import re
import argparse
from pathlib import Path
from fpdf import FPDF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download

download("punkt")
download("stopwords")
stop_words = set(stopwords.words("english"))

def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def remove_stop_words(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return " ".join(filtered_text)

def preprocess_message(message):
    message = remove_urls(message)
    message = remove_stop_words(message)
    return message

def convert_to_training_data(messenger_data, main_name):
    training_data = []

    for message in messenger_data['messages']:
        if 'content' in message:
            preprocessed_content = preprocess_message(message['content'])
            if message['sender_name'] == main_name:
                prompt = f"Bot: {preprocessed_content}"
            else:
                prompt = f"{message['sender_name']}: {preprocessed_content}"
            completion = f"{message['sender_name']} says: {preprocessed_content}"
            training_data.append({"prompt": prompt, "completion": completion})

    return training_data

def generate_pdf(conversations, output_path, main_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for conversation in conversations:
        if conversation['prompt'].startswith(f"Bot: "):
            pdf.set_text_color(0, 0, 255)  # Set text color to blue for the main person (Bot)
        else:
            pdf.set_text_color(0, 0, 0)  # Set text color to black for others
        pdf.cell(200, 10, txt=conversation['prompt'], ln=True)

    pdf.output(output_path)

def main(directory, main_name):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("message_") and file.endswith(".json"):
                print("Processing file: %s" % file)
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    messenger_data = json.load(f)

                # Check if main_name is in the conversation
                participants = {p['name'] for p in messenger_data['participants']}
                print(participants)
                if main_name not in participants:
                    continue

                training_data = convert_to_training_data(messenger_data, main_name)
                output_directory = Path("output")
                output_directory.mkdir(parents=True, exist_ok=True)

                for participant in participants:
                    if participant == main_name:
                        output_path = output_directory / f"{participant}_main_bot.pdf"
                    else:
                        output_path = output_directory / f"{participant}.pdf"
                    generate_pdf(training_data, output_path, main_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Facebook Messenger data and generate PDFs.")
    parser.add_argument("directory", type=str, help="Path to the directory containing Facebook Messenger data.")
    parser.add_argument("main_name", type=str, help="Name of the main person to be treated as the main bot.")
    args = parser.parse_args()

    main(args.directory, args.main_name)

