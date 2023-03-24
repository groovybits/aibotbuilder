#!/usr/local/bin/python3

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fpdf
from halo import Halo
import itertools
import json
import openai
import os
import PyPDF2
import spacy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import xml.etree.ElementTree as ET


nlp = spacy.load("en_core_web_sm")

@contextmanager
def progress_spinner():
    spinner = Halo(text='Summarizing...', spinner='dots')
    spinner.start()
    try:
        yield spinner
    finally:
        spinner.stop()

def load_model(use_nlm, use_gpt3):
    if use_nlm:
        return None, None, "nlm"
    elif use_gpt3:
        return None, None, "gpt3"
    else:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        return model, tokenizer, "gpt2"

def read_facebook_data(file_path):
    with progress_spinner() as spinner:
        spinner.text = f"Reading Facebook data from {file_path}..."
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    return data

def read_sms_data(file_path):
    with progress_spinner() as spinner:
        spinner.text = f"Reading SMS data from {file_path}..."
        tree = ET.parse(file_path)
        root = tree.getroot()
        data = {'messages': []}

        for sms in root.findall('sms'):
            content = sms.get('body')
            sender = sms.get('contact_name')

            data['messages'].append({'sender_name': sender, 'content': content})

    return data

def extract_messages(data, your_name):
    with progress_spinner() as spinner:
        spinner.text = f"Extracting messages for {your_name}..."
        extracted_messages = []

        for message in data['messages']:
            if message['sender_name'] == your_name:
                extracted_messages.append(message)

    return extracted_messages

def transform_messages(messages):
    transformed_messages = []

    for message in messages:
        if 'content' in message:
            transformed_messages.append({'sender': message['sender_name'], 'content': message['content']})

    return transformed_messages

def check_gpt3_api_key(api_key):
    try:
        openai.api_key = api_key
        openai.Engine.list()
        return True
    except openai.OpenAIError as e:
        print(f"GPT-3 API key error: {e}")
        return False

def summarize_message_gpt3(message):
    content = message['content']
    prompt = f"Please summarize the following text:\n\n{content}\n"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.7
    )

    summary = response.choices[0].text.strip()
    return summary


def summarize_message_gpt(message, model, tokenizer):
    content = message['content']

    max_input_length = 1024 - tokenizer.num_special_tokens_to_add(pair=False) - 20  # Reserve some tokens for the prompt
    input_text_chunks = []
    current_chunk = ''

    for word in content.split():
        if len(current_chunk) + len(word) + 1 > max_input_length:
            input_text_chunks.append(current_chunk.strip())
            current_chunk = ''
        current_chunk += word + ' '

    if current_chunk.strip():
        input_text_chunks.append(current_chunk.strip())

    summaries = []
    for text_chunk in input_text_chunks:
        prompt = f"Please summarize the following text:\n\n{text_chunk}\n"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=1024,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.7,
            )

        summary = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        summaries.append(summary)

    return ' '.join(summaries)

def compress_messages_gpt(messages, model, tokenizer, max_chars, use_gpt3=False):
    compressed_text = ''
    current_chars = 0

    num_workers = os.cpu_count()
    print(f"Number of workers: {num_workers}")

    summarize_func = summarize_message_gpt3 if use_gpt3 else summarize_message_gpt

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        if use_gpt3:
            summaries = list(tqdm(executor.map(summarize_message_gpt3, messages), total=len(messages), desc="Summarizing"))
        else:
            summaries = list(tqdm(executor.map(lambda msg: summarize_message_gpt(msg, model, tokenizer), messages), total=len(messages), desc="Summarizing"))


    for summary in summaries:
        compressed_text += summary + ' '
        current_chars += len(summary)

        if current_chars >= max_chars:
            break

    return compressed_text

def compress_messages_nlp(messages):
    compressed_text = ''
    current_chars = 0

    for message in messages:
        content = message['content']
        content_with_space = content + ' '
        doc = nlp(content_with_space)

        for sent in doc.sents:
            sentence = sent.text
            sentence_length = len(sentence)

            compressed_text += sentence
            current_chars += sentence_length

    return compressed_text

def write_messages_to_file(messages, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(messages, file, ensure_ascii=False, indent=4)

def write_messages_to_pdf(text, output_file):
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(output_file)

def remove_non_latin1_characters(text):
    return text.encode('latin-1', errors='ignore').decode('latin-1')

def read_pdf_data(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        text = ''
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text

def main():
    # Create a parser
    parser = argparse.ArgumentParser(description='Script to process and summarize text messages.')

    # Add arguments
    parser.add_argument('--your_name', type=str, default="John Doe", help='Name in the Facebook profile and messages to use.')
    parser.add_argument('--gpt_api_key', type=str, default="None", help='GPT OpenAI Key')
    parser.add_argument('--max_chars', type=int, default=3950000, help='Chatbase.io allowance for input characters.')
    parser.add_argument('--use_nlm', action='store_true', help='Use NLM for summarization instead of GPT-2.')
    parser.add_argument('--use_gpt3', action='store_true', help='Use GPT-3 for summarization instead of GPT-2.')

    # Parse arguments
    args = parser.parse_args()

    # Use the parsed arguments
    your_name = args.your_name
    max_chars = args.max_chars
    use_nlm = args.use_nlm
    use_gpt3 = args.use_gpt3
    nlp.max_length = 9999999

    if use_gpt3:
        if not check_gpt3_api_key(args.gpt_api_key):
            print("Invalid GPT-3 API key. Exiting.")
            return
        openai.api_key = args.gpt_api_key

    output_json_file = "%s_output.json" % your_name.replace(' ', '_')
    output_text_file = "%s_output.txt" % your_name.replace(' ', '_')
    output_pdf_file = "%s_output.pdf" % your_name.replace(' ', '_')

    messages = []

    model, tokenizer, model_type = load_model(args.use_nlm, args.use_gpt3)

    for root, _, files in os.walk('.'):
        for file in files:
            if file.startswith('message_') and file.endswith('.json'):
                file_path = os.path.join(root, file)
                data = read_facebook_data(file_path)
                extracted_messages = extract_messages(data, your_name)
                transformed_messages = transform_messages(extracted_messages)
                messages.extend(transformed_messages)
            elif file.endswith('.xml'):
                file_path = os.path.join(root, file)
                data = read_sms_data(file_path)
                extracted_messages = extract_messages(data, your_name)
                transformed_messages = transform_messages(extracted_messages)
                messages.extend(transformed_messages)
            elif file.endswith('.pdf') and root.startswith('./books'):
                file_path = os.path.join(root, file)
                text = read_pdf_data(file_path)
                messages.append({'sender_name': your_name, 'content': text})

    compressed_text = None
    if use_nlm:
        compressed_text = compress_messages_nlm(messages)
    else:
        compressed_text = compress_messages_gpt(messages, model, tokenizer, max_chars, use_gpt3=args.use_gpt3)
    write_messages_to_file(messages, output_json_file)

    truncated_text = compressed_text[:max_chars]

    # Remove non-Latin-1 characters
    latin1_text = remove_non_latin1_characters(truncated_text)

    with open(output_text_file, 'w', encoding='latin-1') as file:
        file.write(latin1_text)

    write_messages_to_pdf(latin1_text, output_pdf_file)


if __name__ == '__main__':
    main()

