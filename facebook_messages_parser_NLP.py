#!/usr/local/bin/python3

import os
import json
import fpdf
import torch
import spacy
import xml.etree.ElementTree as ET
import PyPDF2
from transformers import GPT2LMHeadModel, GPT2Tokenizer

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 9999999

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def read_facebook_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def read_sms_data(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = {'messages': []}

    for sms in root.findall('sms'):
        content = sms.get('body')
        sender = sms.get('contact_name')

        data['messages'].append({'sender_name': sender, 'content': content})

    return data

def extract_messages(data, your_name):
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

def compress_messages_gpt2(messages):
    compressed_text = ''
    current_chars = 0

    for message in messages:
        content = message['content']
        content_with_space = content + ' '
        prompt = f"Please summarize the following text:\n\n{content_with_space}\n"

        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.7,
            )

        summary = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        compressed_text += summary + ' '
        current_chars += len(summary)

        if current_chars >= 3950000:
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
    your_name = "John Doe"
    output_json_file = "%s_output.json" % your_name.replace(' ', '_')
    output_text_file = "%s_output.txt" % your_name.replace(' ', '_')
    output_pdf_file = "%s_output.pdf" % your_name.replace(' ', '_')

    messages = []

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

    compressed_text = compress_messages_gpt2(messages)
    write_messages_to_file(messages, output_json_file)

    max_chars = 3950000
    truncated_text = compressed_text[:max_chars]

    # Remove non-Latin-1 characters
    latin1_text = remove_non_latin1_characters(truncated_text)

    with open(output_text_file, 'w', encoding='latin-1') as file:
        file.write(latin1_text)

    write_messages_to_pdf(latin1_text, output_pdf_file)


if __name__ == '__main__':
    main()

