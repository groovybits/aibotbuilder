import os
import json
import glob
from fpdf import FPDF
import spacy

nlp = spacy.load('en_core_web_sm')

def read_facebook_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
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

def compress_messages(messages, max_characters):
    compressed_messages = []
    total_characters = 0
    buffer = []

    for message in messages:
        doc = nlp(message['content'])
        for sentence in doc.sents:
            sentence_text = sentence.text.strip()
            if total_characters + len(sentence_text) < max_characters:
                buffer.append(sentence_text)
                total_characters += len(sentence_text)
            else:
                break

        compressed_message = ' '.join(buffer)
        compressed_messages.append({'sender': message['sender'], 'content': compressed_message})
        buffer.clear()

    return compressed_messages

def write_messages_to_file(messages, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(messages, file, ensure_ascii=False, indent=4)

def create_pdf(messages, output_file):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for message in messages:
        pdf.multi_cell(0, 10, txt=message['sender'] + ": " + message['content'], align="L")
        pdf.ln()
    pdf.output(output_file)

def main():
    your_name = 'John Doe'
    output_file = 'output.json'
    output_pdf = 'output.pdf'
    max_characters = 3950000

    all_files = glob.glob("*/*/message_*.json")

    all_messages = []
    for file_path in all_files:
        data = read_facebook_data(file_path)
        messages = extract_messages(data, your_name)
        transformed_messages = transform_messages(messages)
        all_messages.extend(transformed_messages)

    compressed_messages = compress_messages(all_messages, max_characters)
    write_messages_to_file(compressed_messages, output_file)
    create_pdf(compressed_messages, output_pdf)

if __name__ == '__main__':
    main()

