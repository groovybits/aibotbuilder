import os
import json
import fpdf
import spacy
import xml.etree.ElementTree as ET

nlp = spacy.load("en_core_web_sm")

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

def compress_messages(messages):
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

def main():
    your_name = "John Doe"
    output_json_file = "output.json"
    output_text_file = "output.txt"
    output_pdf_file = "output.pdf"

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

    compressed_text = compress_messages(messages)
    write_messages_to_file(messages, output_json_file)

    max_chars = 3950000
    truncated_text = compressed_text[:max_chars]

    with open(output_text_file, 'w', encoding='utf-8') as file:
        file.write(truncated_text)

    write_messages_to_pdf(truncated_text, output_pdf_file)


if __name__ == '__main__':
    main()

