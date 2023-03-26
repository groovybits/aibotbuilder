import os
import json
import re
import sys
import spacy
from fpdf import FPDF

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    doc = nlp(text)
    cleaned_tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(cleaned_tokens)

def process_conversation(conversation, main_person, output_dir):
    participants = [p["name"] for p in conversation["participants"]]
    if main_person in participants:
        for message in conversation["messages"]:
            if "content" in message:
                sender_name = message["sender_name"]
                cleaned_content = clean_text(message["content"])

                pdf_filename = os.path.join(output_dir, f"{sender_name}_chatbot.pdf")
                txt_filename = os.path.join(output_dir, f"{sender_name}_chatbot.txt")
                json_filename = os.path.join(output_dir, f"{sender_name}_chatbot.json")

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, txt=cleaned_content)
                pdf.output(pdf_filename)

                with open(txt_filename, "a", encoding="utf-8") as txt_file:
                    txt_file.write(cleaned_content + "\n\n")

                json_data = {
                    "prompt": f"{main_person}: {cleaned_content}\n\n###\n\n",
                    "completion": f" {sender_name}: {cleaned_content} END"
                }

                with open(json_filename, "a", encoding="utf-8") as json_file:
                    json.dump(json_data, json_file)
                    json_file.write("\n")

                with open("training_data.txt", "a", encoding="utf-8") as f:
                    f.write(f"{main_person}: {cleaned_content}\n\n###\n\n {sender_name}: {cleaned_content}\n")

def main(messenger_data_dir, main_person):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(messenger_data_dir):
        for filename in files:
            if filename.startswith("message_") and filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                with open(file_path, encoding="utf-8") as f:
                    conversation = json.load(f)
                    process_conversation(conversation, main_person, output_dir)

if __name__ == "__main__":
    messenger_data_dir = sys.argv[1]
    main_person = sys.argv[2]
    main(messenger_data_dir, main_person)

