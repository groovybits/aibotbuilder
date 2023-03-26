import os
import json
import re
import sys
from fpdf import FPDF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download

download("stopwords")
download("punkt")

stop_words = set(stopwords.words("english"))

def remove_stop_words_and_links(text):
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w in stop_words and not re.match(r"http\S+", w)]
    return " ".join(filtered_text)

def process_conversation(conversation, main_person, output_dir):
    participants = [p["name"] for p in conversation["participants"]]
    if main_person in participants:
        for message in conversation["messages"]:
            if "content" in message:
                sender_name = message["sender_name"]
                cleaned_content = remove_stop_words_and_links(message["content"])

                pdf_filename = os.path.join(output_dir, f"{sender_name}_chatbot.pdf")
                txt_filename = os.path.join(output_dir, f"{sender_name}_chatbot.txt")
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, txt=cleaned_content)

                with open(txt_filename, "a", encoding="utf-8") as txt_file:
                    txt_file.write(cleaned_content + "\n\n")

                pdf.output(pdf_filename)

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

