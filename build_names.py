#!/usr/local/bin/python3
#
import os
import subprocess

names_file = "names.txt"
names = []

# Using a with statement ensures that the file is closed after reading
with open(names_file, "r") as file:
    names = [line.strip() for line in file.readlines()]

for n in names:
    print("Analyzing %s" % n)
    subprocess.run(["ai_chatbot_builder.py", "--your_name", n])


