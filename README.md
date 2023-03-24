# aibotbuilder
AI Bot Builder from FB Messages, SMS, PDF Files/Books

Groovy.org example bots: https://groovy.org/groovy-ai-chat-bots


*How to use:
```
usage: ai_chatbot_builder.py [-h] [--your_name YOUR_NAME] [--gpt_api_key GPT_API_KEY] [--max_chars MAX_CHARS] [--use_gpt2] [--use_gpt3]

Script to process and summarize text messages.

options:
  -h, --help            show this help message and exit
  --your_name YOUR_NAME
                        Name in the Facebook profile and messages to use.
  --gpt_api_key GPT_API_KEY
                        GPT OpenAI Key
  --max_chars MAX_CHARS
                        Chatbase.io allowance for input characters.
  --use_gpt2            Use GPT-2 for summarization, uses NLP by default.
  --use_gpt3            Use GPT-3 for summarization instead of GPT-2.
```

*Run the script in the messages/ folder within the FaceBook download zipfile extraction.
```
$ unzip facebook-USERNAME.*.zip

$ cd ./facebook-USERNAME/messages/

$ ~/path/to/ai_chatbot_builder.py --your_name "John Doe" --max_chars 400000

## script will look for all files in subdirectories
## with the messages_NUMBER.json patter + *.xml and ./books/*.pdf
```

You can add PDF files to the ./books/ directory in that FB Archives messages/ folder like messages/books/BOOK.pdf too.
In addition you can export all your SMS messages from a phone to an XML and place that file in the messages/ folder
as FILENAME.xml which will be picked up too by the script.

The script uses NLP and was developed with chatGPT quickly to get started and now will be honed over time with
chatGPT helping speed up the development by many many times. You can use gpt2 or gpt3 with an API Key.

Check for the output as a .json, .txt and .pdf which the PDF can be put into chatbase.io as data.
Adjust the byte limit for your plans limits via the --max_chars arg. The byte limit is 4000000 by default or 3950000 since
it can be over if not leaving it a bit short.

Impovements welcome, feel free to submit patches / PRs!!!

---
Thank you!!!
Christi Kennedy




