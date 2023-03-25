# aibotbuilder
AI Bot Builder from FB Messages, SMS, PDF Files/Books

Groovy.org example bots: https://groovy.org/groovy-ai-chat-bots use chatbase.co

For v2 you will need a GPT login and a paid account with API access for gpt-3.

There are two versions currently:
 - v1 The first is for chatbase.co as a bot builder script named ai_chatbot_builder.py.
 - v2 is a gpt fine-tuning script as ai_chatbot_builder_v2.py that creates a model with your message data,
     see https://platform.openai.com/docs/guides/fine-tuning to understand how gpt works with fine-tuning.


## Run the script in the messages/ folder within the FaceBook download zipfile extraction.
```
$ unzip facebook-USERNAME.*.zip

$ cd ./facebook-USERNAME/messages/

# v1
## script will look for all files in subdirectories
## with the messages_NUMBER.json patter + *.xml and ./books/*.pdf
$ ~/path/to/ai_chatbot_builder.py --your_names "John Doe","Jane Doe" --max_chars 400000

# v2
## script will use facebook messages and create prompt and response json set for
## use as a gpt fine tuning data input to a model like curie, davinci, babbage or ada
# first create the json output file
$ ~/path/to/ai_chatbot_builder_v2.py --your_name "John Doe" \
        ----folder ~/Documents/facebook-johndoe/messages/inbox/ --output john_doe.json

# format it with openai, then upload and have gpt analyze (costs money)
$ openai tools fine_tunes.prepare_data -f john_doe.json
$ openai api fine_tunes.create -t john_doe_prepared.jsonl -m curie

# wait till finished building model, check periodically the output <YOUR_FINE_TUNE_JOB_ID>
$ openai api fine_tunes.get -i <YOUR_FINE_TUNE_JOB_ID>

# when finished, you can reference the gpt fine_tuned_model this way
$ ./ai_chatbot_builder_v2.py --your_name "John Doe" --api_key=`cat API_KEY.txt` \
        --gpt_fine_tuned_model babbage:organization-2023-03-25-10-52-21 --question "Tell me about yourself?"

# Also you can go to the GPT Playground and see your model there
https://platform.openai.com/playground
```

## How to use v1:
```
usage: ai_chatbot_builder.py [-h] [--your_names YOUR_NAMES] [--gpt_api_key GPT_API_KEY] [--max_chars MAX_CHARS] [--use_gpt2] [--use_gpt3]

Script to process and summarize text messages.

options:
  -h, --help            show this help message and exit
  --your_names YOUR_NAMES
                        Comma-separated names in the Facebook profile and messages to use.
  --gpt_api_key GPT_API_KEY
                        GPT OpenAI Key
  --max_chars MAX_CHARS
                        Chatbase.io allowance for input characters.
  --use_gpt2            Use GPT-2 for summarization, the default is NLP (fast).
  --use_gpt3            Use GPT-3 for summarization instead of GPT-2. Use your API Key
```

## How to use v2:
```
usage: ai_chatbot_builder_v2.py [-h] [--api_key API_KEY] --your_name YOUR_NAME [--folder FOLDER] [--gpt_summarizer_model GPT_SUMMARIZER_MODEL]
                                [--gpt_fine_tuned_model GPT_FINE_TUNED_MODEL] [--output OUTPUT] [--t5_summarizer_model T5_SUMMARIZER_MODEL] [--personality PERSONALITY]
                                [--question QUESTION] [--max_chars MAX_CHARS]

Fine-tune Codex with Facebook Messenger data

options:
  -h, --help            show this help message and exit
  --api_key API_KEY     Your OpenAI API key
  --your_name YOUR_NAME
                        Your name for the chatbot
  --folder FOLDER       Path to the folder containing Facebook data
  --gpt_summarizer_model GPT_SUMMARIZER_MODEL
                        Name of the GPT summarization model to use
  --gpt_fine_tuned_model GPT_FINE_TUNED_MODEL
                        Name of the GPT fine-tuned model to use
  --output OUTPUT       Output file for the training data
  --t5_summarizer_model T5_SUMMARIZER_MODEL
                        Summarization model for the training data: use t5-small to enable
  --personality PERSONALITY
                        General personality of your AI bot
  --question QUESTION   Question to ask your AI bot
  --max_chars MAX_CHARS
                        Maximum number of characters for summary
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




