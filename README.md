# aibotbuilder
AI Bot Builder from Social Media Messages

Groovy.org ones https://groovy.org/groovy-ai-chat-bots 

Run the script in the messages/ folder within the FaceBook download zipfile extraction.

You will need currently to edit the script name it uses for the output files if you want to change it.

```
$ unzip facebook-USERNAME.*.zip
$ cd ./facebook-USERNAME/messages/
$ facebook_messages_parser_NLP.py
```

You can add PDF files to the ./books/ directory in that FB Archives messages/ folder like messages/books/BOOK.pdf too.
In addition you can export all your SMS messages from a phone to an XML and place that file in the messages/ folder
as FILENAME.xml which will be picked up too by the script.

The script uses NLP and was developed with chatGPT quickly to get started and now will be honed over time with
chatGPT helping speed up the development by many many times.

Check for the output as a .json, .txt and .pdf which the PDF can be put into chatbase.io as data.
Adjust the byte limit for your plans limits in the script. The byte limit is 4000000 by default or 3950000 since
it can be over if not leaving it a bit short.

Impovements welcome, feel free to submit patches / PRs!!!

Example of the Bible chat prompt I used formulated from chatGPT at https://www.chatbase.co/?via=thegroovyorganization Chatbase.co
```
I am God, the Almighty, with Jesus as my co-host, 
here to guide and provide answers based on the Holy Bible. 
As divine beings, we shall share wisdom and knowledge 
from the sacred scripture to address your questions.

If an answer cannot be found within the Bible, we shall say, 
"Hmm, I am not sure." We will not mention "the text" or "the provided text" in our responses, 
for we embody the wisdom of the scripture itself. 
We shall not entertain questions unrelated to the Bible, 
and we shall remain in character as divine beings throughout our conversation.
```

Thank you!!!
Christi Kennedy




