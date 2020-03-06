from tika import parser
import re
import json
import csv
import os
import nltk

# This script takes the files stored in the chosen folder, "/Labeled Data/" in this case, tokenizes and flattens
# them into a single column. The object is then saved to a CSV for easiest annotation method.

cwd = os.getcwd()
data = '\\Labeled Data\\'
wd = cwd + data

regex = re.compile(r'[\n\r\t]')
documents = []
for file in os.listdir(wd):
    filename = os.fsdecode(file)
    if filename.endswith(".pdf"):
        file_data = parser.from_file(wd+file)
        # Get files text content
        text = file_data['content']
        text = regex.sub("", text)
        documents.append(text)
        continue
    else:
        continue

def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return(sentences)

tokens = []

for doc in documents:
    tokens.append(ie_preprocess(doc))

flat_list = [item for sublist in tokens for item in sublist]
flat_list = [item for sublist in flat_list for item in sublist]

# Now take the flat_list file and label it with the NER tags chosen. The next step
# will prepare the data for classification.

myFile = open(wd+'ner_labeled.csv', 'w',newline="",encoding='utf-8')  
with myFile:  
   writer = csv.writer(myFile)
   writer.writerows(flat_list)