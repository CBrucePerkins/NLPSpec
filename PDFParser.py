'''
Sources:
+h
+ttps://automatetheboringstuff.com/chapter13/ (importing text from PDF)

https://www.science-emergence.com/Articles/How-to-remove-string-control-characters-n-t-r-in-python/ 
    (remove control characters)
    
    
NLTK Book:
http://www.nltk.org/book/

Using Spacy:
https://spacy.io/usage/linguistic-features
'''

from tika import parser
import re
import nltk
import pprint
import os

# Setting directory to "Data" folder in current directory. Your PDF's should be in this folder.
cwd = os.getcwd()
data = '\\Data\\'
wd = cwd + data

# This will run through the directory of the "Data" folder, parsing the PDF files into a python list
# named "documents". Note the object "regex" is used to remove string control characters such as "\n"
# and "\t".
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

# A function for tokenizing and tagging the data we just loaded, returning tokenized sentences
# with POS tags.
def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return(sentences)

# Applies the function to the documents object, saving the result into "tokens" object.
#tokens = []
#for doc in documents:
#    tokens.append(ie_preprocess(doc))

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

# Applies all relevant pre-processing measures using Spacy library including:
# POS-tagging, token shape, dependency parsing, etc.
processed_docs = []
for doc in documents:
    processed_docs.append(nlp(doc))

#Organizes the processed text data by document, adding relevant features to each token
#in a useful python list
train = []
for doc in processed_docs:
    data = []
    i = 0
    for token in doc:
        children = [child for child in token.children]
        children_index = []
        
        for child in children:
            children_index.append(child.idx)
        
        temp = {'index': i,'text':token.text, 'lemma':token.lemma_,
        	 'pos':token.pos_, 'tag':token.tag_, 'dep':token.dep_,
              'shape':token.shape_, 'alphatoken':token.is_alpha, 
              'isstop':token.is_stop, 'parent_index':token.head.idx,
               'parent':token.head.text, 'children_index': children_index, 
               'children': children, 'iob': token.ent_iob_,
                'embedding': token.vector}
        data.append(temp)
        i+=1
    train.append(data)



# Writes the "documents" and "tokens" objects to JSON files in the folder 
# called "train" in the current directory.
import json
with open('train/documents.json', 'w') as outfile:
    json.dump(documents, outfile)

with open('train/train.json', 'w') as outfile:
    json.dump(train, outfile)