import csv
import string
import re
import nltk
import os
import pandas as plot_confusion_matrix_dict
from nltk.tag.stanford import StanfordNERTagger
import os

java_path = "C:/Program Files/Java/jre1.8.0_201/bin"
os.environ['JAVAHOME'] = java_path
ner_model_path = 'C:/NLTK/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz'
path_to_ner = 'C:/NLTK/stanford-ner-2018-10-16/stanford-ner.jar'
path_custom_ner_model = 'C:/NLTK/stanford-ner-2018-10-16/ner-model.ser.gz'

def plot_confusion_matrix_dict(matrix_dict, rotation=45, outside_label=""):
    labels = set([y for y, _ in matrix_dict.keys()] + 
    	[y for _, y in matrix_dict.keys()])
    sorted_labels = sorted(labels)
    matrix = np.zeros((len(sorted_labels), len(sorted_labels)))
    for i1, y1 in enumerate(sorted_labels):
        for i2, y2 in enumerate(sorted_labels):
            if y1 != outside_label or y2 != outside_label:
                matrix[i1, i2] = matrix_dict[y1, y2]
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(sorted_labels))
    plt.xticks(tick_marks, sorted_labels, rotation=rotation)
    plt.yticks(tick_marks, sorted_labels)
    plt.tight_layout()
    # return matrix

def create_confusion_matrix(data, predictions):
    """
    Creates a confusion matrix that counts for each gold label how 
    often it was labelled by what label in the predictions.
    Args:
        data: a list of gold (x,y) pairs.
        predictions: a list of y labels, same length and with matching order.

    Returns:
        a `defaultdict` that maps `(gold_label,guess_label)` 
        pairs to their prediction counts.
    """
    confusion = defaultdict(int)
    for (x, y_gold), y_guess in zip(data, predictions):
        confusion[(y_gold, y_guess)] += 1
    return confusion

def full_evaluation_table(confusion_matrix):
    """
    Produce a pandas data-frame with Precision, F1 and Recall for all labels.
    Args:
        confusion_matrix: the confusion matrix to calculate metrics from.

    Returns:
        a pandas Dataframe with one row per gold label, and one more row for the aggregate of all labels.
    """
    labels = sorted(list({l for l, _ in confusion_matrix.keys()} | {l for _, 
    	l in confusion_matrix.keys()}))
    gold_counts = defaultdict(int)
    guess_counts = defaultdict(int)
    for (gold_label, guess_label), count in confusion_matrix.items():
        if gold_label != "None":
            gold_counts[gold_label] += count
            gold_counts["[All]"] += count
        if guess_label != "None":
            guess_counts[guess_label] += count
            guess_counts["[All]"] += count

    result_table = []
    for label in labels:
        if label != "None":
            result_table.append((label, gold_counts[label], guess_counts[label], 
            	*evaluate(confusion_matrix, {label})))

    result_table.append(("[All]", gold_counts["[All]"], guess_counts["[All]"], 
    	*evaluate(confusion_matrix)))
    return pd.DataFrame(result_table, columns=('Label', 'Gold', 'Guess', 'Precision', 'Recall', 'F1'))

# A set of useful functions to be used in the feature template function.
# Return boolean values for specific criterion that may help
# separate class distinctions.
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def has_numbers(inputString):
     return any(char.isdigit() for char in inputString)
    
def contains_mw(word):
    res = 'FALSE'
    for i in range(0,len(word)):    
        if(word[i]=='W'):
            if(word[i-1]=='M'):
                res = 'TRUE'
    return res 

def contains_gw(word):
    res = 'FALSE'
    for i in range(0,len(word)):    
        if(word[i]=='W'):
            if(word[i-1]=='G'):
                res = 'TRUE'
    return res 

def contains_mwh(word):
    res = 'FALSE'
    for i in range(0,len(word)):    
        if(word[i]=='h'):
            if(word[i-1]=='W'):
                if(word[i-2]=='M'):
                    res = 'TRUE'
    return res 

#Some tokens contain a '-' in the token to separate numbers. This might help the model
#differentiate between some classes
def contains_dash(word):
    res = 'FALSE'
    for i in range(0,len(word)):    
        if(word[i]=='-'):
            res = 'TRUE'
    return res 


train = []
with open("labeled data/train.tsv",encoding = 'utf-8') as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        train.append(row)
        
train_x = [i[0] for i in train]
train_y = [i[1] for i in train]

test = []
with open("labeled data/test.tsv",encoding = 'utf-8') as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        test.append(row)
        
test_x = [i[0] for i in test]
test_y = [i[1] for i in test]

ner_tagger = StanfordNERTagger(path_custom_ner_model, path_to_ner, encoding='utf8')
tags = ner_tagger.tag(train_x)

guess = [i[1] for i in tags]

cm_dev = create_confusion_matrix(test,guess)  
# This line turns the confusion matrix into a evaluation table with Precision, Recall and F1 for all labels.
full_evaluation_table(cm_dev)

# The stanford tagger performs very poorly on our data. Most likely due to
# their use of features. A class such as "numbers" should be relatively easy
# for any NER tagger to pick up.

plot_confusion_matrix_dict(cm_dev,90, outside_label="O")


# Function for generating features the model can learn from
def features(train):
    result = []
    
    for i in range(0,len(train)):
        word = train[i]
        prev_word = 'PAD'
        if(i>0):
            prev_word = train[i-1]
        next_word = 'PAD'
        if(i < (len(train)-1)):
            next_word = train[i+1]
        is_num = is_number(word)
        has_num = has_numbers(word)
        is_percent = word == '%'
        next_percent = next_word == '%'
        has_mw = contains_mw(word)
        has_gw = contains_gw(word)
        has_mwh = contains_mwh(word)
        has_dash = contains_dash(word)
        try: 
            first_upper = str.isupper(word[0])
        except:
            ''
        feat = {'word':word,'previous word':prev_word,'next word':next_word,
        		'is number':is_num,'has number':has_num,
                'percent symbol':is_percent,'next is %':next_percent,'contains /MW':has_mw,
                'first letter capital':first_upper,'Contains GW':has_gw,'Contains MWh':has_mwh,
               'Contains dash':has_dash}
        result.append(feat)
    
    return result

train_feats = features(train_x)

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
vectorizer = DictVectorizer()
label_encoder = LabelEncoder()

xtrain = vectorizer.fit_transform(train_feats)
ytrain = label_encoder.fit_transform(train_y)

lr = LogisticRegression(C=1)
lr.fit(xtrain, ytrain)

# Function for making predictions
def predict(test):
    x = vectorizer.transform(features(test))
    y = label_encoder.inverse_transform(lr.predict(x))
    return y

guess = predict(test_x)

cm_dev = bio.create_confusion_matrix(test,guess)  
# This line turns the confusion matrix into a evaluation table with Precision,
# Recall and F1 for all labels.
full_evaluation_table(cm_dev)
plot_confusion_matrix_dict(cm_dev,90, outside_label="O")


sent1 = """Subsidies for pre-2020 projects continue to be highly attractive, even 
more than our original forecasts as we expect scale factors and efficiencies to 
reduce opex. For new projects post 2020, we believe Dong Energy could achieve a 
7-9% project IRR for the Borselle I&II project, whereas some of the other project 
winners could struggle to reach a high-single-digit project IRR without assuming 
the use of larger platforms. Other assumptions play a part in this judgement."""
sent2 = """Outside of Europe, despite higher subsidies being offset by higher 
costs (due to lack of supply chain), our preliminary analysis suggests that 
double-digit returns remain possible, albeit with higher risk. As we highlight 
in Sailing in the wind of growth, generic project IRRs in Germany are at 7-8% 
in Germany and a higher 12-15% in the UK, see Figure 45. We assume these figures 
to be accurate measures of the future."""
sent3 = """We find that the upcoming project tenders provide a lucrative opportunity 
for Orsted and Innogy. The Kriegers Flak project is expected to give the winner a 5% 
return. This is assuming projected bid prices shown in figures below."""

sent = [sent1,sent2,sent3]

def process_sents(sent):
    data = []
    for s in sent:
        sentences = nltk.sent_tokenize(s)
        tokens = []
        for sent in sentences:
            tokens.append(nltk.word_tokenize(sent))
        ner = []
        for tok in tokens:
            ner.append(predict(tok))
        row = {'sentences':sentences, 'tokens':tokens, 'entities':ner}
        data.append(row)
    return(data)

data = process_sents(sent)

# The model will assume a 'tri-gram' of sentences as input. The idea is to process the 
# text corpus beforehand in such a way that allows the model to move across each 
# sentence individually while looking at the adjacent sentences forrelevant information:

# 1) Run raw text through method that allows to cycle through three sentences at a time
# 2) Run tri-gram sentences through 'process_sents' method to be able to extract features
# 3) Run processed sentences through a feature extraction method in preparation for training

# Need to determine how the output should be structured/labeled for prediction

pairs = []
for i in range(0,len(data)):
    for j in range(0,len(data[i]['tokens'])):
        for l in range(0,len(data[i]['tokens'][j])):
            for m in range(l+1,len(data[i]['tokens'][j])):
                first = [data[i]['tokens'][j][l],data[i]['entities'][j][l]]
                second = [data[i]['tokens'][j][m],data[i]['entities'][j][m]]
                pairs.append([first,second])

rel_y = []
with open('train/rel_labels.csv', 'r') as f:
  reader = csv.reader(f)
  for row in reader:
        rel_y.append(row)

nr = ['NO RELATION']
irr = ['IRR OF']

for i in range(0,len(rel_y)):
    if(rel_y[i] == nr):
        rel_y[i] = 'NO RELATION'
        
    if(rel_y[i] == irr):
        rel_y[i] = 'IRR OF'

def relation_feat_extract(data):
    result = []
    for i in range(0,len(data)):
        for j in range(0,len(data[i]['tokens'])):
            for l in range(0,len(data[i]['tokens'][j])):
                irr = "FALSE"
                ret = "FALSE"
                for m in range(l+1,len(data[i]['tokens'][j])):
                    first = [data[i]['tokens'][j][l],data[i]['entities'][j][l]]
                    second = [data[i]['tokens'][j][m],data[i]['entities'][j][m]]
                    
                    for x in range(0,len(data[i]['tokens'])):
                        for y in range(0, len(data[i]['tokens'][x])):
                            if(data[i]['tokens'][x][y] == 'IRR' or data[i]['tokens'][x][y] == 'irr'):
                                irr = 'TRUE'
                            if(data[i]['tokens'][x][y] == 'return' or data[i]['tokens'][x][y] == 'returns'):
                                ret = 'TRUE'
                            if(data[i]['entities'][x][y] == 'LOC' or data[i]['entities'][x][y] == 'LOC'):
                                LOC = 'TRUE'
                    
                    
                    feat = {'First token':data[i]['tokens'][j][l],'First tag':data[i]['entities'][j][l],
                            'Second token':data[i]['tokens'][j][m],'Second tag':data[i]['entities'][j][m],
                           'Mentions IRR':irr,'Mentions returns':ret}
                    result.append(feat)
    return(result)

rel_x = relation_feat_extract(data)
vectorizer_rel = DictVectorizer()
label_encoder_rel = LabelEncoder()

xtrain = vectorizer_rel.fit_transform(rel_x)
ytrain = label_encoder_rel.fit_transform(rel_y)

lr_rel = LogisticRegression(C=1)
lr_rel.fit(xtrain, ytrain)

def predict_rel(test):
    pairs = relation_feat_extract(test)
    x = vectorizer_rel.transform(pairs)
    y = label_encoder_rel.inverse_transform(lr_rel.predict(x))
    relations = []
    rel = []
    for i in range(0,len(y)):
        if(y[i] != 'NO RELATION'):
            rel = [pairs[i][0],y[i],pairs[i][1]]
            relations.append(rel)
    
    if(rel == []):
        return y
    else:
        return rel



test = ["""At a price of £77/MWh, which is our low estimate after adjusting Dong's earlier winning 
tender in Netherlands (€72.7/MWh) for a number of factors (FX, inflation, transmission costs etc), 
we expect Dong's Hornsea 2 (capped at 1.5GW) and part of Innogy's Triton Knoll to clear the UK auction. 
We expect Dong to capture an IRR of ~9%-11% at this clearing price on the basis that its capex costs 
are 10%-20% lower than its adjacent under construction Hornsea 1 project"""]
test_x_rel = process_sents(test)
predict_rel(test_x_rel)