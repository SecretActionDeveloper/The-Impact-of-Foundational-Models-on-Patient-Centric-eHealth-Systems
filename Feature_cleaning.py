### Step 2: preprocess the dataset by removing stop words, lemmatizing, and removing punctuations

import pandas as pd
import ast
import os
from groq import Groq
import string
import re
import nltk
# nltk.download()
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def clean_text(text):
  if type(text) == str:
    # text = emoji.demojize(text)
    text = text.lower()
    punc =  string.punctuation
    text = "".join([char for char in text if char not in punc])
    text =  " ".join(text.split())
    text = re.sub(r'\s+',' ', text)
    words = word_tokenize(text)
    # #Remove stop words,
    filtered_words = [word for word in words if word not in stop_words]
    #     # Lemmatize the reviews.
    lem = [lemmatizer.lemmatize(word) for word in filtered_words]
    text = " ".join(lem)
    replace_list = [',', '\'', '\"', '_', '-', "\\",  "\/", "\n"]
    for elm in replace_list:
        try: text = text.replace(elm, '')
        except: pass
  return text

def parse_list(s):
    # Use regex to match quoted strings correctly (single or double quotes)
    # Escape internal single quotes (like in "seek doctor's advice")
    list_str=  s.split(',')
    res = []
    for elm in list_str: 
        cleaned_string = re.sub(r'[^a-zA-Z0-9\s,]', '', elm)
        res.append(cleaned_string)
    # Now use ast.literal_eval safely
    return res

path = "./Application_with_features.csv"

df = pd.read_csv(path)
df["features_processed"] = ""
for i in range(len(df.index)):
    features = parse_list(df["uncleaned_LLM_features"][i])
    clean = []
    for feature in features:
        feature = clean_text(feature)
        clean.append(feature)
    df["features_processed"][i] = clean

df.to_csv(path, index = False)