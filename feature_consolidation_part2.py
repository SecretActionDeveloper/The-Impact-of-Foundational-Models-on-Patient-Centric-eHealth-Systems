#### Step 4:clean the dataset by comparing to unique features

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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
warnings.filterwarnings('ignore')

path = "data\\llm_features_processed.csv"
save_path = "data\\Application_with_features_consolidated.csv"

df = pd.read_csv(path)
print(len(list(df["appId"])))
print(len(list(set(list(df["appId"])))))

feature_path = "data\\unique_features.csv"
df_features = pd.read_csv(feature_path)
print("number of unique features: "  + str(len(df_features["Feature"])))
print(len(list(set(list(df_features["Feature"])))))


model = SentenceTransformer('bert-base-nli-mean-tokens')
THRESHOLD = 90

ci_features = []
df["consolidated_features"] = ""; df["Feature_embedding"] = ""
for i in range(len(df.index)):
    print("working on " + str(i) +"/" + str(len(df.index)))
    features_list_raw = ast.literal_eval(df["features_processed"][i])
    features_list = list(set(features_list_raw))
    cleaned_feature = []; embeddings = []
    for feature in features_list:
        feature_embed = model.encode(feature)
        vector1 = feature_embed.reshape(1, -1)
        found_feature = False
        for j in range(len(df_features.index)):
            uniq_feat = df_features["Feature"][j]
            emb_str = df_features["embedding"][j]
            try: 
                    emb_str = emb_str.replace("[", "")
                    emb_str = emb_str.replace("]", "")
                    uniq_embed = np.fromstring(emb_str, sep=' ')
            except: uniq_embed = emb_str
            vector2 = uniq_embed.reshape(1, -1)
            similarity_scores = cosine_similarity(vector1, vector2)[0][0] * 100
            if similarity_scores >= THRESHOLD: 
                found_feature = True
                if uniq_feat not in cleaned_feature:
                    cleaned_feature.append(uniq_feat)
                    embeddings.append(uniq_embed)
                break
        if found_feature ==  False:
            print("did not found the unique feature")
            if feature not in cleaned_feature:
                cleaned_feature.append(feature)
                embeddings.append(feature_embed)
    ci_features.extend(cleaned_feature)
    df["consolidated_features"][i] = cleaned_feature
    df["Feature_embedding"][i] = embeddings
    df.to_csv(save_path, index = False)

print("number of unique features: "  + str(len(df_features["Feature"])))
print("number of unique features: "  + str(len(list(set(list(df_features["Feature"]))))))


print("number of features in dataset: " + str(len(list(set(ci_features)))))
# number of unique features: 942
# number of unique features: 942
# number of features in dataset: 942