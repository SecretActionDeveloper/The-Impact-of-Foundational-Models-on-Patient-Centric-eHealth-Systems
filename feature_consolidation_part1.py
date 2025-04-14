#### consolidate all the features from the LLMs into a single file 

import pandas as pd
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
warnings.filterwarnings('ignore')



path =  "./Application_with_features.csv"
model = SentenceTransformer('bert-base-nli-mean-tokens')
THRESHOLD = 90

df = pd.read_csv(path)
feature_path = "./unique_features.csv"
# df_features = pd.read_csv(feature_path)
df_features = pd.DataFrame(columns=["Feature", "embedding"])
unique_feature = list(df_features["Feature"])

all_features = []
for i in range(len(df.index)):
    features = ast.literal_eval(df["features_processed"][i])
    all_features.extend(features)

feature_list = []; 
for i in range(len(df.index)):
    features = list(set(ast.literal_eval(df["features_processed"][i])))
    print("working on " + str(i) +"/" + str(len(df.index)) + ": " + str(len(features)))
    if len(unique_feature) == 0: 
        feature_list.extend(features)
        for feature in features:
            sentence_embeddings = model.encode(feature)
            new_row = {'Feature': feature, 'embedding': sentence_embeddings}
            df_features.loc[len(df_features)] = new_row
            df_features.to_csv(feature_path, index = False)
        unique_feature = list(df_features["Feature"])
    else:
        count = 0
        # if i == 1082: print(features)
        for feature_can in features:
            if feature_can not in unique_feature: 
                count += 1
                print("working on " + str(i) +"/" + str(len(df.index)) + ": Feature" +str(count) +"/"+ str(len(features)))
                feature_can_embeddings = model.encode(feature_can)
                vector1 = feature_can_embeddings.reshape(1, -1)
                found_feature = False
                for j in range(len(df_features.index)):
                    feature = df_features["Feature"][j]
                    emb_str = df_features["embedding"][j]
                    try: 
                        emb_str = emb_str.replace("[", "")
                        emb_str = emb_str.replace("]", "")
                        embedding = np.fromstring(emb_str, sep=' ')
                    except: embedding = emb_str
                    vector2 = embedding.reshape(1, -1)
                    similarity_scores = cosine_similarity(vector1, vector2)[0][0] * 100
                    if similarity_scores >= THRESHOLD: 
                        found_feature = True
                        break
                if found_feature == False:
                    feature_list.append(feature_can)
                    new_row = {'Feature': feature_can, 'embedding': feature_can_embeddings}
                    df_features.loc[len(df_features)] = new_row
                    df_features.to_csv(feature_path, index = False)
                    unique_feature = list(df_features["Feature"])


print("Total features in this category: " + str(len(list(set(all_features)))))   
print("Combined features: " + str(len(feature_list)))

# Total features in this category: 1228
# Combined features: 942