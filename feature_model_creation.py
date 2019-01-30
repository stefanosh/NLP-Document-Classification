from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
import string
import numpy as np
import os
from pathlib import Path
from pprint import pprint
import time
from scipy import sparse, io
from scipy.sparse import csr_matrix
import random
import itertools
import pandas as pd

start_time = time.time()

documentsDic = {}
directory_in_str = str(Path(__file__).parent) + '/20news-bydate-preprocessed/20news-bydate-train'
directory = os.fsencode(directory_in_str)

for folder in os.listdir(directory_in_str):
        for file in os.listdir(directory_in_str+"/"+folder):
            filename = directory_in_str+"/"+folder+"/"+file
            with open(filename, encoding="utf8", errors='replace') as inputfile:
                data = inputfile.read()
                documentsDic[file] = data
                               
tfidf = TfidfVectorizer()
vector = tfidf.fit_transform(documentsDic.values()).toarray()

# List of all stems of all words in all files
feature_names = tfidf.get_feature_names()

df = pd.DataFrame(vector, index=documentsDic, columns=feature_names)

max_columns_df = df.max().to_frame().T
sorted_df = max_columns_df.sort_values(by=0, ascending=False, axis=1)
print(sorted_df)

allCols = sorted_df.columns.tolist()
selectedCols = []
for columnName in allCols[0:8000]:
    selectedCols.append(columnName)

print(selectedCols)
    
print("---Total execution time in minutes: %s ---" % ((time.time() - start_time)/60))