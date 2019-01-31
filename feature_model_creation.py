from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

# Disclaimer! Variable names train and test refer to Collection E (training set) & Collection A (test test)
# Values: tf and idf refer to Term-Frequency and  Inverted Document Frequency respectively

# Calculation of vectors for collection E
# Store all documents in a dictionary(keys are the filenames) to use it as input in tf-idf calculation below
trainDic = {}
directory_in_str = str(Path(__file__).parent) + \
    '/20news-bydate-preprocessed/20news-bydate-train'
directory = os.fsencode(directory_in_str)
for folder in os.listdir(directory_in_str):
    for file in os.listdir(directory_in_str+"/"+folder):
        filename = directory_in_str+"/"+folder+"/"+file
        with open(filename, encoding="utf8", errors='replace') as inputfile:
            data = inputfile.read()
            trainDic[folder+"/"+file] = data


# Sparse matrix containining tf-idf weights for all stems of all words in all documents
tfidf = TfidfVectorizer()
tf_idf_matrix = tfidf.fit_transform(trainDic.values()).toarray()

# Vector containing only idf's of words
idf_matrix = tfidf.idf_

# List of all stems of all words in all files
feature_names = tfidf.get_feature_names()

# Create pandas data frame from tf_idf sparse matrix calculated above
tf_idf_frame = pd.DataFrame(
    tf_idf_matrix, index=trainDic, columns=feature_names)

# Return in a single row the max value of every column
max_columns_df = tf_idf_frame.max().to_frame().T
# Sort columns
sorted_df = max_columns_df.sort_values(by=0, ascending=False, axis=1)

# Get the columns names of the largest 8000 tf-idf values
allCols = sorted_df.columns.tolist()
selectedColumns = []
for columnName in allCols[0:8000]:
    selectedColumns.append(columnName)

# Store idf values in a dictionary,only those
idf_dic = {}
index = 0
for i in feature_names:
    if i in selectedColumns:
        idf_dic[i] = idf_matrix[index]
    index += 1

# Filter data frame only with the selected columns
# CollectionE_dict contains as key the documents filename and values the tf-idfs weights for each word
train_frame = tf_idf_frame[selectedColumns]
train_sparse = sparse.csr_matrix(train_frame.values)


# Calculation of vectors for collection A
# Store all documents in a dictionary(keys are the filenames) to use it as input in tf-idf calculation below
testDic = {}
directory_in_str = str(Path(__file__).parent) + \
    '/20news-bydate-preprocessed/20news-bydate-test'
directory = os.fsencode(directory_in_str)
for folder in os.listdir(directory_in_str):
    for file in os.listdir(directory_in_str+"/"+folder):
        filename = directory_in_str+"/"+folder+"/"+file
        with open(filename, encoding="utf8", errors='replace') as inputfile:
            data = inputfile.read()
            testDic[folder+"/"+file] = data


# Sparse matrix containining tf-idf weights for all stems of all words in all documents
cv = TfidfVectorizer()
vectorizer = cv.fit(selectedColumns)
tf_idf_matrix_test = vectorizer.transform(testDic.values()).toarray()

# Vector containing only idf's of words
idf_matrix_test = vectorizer.idf_

# List of all stems of all words in all files
feature_names = vectorizer.get_feature_names()

# Store idf values in a dictionary,only those
idf_dic_test = {}
index = 0
for i in feature_names:
    idf_dic_test[i] = idf_matrix_test[index]
    index += 1

# Create pandas data frame from tf_idf sparse matrix calculated above
test_frame = pd.DataFrame(
    tf_idf_matrix_test, index=testDic, columns=feature_names)

# Divide each tf-idf  value with the idf calculated above to get tf values
for key in idf_dic_test:
    test_frame[key] = test_frame[key].divide(idf_dic_test[key])

# Calculate tf-idf weights of collection A with tf derived from that collection and idf dervided from collection E for these words
for key in idf_dic:
    test_frame[key] = test_frame[key].multiply(idf_dic[key])
test_sparse = sparse.csr_matrix(test_frame.values)


# Compare documents with similarity functions and classify each document to the category of it's most similar document
# Each Test's vector is calculated torwards each Train's vector, and maxSimilartyIndex holds the train's index which is found as the most similar with the test's vector.
maxSimilarity = []
maxSimilartyIndex = []
for i in range(0, test_frame.shape[0]):
    maxSimilarity.append(0)
    maxSimilartyIndex.append("")
    # for j in range(0, train_frame.shape[0]):
    resultt = cosine_similarity(sparse.csr_matrix(
        test_frame.iloc[i].values), train_sparse)[0]
    for j in range(0, len(resultt)):
        if (resultt[j] > maxSimilarity[i]):
            maxSimilarity[i] = resultt[j]
            maxSimilartyIndex[i] = train_frame.index[j]

print(resultt)
print(maxSimilartyIndex)
print(test_frame.index)

# pprint(result)
# pprint(result.shape)

print("---Total execution time in minutes: %s ---" %
      ((time.time() - start_time)/60))
