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
from numpy import linalg as la
import gc
start_time = time.time()

# Disclaimer! Variable names train and test refer to Collection E (training set) & Collection A (test test)
# Values: tf and idf refer to Term-Frequency and  Inverted Document Frequency respectively

# Function to calculate correct cases of  document classification
# Remove everything after '/' for both they key and its value. Goal is to compare:
# 'talk.religion.misc/84018': 'talk.religion.misc/82816' as  'talk.religion.misc' == 'talk.religion.misc'


def calculate_percentage(dic):
    success_counter = 0
    for key in dic:
        text = key
        head, sep, tail = text.partition('/')
        key_string = head
        text = dic[key]
        head, sep, tail = text.partition('/')
        value_string = head
        if key_string == value_string:
            success_counter += 1
    total = len(dic)
    return ((success_counter/total) * 100)


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


# Sparse matrix containining normalized tf-idf weights for all stems of all words in all documents
tfidf = TfidfVectorizer(min_df=0.001)
tf_idf_matrix = tfidf.fit_transform(trainDic.values()).toarray()

# Vector containing only idf's of words.Will be used later for calculation of tf-idf weights for collection A
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

# Store idf values in a dictionary but only those which belong to selectedColumns
idf_dic = {}
index = 0
for i in feature_names:
    if i in selectedColumns:
        idf_dic[i] = idf_matrix[index]
    index += 1

# Filter data frame only with the selected columns
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
# Calculate tf-idf of documents only for words which belong to selectedColumns
cv = TfidfVectorizer()
vectorizer = cv.fit(selectedColumns)
tf_idf_matrix_test = vectorizer.transform(testDic.values()).toarray()

# Vector containing only idf's of words
idf_matrix_test = vectorizer.idf_

# List of all stems of all words in all files
feature_names = vectorizer.get_feature_names()

# Store idf values in a dictionary
idf_dic_test = {}
index = 0
for i in feature_names:
    idf_dic_test[i] = idf_matrix_test[index]
    index += 1

# Create pandas data frame from tf_idf sparse matrix calculated above
test_frame = pd.DataFrame(
    tf_idf_matrix_test, index=testDic, columns=feature_names)
test_frame = test_frame[selectedColumns]

# Divide each tf-idf  value with the idf(of collection A) calculated above to get tf values
for key in idf_dic_test:
    test_frame[key] = test_frame[key].divide(idf_dic_test[key])

# Calculate tf-idf weights of collection A with tf derived from that collection and idf dervided from collection E for these words
for key in idf_dic:
    test_frame[key] = test_frame[key].multiply(idf_dic[key])

del trainDic
del testDic
del feature_names
del idf_dic
del idf_dic_test
del tf_idf_matrix_test
del allCols
del selectedColumns
del tf_idf_matrix
del idf_matrix

gc.collect()

# Compare documents with similarity functions and classify each document to the category of it's most similar document

# Cosine Similarity d(x,y) = x.y / (|x| * |y|)
# Each Test's vector is calculated torwards each Train's vector, and maxSimilartyIndex holds the train's index which is found as the most similar with the test's vector.
cosine_prediction_dic = {}
for i in range(0, test_frame.shape[0]):
    results = cosine_similarity(sparse.csr_matrix(
        test_frame.iloc[i].values), train_sparse)[0]
    sorted_indexes = np.argsort(results)
    sorted_indexes = sorted_indexes[::-1]
    cosine_prediction_dic[test_frame.index[i]
                          ] = train_frame.index[sorted_indexes[0]]

cosine_percentage = calculate_percentage(cosine_prediction_dic)
del cosine_prediction_dic
del sorted_indexes
gc.collect()

# *MAGIC HAPPENS HERE*


def tanimoto_similarity(X, Y, norms):
    K = X * Y.T
    K = K.toarray()
    return K/(la.norm(X.toarray())**2+norms-K)


# Tanimoto distance measure, d(x,y) = x.y / (|x|*|x|) + (|y|*|y|)- x*y
tanimoto_prediction_dic = {}
norms_of_train = la.norm(train_sparse.toarray(), axis=1)**2
for i in range(0, test_frame.shape[0]):
    results = tanimoto_similarity(sparse.csr_matrix(
        test_frame.iloc[i].values), train_sparse, norms_of_train)[0]
    # # Result list has indexes with the same order as in train_frame, so np.argsort is used to return
    # # the sorted indexes of result_list and in this way  the index of the maximum value can be easily found
    # # Then, we can use it to assign the respective category name from train_frame to the current document in test_frame
    sorted_indexes = np.argsort(results)
    sorted_indexes = sorted_indexes[::-1]
    tanimoto_prediction_dic[test_frame.index[i]
                            ] = train_frame.index[sorted_indexes[0]]

del train_frame
del test_frame
del train_sparse
gc.collect()

tanimoto_percentage = calculate_percentage(tanimoto_prediction_dic)

print("---Successful classification rate for cosine metric: %s" %
      cosine_percentage)
print("---Successful classification rate for tanimoto metric: %s" %
      tanimoto_percentage)
print("---Total execution time in minutes: %s ---" %
      ((time.time() - start_time)/60))
