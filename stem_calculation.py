import os
from pathlib import Path
import re
import time
import nltk
from nltk.stem import PorterStemmer
start_time = time.time()
ps = PorterStemmer()
closedClassCategoriesTuple = ("CD", "CC", "DT", "EX", "IN", "LS", "MD", "PDT",
                              "POS", "PRP",  "PRP",  "RP",   "TO", "UH", "WDT", "WP", "WP", "WRB")

if not os.path.exists('20news-bydate-preprocessed'):
    os.mkdir('20news-bydate-preprocessed')

directory_in_str = str(Path(__file__).parent) + '/20news-bydate'
directory = os.fsencode(directory_in_str)

count = 0
for folder in os.listdir(directory_in_str):
    if not os.path.exists('20news-bydate-preprocessed/' + folder):
        os.mkdir('20news-bydate-preprocessed/' + folder)

    for nested_folder in os.listdir(directory_in_str+"/"+folder):
        if not os.path.exists('20news-bydate-preprocessed/' + folder + "/" + nested_folder):
            os.mkdir('20news-bydate-preprocessed/' +
                     folder + "/" + nested_folder)
        count = 0
        for file in os.listdir(directory_in_str+"/"+folder+"/"+nested_folder):
            if folder == '20news-bydate-test':
                limit = 100
            else:
                limit = 800
            if count == limit:
                break
            else:
                filename = directory_in_str+"/"+folder+"/"+nested_folder+"/"+file
                with open(filename, encoding="utf8", errors='replace') as inputfile:
                    data = inputfile.read()
                    data = re.sub(r'[^a-zA-Z0-9]', " ", data).lower()
                    tokens = nltk.word_tokenize(data)
                    tags = nltk.pos_tag(tokens)
                    final_text = ""
                    for tag in tags:
                        if tag[1] not in closedClassCategoriesTuple:
                            final_text += ps.stem(tag[0]) + " "
                    count += 1
                    with open('20news-bydate-preprocessed/' + folder + "/" + nested_folder+"/" + file, 'w') as outfile:
                        outfile.write(final_text)

print("---Total execution time in minutes: %s ---" %
      ((time.time() - start_time)/60))
