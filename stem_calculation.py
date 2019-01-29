import os
from pathlib import Path

directory_in_str = str(Path(__file__).parent) + '/20news-bydate/20news-bydate-train'
directory = os.fsencode(directory_in_str)

count = 0 
for folder in os.listdir(directory_in_str):
    for file in os.listdir(directory_in_str+"/"+folder):
        count +=1

print (count)

""" with open('data.txt', 'r') as myfile:
    data=myfile.read() """