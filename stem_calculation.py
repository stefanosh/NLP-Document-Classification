import os

directory_in_str = "/Users/heikki/Desktop/20news-bydate/20news-bydate-train"
directory = os.fsencode(directory_in_str)

count = 0 
for folder in os.listdir(directory_in_str):
    for file in os.listdir(directory_in_str+"/"+folder):
        count +=1

print (count)

""" with open('data.txt', 'r') as myfile:
    data=myfile.read() """