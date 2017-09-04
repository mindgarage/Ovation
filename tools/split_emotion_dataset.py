import os
import csv

X = []
Y= []
classes = None
with open('data/emotion_text.txt') as csvDataFile:
    csvReader = csv.reader(csvDataFile, delimiter='\t')
    for row in csvReader:
        print(row)