import csv
from nltk.stem.lancaster import LancasterStemmer
import numpy as np

import random

extra_chars = ['?', ',', '.', "'s", "_"]
stemmer = LancasterStemmer()

csv_data = []
with open("intent_dataset.csv", "r") as file:
    dataset = csv.reader(file, delimiter=',')
    for row in dataset:
        csv_data.append(row)


def getcsv_data():
    classes = []
    data = []
    cl = ''
    for j in range(len(csv_data)):
        sentence_dict = {}
        row = csv_data[j]
        for i in range(len(row)):
            col = row[i]
            if col:
                if i == 2:
                    if col not in classes:
                        cl = col
                        classes.append(col)
                if i == 4:
                    sentence_dict[cl] = col
        if sentence_dict:
            data.append(sentence_dict)
    return data, classes


def data_prep(csv_data):
    data, classes = getcsv_data()
    documents = {}
    for d in data:
        sentence = d.values()[0]
        if d.keys()[0] not in documents.keys():
            temp = []
            temp.append(sentence)
            documents[d.keys()[0]] = temp
        else:
            sentence_list = documents[d.keys()[0]]
            sentence_list.append(sentence)
            documents[d.keys()[0]] = sentence_list
    return documents


def get_avg_score(dict):
    res_dict = {}
    intents = dict.keys()
    for intent in intents:
        score_list = dict[intent]
        res_dict[intent] = np.mean(score_list)
    return res_dict


def intent_classify(documents, test_input='Hi', ranking=True ):
    dict = {}
    for intent in documents.keys():
        sentence_list = documents[intent]
        for sentence in sentence_list:
            curr_sim = random.uniform(1.0, 99.99)#similarity(test_input, doc[0][0])
            if intent not in dict.keys():
                score_temp = []
                score_temp.append(curr_sim)
                dict[intent] = score_temp
            else:
                score_list = dict[intent]
                score_list.append(curr_sim)
        dict[intent] = score_list
    dict = get_avg_score(dict)

    max_score_index = np.argmax(dict.values())
    predicted_intent = dict.keys()[max_score_index]

    if ranking:
        result = {}
        intent_ranking = []
        intent_dict = {}
        intent_dict['confidence'] = dict[predicted_intent]
        intent_dict['name'] = predicted_intent
        for k,v in dict.iteritems():
            element = {}
            element['confidence'] = v
            element['name'] = k
            intent_ranking.append(element)
        result['intent_ranking'] = intent_ranking
        result['intent'] = intent_dict
        result['text'] = test_input
        return result

    return predicted_intent


documents = data_prep(csv_data)
print intent_classify(documents)
