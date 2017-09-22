import csv
import json

data = []
with open("intent_dataset.csv", "r") as file:
    csv_data = csv.reader(file, delimiter=',')
    for row in csv_data:
        data.append(row)

def getcsv_data(raw_data):
    classes = []
    data = []
    cl = ''
    for j in range(len(raw_data)):
        sentence_dict = {}
        row = raw_data[j]
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


def getJson(data):
    rasa_nlu = {}
    element_nlu = {}
    common_examples = []
    for d in data:
        for k,v in d.iteritems():
            common_example = {}
            common_example['text'] = v
            common_example['intent'] = k
            common_example['entities'] = []
        common_examples.append(common_example)
    element_nlu['common_examples'] = common_examples
    rasa_nlu["rasa_nlu_data"] = element_nlu
    return rasa_nlu

processed_data, classes = getcsv_data(data)
json_data = getJson(processed_data)

with open("train_dataset.json", "wb") as outfile:
    json.dump(json_data, outfile)