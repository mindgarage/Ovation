import csv
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tf_plugger import *
import rasa_intent
import numpy as np


model_name = "siamese"
# softmax_op = np.mean
# def_th = 0.003
softmax_op = np.max
def_th = 0.0


# model_name = "normal_blstm"
# model_name = "siamese"
#model_name = "attention_blstm"
model, sess = load_model(model_name)

# softmax_op = np.mean
# def_th = 0.03

# model_name = "attention_blstm"
# softmax_op = np.max
# def_th = 0.02




extra_chars = ['?', ',', '.', "'s", "_"]

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
        first_val = list(d.values())[0]
        first_key = list(d.keys())[0]
        sentence = first_val
        if first_key not in documents.keys():
            temp = []
            temp.append(sentence)
            documents[first_key] = temp
        else:
            sentence_list = documents[first_key]
            sentence_list.append(sentence)
            documents[first_key] = sentence_list
    return documents


documents = data_prep(csv_data)


import numpy as np
import json

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_sofmax_scores(dict, op_ev = softmax_op):
    cls = []
    vals = []
    for intent in dict.keys():
        score_list = dict[intent]
        v = op_ev(score_list)

        vals.append(v)
        cls.append(intent)
    vals = softmax(np.array(vals))
    return { k : v for k,v in zip(cls, vals) }

def intent_classify(ranking=True, test_input="", thr=def_th, model_type='blstm'):
    if model_type == 'rasa':
        result = rasa_intent.predict(ranking, test_input)
        new_lst = []
        for map_one in result["intent_ranking"]:
            new_lst.append((map_one['name'], map_one['confidence']))
        res = new_lst
        sorted_res = sorted(res,  key = lambda x : x[1])
        print(sorted_res)

        if (sorted_res[-1][1] - sorted_res[-2][1]) < thr:
            return None

        if ranking:
            return str(sorted_res)
        else:
            return sorted_res[-1][0]
    elif model_type == 'blstm':
        return intent_classify_tensorflow(documents, test_input, ranking=ranking, thr=thr)


def intent_classify_tensorflow(documents, test_input, ranking=True, thr=def_th):
    dict = {}
    # print("TEST", test_input)

    for intent in documents.keys():
        # print("   INTENT", intent)
        sentence_list = documents[intent]
        for sentence in sentence_list:
            curr_sim = get_similarity(test_input, sentence, model_name, model, sess)
            curr_sim = curr_sim[0]
            # print("SIM ", sentence, curr_sim)
            if intent not in dict.keys():
                score_temp = []
                score_temp.append(curr_sim)
                dict[intent] = score_temp
            else:
                score_list = dict[intent]
                score_list.append(curr_sim)
        dict[intent] = score_list
        # print("AVERAGE", np.mean(score_list), "MAX", np.max(score_list), "MIN", np.min(score_list))

    dict = get_sofmax_scores(dict)
    res = [(intent_name, score) for intent_name, score in dict.items()]
    sorted_res = sorted(res,  key = lambda x : x[1])

    print(sorted_res)
    if (sorted_res[-1][1] - sorted_res[-2][1]) < thr:
        return None

    predicted_intent = sorted_res[-1][0]

    if ranking:
        return str(sorted_res)

    # best_val = -999.0
    # best_selected = None
    # for intent_name, score in dict.items():
    #     print(intent_name, score)
    #     if score >= best_val:
    #         best_selected = intent_name
    #         best_val = score
    #         # print("BEST", best_selected)


    # max_score_index = np.argmax(dict.values())
    # predicted_intent = list(dict.keys())[max_score_index]
    #
    # if ranking:
    #     result = {}
    #     intent_ranking = []
    #     intent_dict = {}
    #     intent_dict['confidence'] = dict[predicted_intent]
    #     intent_dict['name'] = predicted_intent
    #     for k,v in dict.items():
    #         element = {}
    #         element['confidence'] = v
    #         element['name'] = k
    #         intent_ranking.append(element)
    #     result['intent_ranking'] = intent_ranking
    #     result['intent'] = intent_dict
    #     result['text'] = test_input
    #     return result
    #
    # predicted_intent = best_selected

    return predicted_intent

import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def precomp_dist_matrix():
    map_res = {}
    with open("val_dataset_labeled.csv", 'r') as test_f:
        for test_line in test_f:
            text, gt_intent = test_line.split('\t')
            # [ (cl_name, sim) ]
            rankings = intent_classify(test_input=text, ranking=True,
                                       thr=0.0, model_type='rasa')
            map_res[text] = eval(rankings)
    return map_res

def automatic_thr_selection(is_max = True):
    # Calculate all distances on dataset beforehand.
    map_res = precomp_dist_matrix()

    step_size = 0.001 if is_max else 0.001
    thr = np.arange(0.0, 0.99, step_size)
    t_xs, t_correct, t_wrong, t_rejected = [], [], [], []
    for t in thr:
        n_rejected = 0
        n_total = float(0)
        n_correct = 0
        n_wrong = 0
        with open("val_dataset_labeled.csv", 'r') as test_f:
            for test_line in test_f:
                text, gt_intent = test_line.split('\t')

                pred_intent = None
                ranks_now = map_res[text]
                if (ranks_now[-1][1] - ranks_now[-2][1]) >= t:
                    pred_intent = ranks_now[-1][0]


                # pred_intent = intent_classify(test_input=text, ranking=False, thr=t)
                # Get results.
                if pred_intent is None:
                    n_rejected += 1
                elif pred_intent.strip() == gt_intent.strip():
                    n_correct += 1
                else:
                    n_wrong += 1
                n_total += 1
        # Gather statistics
        t_correct.append( n_correct) #/ float(n_correct + n_wrong) )
        t_wrong.append( n_wrong) #  / n_total )
        t_rejected.append( n_rejected) # / n_total )
        t_xs.append(t)
        # Stop once thr is too big.
        if n_rejected == n_total:
            break

    # Plot.
    # print(t_xs)
    # print(t_correct)
    # print(t_rejected)
    plt.xlabel("threshold")
    plt.yticks(np.arange(0, n_total, 1.0))
    plt.xticks(np.arange(0.0, max(t_xs), 0.1))
    plt.plot(t_xs, t_correct, '--bo', label = "correct")
    plt.plot(t_xs, t_rejected,'--go', label = "rejected")
    plt.plot(t_xs, t_wrong, '--ro', label="wrong")
    plt.legend()
    plt.show()

    # plt.plot(t_xs, t_correct)
    # plt.show()
    # plt.plot(t_xs, t_rejected)
    # plt.show()

responses = {'Greetings': ['Hello!', 'Hi! How can I help You?'],
             'Goodbye': ['Bye! Have a nice day', 'Bye! It was a pleasure talking to you.'],
             'New Contract': [
                 'Thank you so much for choosing us. To make a contract, I would be needing some of your personal details.'],
             'Change Contract': ['Yes, we can change your contract. Can you tell me your customer reference ID?',
                                 'Sure, to change I need your customer ID'],
             'Inquire more': ['Yes, I can tell you about it.',
                              'Please wait, I will ask my human to respond to your query.'],
             'Claim': ['Sorry for your loss. Sure I will try to help you',
                       'I am very sorry for your loss. I will check with my human for your query.'],
             'Accept_app': ["That's great!", "Ok then, I will book the appointment for you.",
                            "Thanks, your appointment has been booked."],
             'Reject_app': ["Oh! Sure I will book your next appointment at the preferred time slot.",
                            "I am really sorry. We don't have any other free time slot"],
             'Change_app': ["Let me check if I have any free time slot for you.",
                            "I have to check whether we are free on the prposed time."]}
import random
def get_response(intent):
    print(intent)
    if intent in responses.keys():
        response_list = responses[intent]
        response_index = random.randint(0,len(response_list)-1)
        return response_list[response_index]
    else:
        return "No intent found!!"



    # automatic_thr_selection()

# #
# with open("test_dataset_labeled.csv", 'r') as test_f:
#     ground_truths = []
#     predictions = []
#     for test_line in test_f:
#         text, gt_intent = test_line.split('\t')
#         pred_intent = intent_classify(test_input=text, ranking=False, thr = 0.0, model_type='rasa' )
#
#         if pred_intent is None:
#             pred_intent = "Rejected"
#         predictions.append(pred_intent.strip())
#         ground_truths.append(gt_intent.strip())
#
#         print("----------------------------------------")
#         print(text)
#         print("vvvv")
#         print("true intent : ", gt_intent, " predicted intent : ", pred_intent)
#         print("----------------------------------------")
#
#     print("MODEL", model_name)
#     print("ACCURACY", accuracy_score(ground_truths, predictions))
#     print("CFM")
#
#     # classes = list(np.unique(ground_truths)) + ["Rejected"]
#
#     classes = ["Greetings","Goodbye","New Contract","Change Contract",
#                "Accept_app", "Change_app", "Reject_app", "claim",
#                "Rejected"]
#     plt.figure()
#     plot_confusion_matrix(confusion_matrix(ground_truths, predictions, labels=classes),
#                           classes)
#     plt.show()
#     print("F1-MACRO")
#     print(f1_score(ground_truths, predictions,average='macro'))