import numpy as np
from tflearn.data_utils import to_categorical

from feature_extractors.senti_wordnet import *

tag_list = [
    ".",
    ",",
    "-LRB-",
    "-RRB-",
    "``",
    "\"\"",
    "''",
    ":",
    "$",
    "#",
    "AFX",
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "HYPH",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NIL",
    "NN",
    "NNP",
    "NNPS",
    "NNS",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "SYM",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
    "SP",
    "ADD",
    "NFP",
    "GW",
    "XX",
    "BES",
    "HVS"
]

pos_list = [
    "PUNCT",
    "SYM",
    "X",
    "ADJ",
    "VERB",
    "CONJ",
    "NUM",
    "DET",
    "ADV",
    "ADP",
    "NOUN",
    "PART",
    "PRON",
    "SPACE",
    "INTJ",
]
entity_list = [
    "PERSON",
    "NORP",
    "FACILITY",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LANGUAGE",
    "DATE",
    "TIME",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL"
]

def augment_word_vector(nlp, sentence,
                        include_pos=True,
                        include_tag=True,
                        include_entity=True,
                        include_sentiment=True):
    all_augmented_word_vectors = []

    doc = nlp(sentence)
    for i in doc:
        vector_elements = []
        if include_pos:
            pos = i.pos_
            one_hot_pos = to_categorical([pos_list.index(pos)], len(pos_list))[0]
            vector_elements.append(one_hot_pos)
        if include_tag:
            tag = i.tag_
            one_hot_tag = to_categorical([tag_list.index(tag)], len(tag_list))[0]
            vector_elements.append(one_hot_tag)
        if include_entity:
            entity = i.ent_type_
            if (entity != ''):
                one_hot_entity = to_categorical([entity_list.index(entity)],
                                                    len(entity_list))[0]
            else:
                one_hot_entity = np.zeros([len(entity_list)])
            vector_elements.append(one_hot_entity)
        if include_sentiment:
            sentiment = get_senti_wordnet().get_sentiment(i)
            vector_elements.append([sentiment])

        augmented_word_vector = np.concatenate(vector_elements)

        all_augmented_word_vectors.append(augmented_word_vector)
    return all_augmented_word_vectors

if __name__ == '__main__':
    import spacy
    nlp = spacy.load('en')
    vec = augment_word_vector(nlp, 'I like food')
    print("works...")
