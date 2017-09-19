import spacy
from tflearn.data_utils import to_categorical


pos_list = [
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

tag_list = []
entity_list = []

def augment_word_vector(nlp, sentence):
    augmented_word_vectors = []

    doc = nlp(sentence)
    for i in doc:
        pos = i.pos_
        tag = i.tag_
        entity = i.ent_type_
        one_hot_pos = to_categorical()


    return augmented_word_vectors

