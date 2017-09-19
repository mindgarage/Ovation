import spacy
import os
import numpy as np

from tflearn.data_utils import to_categorical


class SentiWordNet():
    def __init__(self, file_path):
        self.file_path = file_path
        self.descriptions = []
        self.words = {}

    def load(self):
        if not os.path.isfile(self.file_path):
            print("ERROR: File not found")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i in f.readlines():
                if (i.startswith('#') or i.startswith('\t')):
                    continue

                self.parse_row(i)

    def parse_row(self, row):
        part_of_speech, synset_id, positivity_str, negativity_str, \
                                synset_terms, gloss = row.split('\t')

        synonyms = synset_terms.split()

        # Concepts of Positivity, Negativity and Objectivity are described in
        # the SentiWordNet papers
        positivity = float(positivity_str)
        negativity = float(negativity_str)
        objectivity = float(1 - (positivity + negativity))
        synset_desc = (part_of_speech,
                            int(synset_id),
                            positivity,
                            negativity,
                            objectivity,
                            gloss,
                            synonyms)
        self.descriptions.append(synset_desc)

        # I have no way to disambiguate the words based on their senses.
        # For now, I am keeping an ID of the sense
        for term in synonyms:
            word, sense_id = term.split('#')
            curr_sense = (sense_id, len(self.descriptions)-1)
            if word not in self.words.keys():
                self.words[word] = [curr_sense]
            else:
                self.words[word].append(curr_sense)

    def get_sentiment(self, token):
        if token.lemma not in self.words:
            return 0

        descriptions = self.descriptions[token.lemma]

        total_sentiment = 0
        for i in descriptions:
            curr_sentiment = i[2] - i[3]
            total_sentiment += curr_sentiment
        final_sentiment = total_sentiment / len(descriptions)
        return final_sentiment


senti_wordnet = None
def get_senti_wordnet():
    if senti_wordnet is None:
        ret = SentiWordNet(SENTI_WORDNET_PATH)
        ret.load()
        return ret
    return senti_wordnet


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

def augment_word_vector(nlp, sentence):
    all_augmented_word_vectors = []

    doc = nlp(sentence)
    for i in doc:
        pos = i.pos_
        tag = i.tag_
        entity = i.ent_type_

        one_hot_pos = to_categorical(pos, len(pos_list))
        one_hot_tag = to_categorical(pos, len(tag_list))
        one_hot_entity = to_categorical(pos, len(entity_list))

        word_vector = i.vector
        sentiment = senti_wordnet.get_sentiment(i)

        augmented_word_vector = np.concatenate([word_vector, one_hot_pos,
                                                one_hot_tag, one_hot_entity,
                                                sentiment])

        all_augmented_word_vectors.append(augmented_word_vector)
    return all_augmented_word_vectors

