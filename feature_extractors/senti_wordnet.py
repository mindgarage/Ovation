import os

SENTI_WORDNET_PATH = '/home/gamboa/Documents/OSA-alpha/data/datasets/SentiWordNet/SentiWordNet.txt'

class SentiWordNet():
    def __init__(self):
        self.descriptions = []
        self.words = {}

        if not os.path.isfile(SENTI_WORDNET_PATH):
            print("ERROR: File not found")
        with open(SENTI_WORDNET_PATH, 'r', encoding='utf-8') as f:
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
        # `token` is a spaCy token

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
    global senti_wordnet
    if senti_wordnet is None:
        senti_wordnet = SentiWordNet()
        return senti_wordnet
    return senti_wordnet


