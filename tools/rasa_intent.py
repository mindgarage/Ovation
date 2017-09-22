from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter

import sys, getopt



def rasa_train_spacy():
	training_data = load_data('data/examples/rasa/train_dataset.json')
	trainer = Trainer(RasaNLUConfig("./config_spacy.json"))
	trainer.train(training_data)
	model_directory = trainer.persist('./models/')
	print(model_directory)


def rasa_train_MITIE():
	training_data = load_data('data/examples/rasa/train_dataset.json')
	trainer = Trainer(RasaNLUConfig("./config_mitie.json"))
	trainer.train(training_data)
	model_directory = trainer.persist('./models/')
	print(model_directory)


def predict(getranking_FLAG, teststring):
	test_data = (teststring)
	interpreter = Interpreter.load('./models/default/spacy', RasaNLUConfig("sample_configs/config_spacy.json"))
	result = interpreter.parse(teststring)
	if(getranking_FLAG):
		print(result)
	else:
		print("Only INTENT {} With Text:  {}".format(result['intent'], result['text']))

def get_inputSentence():
	list_text = []
	temp = sys.argv[1:]
	print (temp)
	string = " ".join(temp)
	return string

def main():
	string = get_inputSentence()
	print (string)
	#print (sys.argv)
	getranking_FLAG =  False
	predict(getranking_FLAG, string)

if __name__ == "__main__": 
	main()  #rasa_train_spacy()