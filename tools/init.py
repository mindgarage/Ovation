import argparse
import intent_classify
from prompt_toolkit import prompt
import Server

if __name__ == '__main__':
    ranking = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices = ['cli', 'web'],
                        help='"cli" for command line, "web" for web interface', required='True')
    parser.add_argument('--ranking', type=bool, choices=[True, False],
                        help='"True" for getting the rank for every intent, '
                             '"False for getting just the response corresponding to the predicted intent')
    args = parser.parse_args()
    if args.mode == "cli":
        while True:
            sentence = prompt(u'>')
            if sentence == "exit":
                break
            ranking = False
            if args.ranking:
                ranking = True

            ovation_intent = intent_classify.intent_classify(ranking=ranking,
                                                     test_input=sentence,
                                                     model_type='blstm')

            rasa_intent = intent_classify.intent_classify(ranking=ranking,
                                                     test_input=sentence,
                                                     model_type='rasa')

            print ("ovation> " + intent_classify.get_response(ovation_intent))
            print ("rasa> "    + intent_classify.get_response(rasa_intent))

    if args.mode == "web":
        Server.init()