# Adapted from:
# https://pythonprogramminglanguage.com/text-to-speech/

import pyttsx3

if __name__ == '__main__':
    # Initializes the engine
    engine = pyttsx3.init()

    # Says somethings
    engine.say('I like coconut.')

    # Produce audio
    engine.runAndWait()

