#!/usr/bin/env python3

# Adapted from:
# https://github.com/Uberi/speech_recognition/blob/master/examples/microphone_recognition.py

import argparse

# NOTE: this example requires PyAudio because it uses the Microphone class
import speech_recognition as sr

def parse_args():
    parser = argparse.ArgumentParser(
                description="A simple Speech to Text engine")
    parser.add_argument('engine', type=str, default="Google",
                    help='Engine used to recognize speech.')
    return parser.parse_args()

class SpeechRecognizer:
    def __init__(self, engine):
        self.engine = engine
        self.recognizer = sr.Recognizer()
        self.all_engines = {
            'sphinx' : self.sphinx_engine,
            'google_sr' : self.google_sr_engine,
            'google_cloud' : self.google_cloud_engine,
            'wit' : self.wit_engine,
            'ms_bing' : self.ms_bing_engine,
            'houndify' : self.houndify_engine,
            'watson' : self.watson_engine,
        }

    def recognize(self):
        # obtain audio from the microphone
        with sr.Microphone() as source:
            print("Say something!")
            audio = self.recognizer.listen(source)
            text = self.process_audio(audio)
        return text

    def sphinx_engine(self, audio):
        try:
            text = self.recognizer.recognize_sphinx(audio)
            print("Sphinx thinks you said: " + text)
        except sr.UnknownValueError:
            print("Sphinx could not understand audio")
        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))
        return text

    def google_sr_engine(self, audio):
        # recognize speech using Google Speech Recognition
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            text = self.recognizer.recognize_google(audio)
            print("Google Speech Recognition thinks you said " + text)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return text

    def google_cloud_engine(self, audio):
        # recognize speech using Google Cloud Speech
        GOOGLE_CLOUD_SPEECH_CREDENTIALS = r"""INSERT THE CONTENTS OF THE GOOGLE CLOUD SPEECH JSON CREDENTIALS FILE HERE"""
        try:
            text = self.recognizer.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
            print("Google Cloud Speech thinks you said " + text)
        except sr.UnknownValueError:
            print("Google Cloud Speech could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Cloud Speech service; {0}".format(e))
        return text

    def wit_engine(self, audio):
        # recognize speech using Wit.ai
        WIT_AI_KEY = "INSERT WIT.AI API KEY HERE"  # Wit.ai keys are 32-character uppercase alphanumeric strings
        try:
            text = self.recognizer.recognize_wit(audio, key=WIT_AI_KEY)
            print("Wit.ai thinks you said " + text)
        except sr.UnknownValueError:
            print("Wit.ai could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Wit.ai service; {0}".format(e))
        return text


    def ms_bing_engine(self, audio):
        # recognize speech using Microsoft Bing Voice Recognition
        BING_KEY = "INSERT BING API KEY HERE"  # Microsoft Bing Voice Recognition API keys 32-character lowercase hexadecimal strings
        try:
            text = self.recognizer.recognize_bing(audio, key=BING_KEY)
            print("Microsoft Bing Voice Recognition thinks you said " + text)
        except sr.UnknownValueError:
            print("Microsoft Bing Voice Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Microsoft Bing Voice Recognition service; {0}".format(e))
        return text


    def houndify_engine(self, audio):
        # recognize speech using Houndify
        HOUNDIFY_CLIENT_ID = "INSERT HOUNDIFY CLIENT ID HERE"  # Houndify client IDs are Base64-encoded strings
        HOUNDIFY_CLIENT_KEY = "INSERT HOUNDIFY CLIENT KEY HERE"  # Houndify client keys are Base64-encoded strings
        try:
            text = self.recognizer.recognize_houndify(audio, client_id=HOUNDIFY_CLIENT_ID, client_key=HOUNDIFY_CLIENT_KEY)
            print("Houndify thinks you said " + text)
        except sr.UnknownValueError:
            print("Houndify could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Houndify service; {0}".format(e))
        return text


    def watson_engine(self, audio):
        # recognize speech using IBM Speech to Text
        IBM_USERNAME = "INSERT IBM SPEECH TO TEXT USERNAME HERE"  # IBM Speech to Text usernames are strings of the form XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
        IBM_PASSWORD = "INSERT IBM SPEECH TO TEXT PASSWORD HERE"  # IBM Speech to Text passwords are mixed-case alphanumeric strings
        try:
            text = self.recognizer.recognize_ibm(audio, username=IBM_USERNAME, password=IBM_PASSWORD)
            print("IBM Speech to Text thinks you said " + text)
        except sr.UnknownValueError:
            print("IBM Speech to Text could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from IBM Speech to Text service; {0}".format(e))
        return text

    def process_audio(self, audio):
        engine = self.all_engines.get(self.engine, None)
        if engine is not None:
            text = engine(audio)
        return text

if __name__ == '__main__':
    args = parse_args()
    s = SpeechRecognizer(args.engine)
    s.recognize()

