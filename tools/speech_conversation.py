#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import queue
import tempfile
import wave
import pyaudio
import audioop
import math
import collections
import contextlib
import requests
import urllib
import urllib.parse as urlparse
from urllib.parse import urlencode
import json
from gtts import gTTS
import os
import time
from pydub import AudioSegment



threshold = 2.0 ** 16
volume_normalization = 0.5
keyword = "Hello"
language = "en"
greeting = "Hello"
timeout = 3

audio = pyaudio.PyAudio()

http = requests.Session()
query = urlencode({'output': 'json', 'client': 'chromium', 'key': 'AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw', 'lang': language, 'maxresults': 6, 'pfilter': 2})
request_url = urlparse.urlunparse(('https', 'www.google.com', '/speech-api/v2/recognize', '', query, ''))


@contextlib.contextmanager
def open_stream():
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    try:
        yield stream
    finally:
        stream.close()


def say(text):
    tts = gTTS(text=text, lang=language)
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        tmpfile = f.name
    tts.save(tmpfile)
    sound = AudioSegment.from_mp3(tmpfile)
    os.remove(tmpfile)

    with tempfile.SpooledTemporaryFile() as f:
        sound.export(f, format="wav")
        f.seek(0)
        w = wave.open(f, 'rb')
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        chunksize = 1024

        stream = audio.open(rate=rate, channels=channels, format=audio.get_format_from_width(width), output=True)
        data = w.readframes(chunksize)
        while data:
            stream.write(data)
            data = w.readframes(chunksize)

        stream.close()
        w.close()


def record():
    with open_stream() as stream:
        while True:
            try:
                frame = stream.read(1024)
            except IOError as e:
                if type(e.errno) is not int:
                    strerror, errno = e.errno, e.strerror
                else:
                    strerror, errno = e.strerror, e.errno
            else:
                yield frame


def _snr(frames):
    rms = audioop.rms(b''.join(frames), 2)
    return 20.0 * math.log(rms / threshold, 10)


@contextlib.contextmanager
def write_frames_to_file(frames, volume=None):
    with tempfile.NamedTemporaryFile(suffix='.wav', mode='w+b') as f:
        wav_fp = wave.open(f, 'wb')
        wav_fp.setnchannels(1)
        wav_fp.setsampwidth(2)
        wav_fp.setframerate(16000)
        fragment = b''.join(frames)
        if volume is not None:
            maxvolume = audioop.minmax(fragment, 2)[1]
            fragment = audioop.mul(fragment, 2, volume * (2. ** 15) / maxvolume)
        wav_fp.writeframes(fragment)
        wav_fp.close()
        f.seek(0)
        yield f



def transcribe(fp):
    data = fp.read()
    headers = {'content-type': 'audio/l16; rate=16000'}
    r = http.post(request_url, data=data, headers=headers)
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError:
        print('Request failed with http status ' + str(r.status_code))
        return []
    r.encoding = 'utf-8'
    try:
        response = json.loads(list(r.text.strip().split('\n', 1))[-1])
        if len(response['result']) == 0:
            raise ValueError('Nothing has been transcribed.')
        results = [alt['transcript'] for alt in response['result'][0]['alternative']]
    except ValueError as e:
        results = []
    except (KeyError, IndexError):
        print('Cannot parse response.')
        results = []
    else:
        results = tuple(result.upper() for result in results)
    return results


def check_for_keyword(frames):
    with write_frames_to_file(frames, volume_normalization) as f:
        transcribed = transcribe(f)
        keyword_found = False
        if keyword.upper() in ", ".join(transcribed):
            keyword_found = True
            say(greeting)
        return keyword_found


def init_threshold():
    global threshold
    frames = collections.deque([], 30)
    for frame in record():
        frames.append(frame)
        if len(frames) >= 16:
            threshold = float(audioop.rms(b''.join(frames), 2))
            break
    print("initialized")


def passive_listen():
    global threshold
    frame_queue = queue.Queue()
    frames = collections.deque([], 30)
    recording = False
    recording_frames = []
    for frame in record():
        frames.append(frame)
        if not recording:
            snr = _snr([frame])
            if snr >= 10:
                recording = True
                recording_frames = list(frames)[-10:]
            elif len(frames) >= frames.maxlen:
                threshold = float(audioop.rms(b''.join(frames), 2))
        else:
            recording_frames.append(frame)
            if len(recording_frames) >= 30:
                threshold = float(audioop.rms(b''.join(frames), 2))
                if check_for_keyword(recording_frames):
                    return
                recording = False


def active_listen():
    n = 16 * timeout
    frames = []
    for frame in record():
        frames.append(frame)
        if len(frames) >= 2 * n or (len(frames) > n and _snr(frames[-n:]) <= 3):
            break
    with write_frames_to_file(frames, volume_normalization) as f:
        return transcribe(f)


def act(transcription):
    for alt in transcription:        
#        if u"test" in alt:
#            print("test")
#            url = u"http://"
#            response = urllib.urlopen(url)
#            say(response)
#            break
        print(transcription) 
        say(transcription[0].lower())
        break



if __name__ == "__main__":
    init_threshold()

    while True:
        passive_listen()
        transcription = active_listen()
        if transcription:
            act(transcription)



