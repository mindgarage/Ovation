# FAQs

## Q: pyaudio aborts with error: no default output device
You have to compile portaudio with alsa support. On ubuntu you have to install the libasound-dev package, before you compile ![portaudio](http://www.portaudio.com/archives/pa_stable_v190600_20161030.tgz).

