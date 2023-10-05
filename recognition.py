import speech_recognition as sr
import noisereduce as nr
import soundfile as sf
import numpy as np


class Speech:
    def __init__(self, path: str):
        self.path = path
        self.r = sr.Recognizer()

    def to_text(self):
        data = sr.AudioFile(self.path)
        with data as sound:
            audio = self.r.record(sound)

        return self.r.recognize_google(audio_data=audio, language='ko-KR')

    def reduce_noise(self):
        data, rate = sf.read(self.path)
        reduced_noise = np.empty(shape=(0, 0))
        for i in range(0, len(data), rate):
            np.concatenate(reduced_noise, nr.reduce_noise(y=data[i:i + rate], sr=rate))
        return reduced_noise
