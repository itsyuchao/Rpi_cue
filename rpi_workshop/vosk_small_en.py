import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer

model = Model("/home/yuchaowang/vosk_models/vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(model, 16000)

print("Speak now...")

# Record 3 seconds of audio
audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype="int16")
sd.wait()

# Process the audio with Vosk
if rec.AcceptWaveform(audio.tobytes()):
    result = json.loads(rec.Result())
    print("You said:", result["text"])
