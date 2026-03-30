import sounddevice as sd
import queue, json
from vosk import Model, KaldiRecognizer

# --- Setup ---
model = Model("/home/yuchaowang/vosk_models/vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(model, 16000)
q = queue.Queue()

# Callback saves incoming audio chunks into the queue
def callback(indata, frames, time, status):
    q.put(bytes(indata))

print("Speak into the microphone. Press Ctrl+C to stop.")

# Open continuous input stream
with sd.RawInputStream(samplerate=16000, dtype="int16",
                       channels=1, callback=callback):
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):        # Got a full phrase
            result = json.loads(rec.Result())
            print("You said:", result["text"])


