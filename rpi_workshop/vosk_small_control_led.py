import sounddevice as sd
import queue, json
from vosk import Model, KaldiRecognizer
from gpiozero import LED, Button

model = Model("/home/yuchaowang/vosk_models/vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(model, 16000)
led = LED("GPIO20")
button = Button("GPIO21")
q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(bytes(indata))

print("Hold the button, speak, then release it.")

with sd.RawInputStream(samplerate=16000, dtype="int16",
                       channels=1, callback=callback):
    while True:
        button.wait_for_press()
        print("Listening...")

        while button.is_pressed:
            if not q.empty():
                rec.AcceptWaveform(q.get())

        result = json.loads(rec.FinalResult())
        text = result.get("text", "")
        print("Heard:", text)

        if ("light" in text or "lights" in text) and "on" in text:
            led.on()
        elif text in ["lights off", "light off"]:
            led.off()
        elif text == "quit":
            break
