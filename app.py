import time
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import firebase_admin
from firebase_admin import credentials, db
import speech_recognition as sr

# ==================================================
# FIREBASE SETUP
# ==================================================
cred = credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://sesaura-7f414-default-rtdb.firebaseio.com/"
})

sound_ref = db.reference("sound")
mode_ref = db.reference("mode")
speech_ref = db.reference("speech")
vibrate_ref = db.reference("vibrate")

# ==================================================
# AUDIO CONFIG
# ==================================================
SAMPLE_RATE = 16000
RECORD_SECONDS = 1.0
CONFIDENCE_THRESHOLD = 0.75

# ==================================================
# LOAD YAMNET
# ==================================================
print("Loading YAMNet...")
yamnet_model = hub.load("C:/Users/ASUS/yamnet_model")
# ==================================================
# LOAD CUSTOM CLASSIFIER
# ==================================================
interpreter = tf.lite.Interpreter(
    model_path="yamnet_sound_classifier.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==================================================
# CLASS INDEX
# ==================================================
class_names = [
    "Scream",   # 0
    "Silence",  # 1
    "alarm",    # 2
    "bark",     # 3
    "crying",   # 4
    "glass",    # 5
    "knock",    # 6
    "noise"     # 7
]

# ==================================================
# RECORD AUDIO
# ==================================================
def record_audio():
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32
    )
    sd.wait()
    return audio.flatten()

# ==================================================
# RESET SOUND FLAGS
# ==================================================
def reset_sound_flags():
    sound_ref.update({
        "baby": False,
        "dog": False,
        "fire": False,
        "glass": False,
        "knock": False,
        "scream": False
    })
    vibrate_ref.set(False)

# ==================================================
# UPDATE FIREBASE
# ==================================================
def update_firebase(label):
    reset_sound_flags()

    mapping = {
        "crying": "baby",
        "alarm": "fire",
        "Scream": "scream",
        "bark": "dog",
        "glass": "glass",
        "knock": "knock"
    }

    if label in mapping:
        sound_ref.child(mapping[label]).set(True)
        vibrate_ref.set(True)

# ==================================================
# SPEECH RECOGNITION (EVERY 5 SECONDS)
# ==================================================
recognizer = sr.Recognizer()
microphone = sr.Microphone()

def speech_mode_loop():
    print("🎙 Speech recognition mode started")

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

    while True:
        mode = mode_ref.get()
        if mode != 1:
            print("⬅ Exit speech mode")
            break

        try:
            with microphone as source:
                print("Listening...")
                audio = recognizer.listen(
                    source,
                    phrase_time_limit=4
                )

            text = recognizer.recognize_google(audio)
            print("Speech:", text)
            speech_ref.set(text)

        except sr.UnknownValueError:
            print("No clear speech")
        except Exception as e:
            print("Speech error:", e)

        time.sleep(5)

# ==================================================
# MAIN LOOP
# ==================================================
print("✅ SesAura AI System Running")

while True:
    try:
        mode = mode_ref.get()

        # ========== MODE 0 : SOUND DETECTION ==========
        if mode == 0:
            waveform = record_audio()

            scores, embeddings, _ = yamnet_model(waveform)

            embedding = tf.reduce_mean(embeddings, axis=0).numpy()

            interpreter.set_tensor(
                input_details[0]["index"],
                embedding.reshape(1, -1)
            )

            interpreter.invoke()

            predictions = interpreter.get_tensor(
                output_details[0]["index"]
            )[0]

            index = int(np.argmax(predictions))
            confidence = float(predictions[index])
            label = class_names[index]

            if confidence >= CONFIDENCE_THRESHOLD and label not in ["Silence", "noise"]:
                print(f"🔊 Detected: {label} ({confidence:.2f})")
                update_firebase(label)
            else:
                reset_sound_flags()

            time.sleep(0.4)

        # ========== MODE 1 : SPEECH ==========
        elif mode == 1:
            speech_mode_loop()

        time.sleep(0.2)

    except KeyboardInterrupt:
        print("System stopped")
        break
