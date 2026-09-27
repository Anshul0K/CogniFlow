from pynput import keyboard
import time
import numpy as np
import joblib
import threading


# -------------------------
# Load model
# -------------------------

model = joblib.load("models/cognitive_load_model.pkl")

labels = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH"
}


# -------------------------
# Keyboard data
# -------------------------

key_press_times = {}
key_hold_times = []
key_intervals = []
key_sequence = []

last_key_time = None

lock = threading.Lock()
running = True


# -------------------------
# Key press
# -------------------------

def on_press(key):

    global last_key_time

    now = time.time()

    with lock:

        if key not in key_press_times:
            key_press_times[key] = now

        if last_key_time is not None:
            key_intervals.append(
                now - last_key_time
            )

        last_key_time = now
        key_sequence.append(key)


# -------------------------
# Key release
# -------------------------

def on_release(key):

    global running

    now = time.time()

    with lock:

        if key in key_press_times:
            key_hold_times.append(
                now - key_press_times.pop(key)
            )

    if key == keyboard.Key.esc:
        running = False
        return False


# -------------------------
# Calculate features
# -------------------------

def calculate_features():

    with lock:

        keys = key_sequence.copy()
        holds = key_hold_times.copy()
        intervals = key_intervals.copy()

        # Reset current window
        key_sequence.clear()
        key_hold_times.clear()
        key_intervals.clear()

    if not keys:
        return {
            "typing_speed_wpm": 0,
            "mean_key_hold_ms": 0,
            "mean_interkey_interval_ms": 0,
            "interkey_interval_std": 0,
            "typing_pause_count": 0,
            "mean_pause_duration_ms": 0,
            "backspace_rate": 0,
            "typing_variability": 0
        }

    # Typing speed

    typing_speed_wpm = len(keys) / 5 / (10 / 60)

    # Key hold

    mean_key_hold_ms = (
        np.mean(holds) * 1000
        if holds else 0
    )

    # Inter-key intervals

    mean_interkey_interval_ms = (
        np.mean(intervals) * 1000
        if intervals else 0
    )

    interkey_interval_std = (
        np.std(intervals) * 1000
        if intervals else 0
    )

    # Pauses

    pauses = [
        x for x in intervals
        if x > 1.0
    ]

    typing_pause_count = len(pauses)

    mean_pause_duration_ms = (
        np.mean(pauses) * 1000
        if pauses else 0
    )

    # Backspaces

    backspaces = sum(
        1 for key in keys
        if key == keyboard.Key.backspace
    )

    backspace_rate = (
        backspaces / len(keys)
    )

    # Typing variability

    typing_variability = (
        np.std(intervals) / np.mean(intervals)
        if intervals and np.mean(intervals) > 0
        else 0
    )

    return {
        "typing_speed_wpm": typing_speed_wpm,
        "mean_key_hold_ms": mean_key_hold_ms,
        "mean_interkey_interval_ms": mean_interkey_interval_ms,
        "interkey_interval_std": interkey_interval_std,
        "typing_pause_count": typing_pause_count,
        "mean_pause_duration_ms": mean_pause_duration_ms,
        "backspace_rate": backspace_rate,
        "typing_variability": typing_variability
    }


# -------------------------
# Main
# -------------------------

print("Keyboard tracking started.")
print("Type normally.")
print("Features will update every 10 seconds.")
print("Press ESC to stop.\n")


listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
)

listener.start()


while running:

    time.sleep(10)

    if not running:
        break

    features = calculate_features()

    print("\n==============================")
    print("LIVE KEYBOARD FEATURES")
    print("==============================")

    for key, value in features.items():
        print(f"{key}: {value:.4f}")

    # -------------------------
    # Temporary model prediction
    # -------------------------
    #
    # Webcam + interaction values
    # are still temporary.
    #

    values = [[

        # Webcam
        15,
        0.40,
        0.05,
        0.002,
        0.002,
        0.00001,

        # Keyboard
        features["typing_speed_wpm"],
        features["mean_key_hold_ms"],
        features["mean_interkey_interval_ms"],
        features["interkey_interval_std"],
        features["typing_pause_count"],
        features["mean_pause_duration_ms"],
        features["backspace_rate"],
        features["typing_variability"],

        # Interaction
        5,
        10,
        2,
        0.5,
        5,
        20,
        0.5
    ]]

    prediction = model.predict(values)[0]

    cognitive_load = labels[prediction]

    print("\nCognitive Load:", cognitive_load)
    print("==============================")