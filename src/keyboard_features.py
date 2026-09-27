from pynput import keyboard
import time
import numpy as np
import threading

key_press_times = {}
key_hold_times = []
key_intervals = []
key_sequence = []

last_key_time = None
lock = threading.Lock()


def on_press(key):
    global last_key_time

    now = time.time()

    with lock:
        if key not in key_press_times:
            key_press_times[key] = now

        if last_key_time is not None:
            key_intervals.append(now - last_key_time)

        last_key_time = now
        key_sequence.append(key)


def on_release(key):
    now = time.time()

    with lock:
        if key in key_press_times:
            key_hold_times.append(
                now - key_press_times.pop(key)
            )


def start_keyboard_listener():
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )

    listener.start()

    return listener


def calculate_features():

    with lock:
        keys = key_sequence.copy()
        holds = key_hold_times.copy()
        intervals = key_intervals.copy()

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

    # Ignore very long gaps caused by thinking/not typing
    typing_intervals = [
        x for x in intervals
        if x < 2.0
    ]

    pauses = [
        x for x in intervals
        if 1.0 <= x < 5.0
    ]

    typing_speed_wpm = len(keys) / 5

    mean_key_hold_ms = (
        np.mean(holds) * 1000
        if holds else 0
    )

    mean_interkey_interval_ms = (
        np.mean(typing_intervals) * 1000
        if typing_intervals else 0
    )

    interkey_interval_std = (
        np.std(typing_intervals) * 1000
        if typing_intervals else 0
    )

    typing_pause_count = len(pauses)

    mean_pause_duration_ms = (
        np.mean(pauses) * 1000
        if pauses else 0
    )

    backspaces = sum(
        1 for key in keys
        if key == keyboard.Key.backspace
    )

    backspace_rate = backspaces / len(keys)

    typing_variability = (
        np.std(typing_intervals) / np.mean(typing_intervals)
        if typing_intervals and np.mean(typing_intervals) > 0
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


if __name__ == "__main__":

    print("Keyboard listener started.")
    print("Type for 10 seconds...")

    listener = start_keyboard_listener()

    time.sleep(10)

    features = calculate_features()

    listener.stop()

    print("\n--- Keyboard Features ---")

    for key, value in features.items():
        print(f"{key}: {value:.4f}")