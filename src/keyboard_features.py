from pynput import keyboard
import time
import numpy as np

key_press_times = {}
key_hold_times = []
key_intervals = []
key_sequence = []

last_key_time = None
start_time = time.time()


def on_press(key):
    global last_key_time

    now = time.time()

    if key not in key_press_times:
        key_press_times[key] = now

    if last_key_time is not None:
        key_intervals.append(now - last_key_time)

    last_key_time = now
    key_sequence.append(key)


def on_release(key):
    now = time.time()

    if key in key_press_times:
        key_hold_times.append(now - key_press_times.pop(key))

    if key == keyboard.Key.esc:
        return False


print("Start typing... Press ESC to stop.")

with keyboard.Listener(
    on_press=on_press,
    on_release=on_release
) as listener:
    listener.join()


# -------------------------
# Feature calculation
# -------------------------

duration = time.time() - start_time

# Typing speed
typing_speed_wpm = (len(key_sequence) / 5) / (duration / 60)

# Key hold time
mean_key_hold_ms = np.mean(key_hold_times) * 1000

# Inter-key intervals
mean_interkey_interval_ms = np.mean(key_intervals) * 1000
interkey_interval_std = np.std(key_intervals) * 1000

# Typing pauses (> 1 second)
pauses = [x for x in key_intervals if x > 1.0]

typing_pause_count = len(pauses)

mean_pause_duration_ms = (
    np.mean(pauses) * 1000
    if pauses else 0
)

# Backspace rate
backspaces = sum(
    1 for key in key_sequence
    if key == keyboard.Key.backspace
)

backspace_rate = backspaces / len(key_sequence)

# Typing variability
typing_variability = np.std(key_intervals) / np.mean(key_intervals)


print("\n--- Keyboard Features ---")

print(f"Typing speed: {typing_speed_wpm:.2f} WPM")
print(f"Mean key hold: {mean_key_hold_ms:.2f} ms")
print(f"Mean inter-key interval: {mean_interkey_interval_ms:.2f} ms")
print(f"Inter-key interval std: {interkey_interval_std:.2f} ms")
print(f"Typing pauses: {typing_pause_count}")
print(f"Mean pause duration: {mean_pause_duration_ms:.2f} ms")
print(f"Backspace rate: {backspace_rate:.3f}")
print(f"Typing variability: {typing_variability:.3f}")