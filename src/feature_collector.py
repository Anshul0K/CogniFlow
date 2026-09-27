import time
import threading

from src.webcam_features import get_webcam_features
from src.keyboard_features import (
    calculate_features,
    start_keyboard_listener
)


class FeatureCollector:

    def __init__(self):
        self.webcam_features = {}
        self.keyboard_features = {}

        self.lock = threading.Lock()

    def collect_webcam(self):

        while True:

            features = get_webcam_features(10)

            with self.lock:
                self.webcam_features = features

    def collect_keyboard(self):

        start_keyboard_listener()

        while True:

            time.sleep(10)

            features = calculate_features()

            with self.lock:
                self.keyboard_features = features

    def get_features(self):

        with self.lock:

            features = {
                **self.webcam_features,
                **self.keyboard_features
            }

        return features

    def start(self):

        webcam_thread = threading.Thread(
            target=self.collect_webcam,
            daemon=True
        )

        keyboard_thread = threading.Thread(
            target=self.collect_keyboard,
            daemon=True
        )

        webcam_thread.start()
        keyboard_thread.start()

        print("CogniFlow feature collection started.")

        while True:

            time.sleep(10)

            features = self.get_features()

            print("\n==============================")
            print("CURRENT FEATURES")
            print("==============================")

            for key, value in features.items():
                print(f"{key}: {value:.4f}")

            print("==============================")


if __name__ == "__main__":

    collector = FeatureCollector()
    collector.start()