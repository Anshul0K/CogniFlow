import cv2
import mediapipe as mp
import numpy as np
import time
import joblib

mp_face = mp.solutions.face_mesh


model = joblib.load("models/cognitive_load_model.pkl")

labels = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH"
}

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(landmarks, eye):
    p1 = np.array([landmarks[eye[1]].x, landmarks[eye[1]].y])
    p2 = np.array([landmarks[eye[5]].x, landmarks[eye[5]].y])

    p3 = np.array([landmarks[eye[2]].x, landmarks[eye[2]].y])
    p4 = np.array([landmarks[eye[4]].x, landmarks[eye[4]].y])

    p5 = np.array([landmarks[eye[0]].x, landmarks[eye[0]].y])
    p6 = np.array([landmarks[eye[3]].x, landmarks[eye[3]].y])

    return (np.linalg.norm(p1 - p2) + np.linalg.norm(p3 - p4)) / (
        2 * np.linalg.norm(p5 - p6)
    )


def get_webcam_features():

    cap = cv2.VideoCapture(0)

    eye_values = []
    blink_count = 0
    last_blink = False

    head_movements = []
    gaze_movements = []

    prev_head = None
    prev_gaze = None

    window_start = time.time()

    with mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Camera error")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:

                landmarks = results.multi_face_landmarks[0].landmark

                # ---------------- GAZE ----------------

                left_iris = np.array([
                    landmarks[468].x,
                    landmarks[468].y
                ])

                right_iris = np.array([
                    landmarks[473].x,
                    landmarks[473].y
                ])

                gaze = (left_iris + right_iris) / 2

                if prev_gaze is not None:
                    gaze_movements.append(
                        np.linalg.norm(gaze - prev_gaze)
                    )

                prev_gaze = gaze

                # ---------------- HEAD ----------------

                head = np.array([
                    landmarks[1].x,
                    landmarks[1].y
                ])

                if prev_head is not None:
                    head_movements.append(
                        np.linalg.norm(head - prev_head)
                    )

                prev_head = head

                # ---------------- EYES ----------------

                left_ear = eye_aspect_ratio(
                    landmarks,
                    LEFT_EYE
                )

                right_ear = eye_aspect_ratio(
                    landmarks,
                    RIGHT_EYE
                )

                eye_openness = (
                    left_ear + right_ear
                ) / 2

                eye_values.append(eye_openness)

                # ---------------- BLINK ----------------

                is_blinking = eye_openness < 0.20

                if is_blinking and not last_blink:
                    blink_count += 1

                last_blink = is_blinking

            # ---------------- TIMER ----------------

            elapsed = time.time() - window_start

            remaining = max(0, 10 - elapsed)

            cv2.putText(
                frame,
                f"Next analysis: {remaining:.1f}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "CogniFlow Webcam",
                frame
            )

            # ---------------- EVERY 10 SECONDS ----------------

            if elapsed >= 10:

                features = {
                    "blink_rate_per_min":
                        blink_count / elapsed * 60,

                    "mean_eye_openness":
                        np.mean(eye_values)
                        if eye_values else 0,

                    "eye_openness_std":
                        np.std(eye_values)
                        if eye_values else 0,

                    "gaze_movement_rate":
                        np.mean(gaze_movements)
                        if gaze_movements else 0,

                    "head_movement_rate":
                        np.mean(head_movements)
                        if head_movements else 0,

                    "head_pose_variance":
                        np.var(head_movements)
                        if head_movements else 0
                }



                values = [[
                    features["blink_rate_per_min"],
                    features["mean_eye_openness"],
                    features["eye_openness_std"],
                    features["gaze_movement_rate"],
                    features["head_movement_rate"],
                    features["head_pose_variance"],

                    # temporary keyboard values
                    45,
                    70,
                    200,
                    100,
                    2,
                    1500,
                    0.05,
                    1.0,

                    # temporary interaction values
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

                print("Cognitive Load:", cognitive_load)

                print("\n==============================")
                print("LIVE WEBCAM FEATURES")
                print("==============================")

                for key, value in features.items():
                    print(f"{key}: {value:.6f}")

                print("==============================\n")

                # Reset window

                eye_values = []
                blink_count = 0
                last_blink = False

                head_movements = []
                gaze_movements = []

                prev_head = None
                prev_gaze = None

                window_start = time.time()

            # ---------------- QUIT ----------------

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    get_webcam_features()