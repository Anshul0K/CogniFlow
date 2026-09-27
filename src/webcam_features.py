import cv2
import mediapipe as mp
import numpy as np
import time

mp_face = mp.solutions.face_mesh

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


def get_webcam_features(duration=10):

    cap = cv2.VideoCapture(0)

    eye_values = []
    blink_count = 0
    last_blink = False

    head_movements = []
    gaze_movements = []

    prev_head = None
    prev_gaze = None

    start_time = time.time()

    with mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while time.time() - start_time < duration:

            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:

                landmarks = results.multi_face_landmarks[0].landmark

                # Gaze

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

                # Head movement

                head = np.array([
                    landmarks[1].x,
                    landmarks[1].y
                ])

                if prev_head is not None:
                    head_movements.append(
                        np.linalg.norm(head - prev_head)
                    )

                prev_head = head

                # Eye openness

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

                # Blink

                is_blinking = eye_openness < 0.20

                if is_blinking and not last_blink:
                    blink_count += 1

                last_blink = is_blinking

    cap.release()

    elapsed = time.time() - start_time

    return {
        "blink_rate_per_min":
            blink_count / max(elapsed, 1) * 60,

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


if __name__ == "__main__":

    features = get_webcam_features()

    print("\n--- Webcam Features ---")

    for key, value in features.items():
        print(f"{key}: {value:.6f}")