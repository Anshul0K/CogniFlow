import cv2
import mediapipe as mp
import numpy as np
import time

mp_face = mp.solutions.face_mesh

# MediaPipe eye landmarks
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


cap = cv2.VideoCapture(0)

eye_values = []
blink_count = 0
last_blink = False

start_time = time.time()

prev_head_x = None
head_movements = []

prev_gaze = None
gaze_movements = []

with mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            landmarks = results.multi_face_landmarks[0].landmark

            # Approximate gaze movement using iris position
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
                gaze_movement = np.linalg.norm(gaze - prev_gaze)
                gaze_movements.append(gaze_movement)

            prev_gaze = gaze

            gaze_movement_rate = (
                np.mean(gaze_movements)
                if gaze_movements else 0
            )

            cv2.putText(
                frame,
                f"Gaze movement: {gaze_movement_rate:.5f}",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            nose_x = landmarks[1].x
            nose_y = landmarks[1].y

            if prev_head_x is not None:
                movement = np.sqrt(
                    (nose_x - prev_head_x[0]) ** 2 +
                    (nose_y - prev_head_x[1]) ** 2
                )
                head_movements.append(movement)

            prev_head_x = (nose_x, nose_y)

            head_movement_rate = np.mean(head_movements) if head_movements else 0
            head_pose_variance = np.var(head_movements) if head_movements else 0

            cv2.putText(
                frame,
                f"Head movement: {head_movement_rate:.4f}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Head variance: {head_pose_variance:.6f}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)

            eye_openness = (left_ear + right_ear) / 2
            eye_values.append(eye_openness)

            eye_openness_std = np.std(eye_values)

            cv2.putText(
                frame,
                f"Eye variability: {eye_openness_std:.3f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Simple blink detection
            is_blinking = eye_openness < 0.20

            if is_blinking and not last_blink:
                blink_count += 1

            last_blink = is_blinking

            elapsed = time.time() - start_time
            blink_rate = blink_count / max(elapsed, 1) * 60

            cv2.putText(
                frame,
                f"Eye openness: {eye_openness:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Blink rate: {blink_rate:.1f}/min",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("CogniFlow Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()