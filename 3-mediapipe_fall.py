import cv2
import mediapipe as mp
import time
import math
from collections import deque

# ==========================
# TAPO RTSP URL
# ==========================

RTSP_URL = "rtsp://pulkitgarg:pulkitgarg@192.168.247.32:554/stream1"

# ==========================
# MEDIAPIPE SETUP
# ==========================

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================
# FALL DETECTION VARIABLES
# ==========================

posture_history = deque(maxlen=30)

lying_start_time = None
fall_detected = False

# ==========================
# POSTURE CLASSIFICATION
# ==========================

def classify_posture(landmarks):

    xs = []
    ys = []

    for lm in landmarks:
        if lm.visibility > 0.5:
            xs.append(lm.x)
            ys.append(lm.y)

    if len(xs) < 10:
        return "UNKNOWN", 0, 0

    body_width = max(xs) - min(xs)
    body_height = max(ys) - min(ys)

    aspect_ratio = body_height / (body_width + 1e-6)

    # Torso angle

    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    shoulder_x = (ls.x + rs.x) / 2
    shoulder_y = (ls.y + rs.y) / 2

    hip_x = (lh.x + rh.x) / 2
    hip_y = (lh.y + rh.y) / 2

    dx = shoulder_x - hip_x
    dy = shoulder_y - hip_y

    torso_angle = abs(
        math.degrees(
            math.atan2(dy, dx)
        )
    )

    # Classification

    if torso_angle < 35 and aspect_ratio < 1.0:
        posture = "LYING"

    elif aspect_ratio > 2.0:
        posture = "STANDING"

    else:
        posture = "SITTING"

    return posture, torso_angle, aspect_ratio

# ==========================
# VIDEO
# ==========================

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("Cannot open RTSP stream")
    exit()

prev_time = time.time()

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to receive frame")
        break

    frame = cv2.resize(frame, (960, 540))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        posture, torso_angle, aspect_ratio = classify_posture(
            landmarks
        )

        # --------------------------
        # Temporal smoothing
        # --------------------------

        posture_history.append(posture)

        stable_posture = max(
            set(posture_history),
            key=posture_history.count
        )

        # --------------------------
        # Fall logic
        # --------------------------

        current_time = time.time()

        if stable_posture == "LYING":

            if lying_start_time is None:
                lying_start_time = current_time

            lying_duration = (
                current_time - lying_start_time
            )

            if lying_duration > 3:
                fall_detected = True

        else:

            lying_start_time = None
            fall_detected = False

        # --------------------------
        # Display
        # --------------------------

        cv2.putText(
            frame,
            f"POSTURE: {stable_posture}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"Torso Angle: {torso_angle:.1f}",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Aspect Ratio: {aspect_ratio:.2f}",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        if fall_detected:

            cv2.putText(
                frame,
                "FALL DETECTED",
                (20, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                4
            )

    # ==========================
    # FPS
    # ==========================

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "MediaPipe Fall Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()