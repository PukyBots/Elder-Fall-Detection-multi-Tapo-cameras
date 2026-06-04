import cv2
import mediapipe as mp
import time
import numpy as np

# ==========================
# TAPO RTSP URL
# ==========================

cam1 = cv2.VideoCapture(
    "rtsp://pulkitgarg:pulkitgarg@192.168.247.32:554/stream1"
)

cam2 = cv2.VideoCapture(
    "rtsp://pulkitgargtce:pulkitgargtce@192.168.255.233:554/stream2"
)

if not cam1.isOpened():
    print("Cannot open Camera 1")
    exit()

if not cam2.isOpened():
    print("Cannot open Camera 2")
    exit()

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
# VIDEO
# ==========================

prev_time = time.time()

while True:

    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1:
        print("Camera 1 frame missing")
        break

    if not ret2:
        print("Camera 2 frame missing")
        break

    frame1 = cv2.resize(frame1, (640, 360))
    frame2 = cv2.resize(frame2, (640, 360))

    rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    results1 = pose.process(rgb1)

    if results1.pose_landmarks:

        mp_draw.draw_landmarks(
            frame1,
            results1.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # Example: Nose coordinates
        h, w, _ = frame1.shape

        nose = results1.pose_landmarks.landmark[
            mp_pose.PoseLandmark.NOSE
        ]

        x = int(nose.x * w)
        y = int(nose.y * h)

        cv2.putText(
            frame1,
            f"Nose: ({x},{y})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    results2 = pose.process(rgb2)

    if results2.pose_landmarks:

        mp_draw.draw_landmarks(
            frame2,
            results2.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        h, w, _ = frame2.shape

        nose = results2.pose_landmarks.landmark[
            mp_pose.PoseLandmark.NOSE
        ]

        x = int(nose.x * w)
        y = int(nose.y * h)

        cv2.putText(
            frame2,
            f"Nose: ({x},{y})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame2,
        "CAMERA 2",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame1,
        f"FPS: {int(fps)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame2,
        f"FPS: {int(fps)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    combined = np.hstack((frame1, frame2))

    cv2.imshow("MediaPipe Pose - Multi Tapo Camera", combined)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

cam1.release()
cam2.release()
cv2.destroyAllWindows()