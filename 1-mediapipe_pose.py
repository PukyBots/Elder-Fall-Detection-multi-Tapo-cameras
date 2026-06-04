import cv2
import mediapipe as mp
import time

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

    # Optional resize for better FPS
    frame = cv2.resize(frame, (960, 540))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # Example: Nose coordinates
        h, w, _ = frame.shape

        nose = results.pose_landmarks.landmark[
            mp_pose.PoseLandmark.NOSE
        ]

        x = int(nose.x * w)
        y = int(nose.y * h)

        cv2.putText(
            frame,
            f"Nose: ({x},{y})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow("MediaPipe Pose - Tapo Camera", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()