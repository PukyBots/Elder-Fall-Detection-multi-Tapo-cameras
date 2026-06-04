import cv2
import mediapipe as mp
import time
import math

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

def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x) -
        math.atan2(a.y - b.y, a.x - b.x)
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


def classify_posture(landmarks):

    # Shoulders
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Hips
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Knees
    lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]

    # Ankles
    la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # Torso center
    shoulder_x = (ls.x + rs.x) / 2
    shoulder_y = (ls.y + rs.y) / 2

    hip_x = (lh.x + rh.x) / 2
    hip_y = (lh.y + rh.y) / 2

    dx = shoulder_x - hip_x
    dy = shoulder_y - hip_y

    torso_angle = abs(math.degrees(math.atan2(dy, dx)))

    # Knee angles
    left_knee_angle = calculate_angle(lh, lk, la)
    right_knee_angle = calculate_angle(rh, rk, ra)

    knee_angle = (left_knee_angle + right_knee_angle) / 2

    # Classification

    if torso_angle < 35:
        posture = "LYING"

    elif knee_angle < 140:
        posture = "SITTING"

    else:
        posture = "STANDING"

    return posture, torso_angle, knee_angle

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

    # Resize for speed
    frame = cv2.resize(frame, (960, 540))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    if results.pose_landmarks:

        # Draw skeleton
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        # Posture classification
        posture, torso_angle, knee_angle = classify_posture(landmarks)

        # Nose coordinates
        h, w, _ = frame.shape

        nose = landmarks[mp_pose.PoseLandmark.NOSE]

        x = int(nose.x * w)
        y = int(nose.y * h)

        cv2.putText(
            frame,
            f"POSTURE: {posture}",
            (20,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,0),
            3
        )

        cv2.putText(
            frame,
            f"Torso: {torso_angle:.1f}",
            (20,160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Knee: {knee_angle:.1f}",
            (20,200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,0),
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

    cv2.imshow("MediaPipe Posture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()