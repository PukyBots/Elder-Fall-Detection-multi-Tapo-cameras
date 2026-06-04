import cv2
import mediapipe as mp
import numpy as np
import time
import math
import winsound
import threading
from collections import deque

import paho.mqtt.client as mqtt
import json
import time

client = mqtt.Client()
client.connect("broker.hivemq.com", 1883, 60)
client.loop_start()

# ==================================================
# RTSP CAMERAS
# ==================================================

RTSP_URLS = [

    "rtsp://pulkitgarg:pulkitgarg@192.168.247.32:554/stream1",

    "rtsp://pulkitgargtce:pulkitgargtce@192.168.255.233:554/stream2",

    # Add more cameras here
]

# ==================================================
# MEDIAPIPE
# ==================================================

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def send_mqtt_alert(cam_id, event):
    payload = {
        "camera": cam_id,
        "event": event,
        "time": time.time()
    }

    client.publish(
        "eldercare/alert",
        json.dumps(payload)
    )

    print("MQTT SENT:", payload)

# ==================================================
# OPEN CAMERAS
# ==================================================

caps = []

for url in RTSP_URLS:

    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Cannot open:", url)
        exit()

    caps.append(cap)

# ==================================================
# CAMERA STATES
# ==================================================

camera_states = []

for _ in RTSP_URLS:

    camera_states.append({

        "prev_nose_y": None,

        "rapid_drop": False,

        "lying_start": None,

        "fall_detected": False,

        "posture_history": deque(maxlen=20),

        "sit_triggered": False
    })

# ==================================================
# ALARM
# ==================================================

alarm_running = False

def ring_alarm():

    global alarm_running

    if alarm_running:
        return

    alarm_running = True

    end_time = time.time() + 3

    while time.time() < end_time:

        winsound.Beep(2000, 500)

    alarm_running = False

# ==================================================
# POSTURE
# ==================================================

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

    if torso_angle < 35 and aspect_ratio < 1.0:

        posture = "LYING"

    elif aspect_ratio > 2.0:

        posture = "STANDING"

    else:

        posture = "SITTING"

    return posture, torso_angle, aspect_ratio

# ==================================================
# FPS
# ==================================================

prev_time = time.time()

# ==================================================
# MAIN LOOP
# ==================================================

while True:

    frames = []

    current_time = time.time()

    for cam_index, cap in enumerate(caps):

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (640, 360))

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = pose.process(rgb)

        state = camera_states[cam_index]

        if results.pose_landmarks:

            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            landmarks = results.pose_landmarks.landmark

            posture, torso_angle, aspect_ratio = \
                classify_posture(landmarks)

            # =====================================
            # TEMPORAL SMOOTHING
            # =====================================

            state["posture_history"].append(
                posture
            )

            stable_posture = max(
                set(state["posture_history"]),
                key=state["posture_history"].count
            )

            # =====================================
            # NOSE TRACKING
            # =====================================

            nose = landmarks[
                mp_pose.PoseLandmark.NOSE
            ]

            nose_y = nose.y

            rapid_drop = False

            if state["prev_nose_y"] is not None:

                delta_y = (
                    nose_y -
                    state["prev_nose_y"]
                )

                if delta_y > 0.08:

                    rapid_drop = True

            state["prev_nose_y"] = nose_y

            if rapid_drop:
                state["rapid_drop"] = True

            # =====================================
            # FALL DETECTION
            # =====================================

            if stable_posture == "LYING":

                if state["lying_start"] is None:

                    state["lying_start"] = \
                        current_time

                lying_duration = \
                    current_time - \
                    state["lying_start"]

                if (
                    lying_duration > 2
                    and
                    state["rapid_drop"]
                ):

                    state["fall_detected"] = True

            else:

                state["lying_start"] = None

                if stable_posture == "STANDING":

                    state["rapid_drop"] = False

                    state["fall_detected"] = False

            # =====================================
            # SITTING ALARM
            # =====================================

            if stable_posture == "SITTING":

                send_mqtt_alert(cam_index, "SITTING")

                if not state["sit_triggered"]:

                    state["sit_triggered"] = True

                    threading.Thread(
                        target=ring_alarm,
                        daemon=True
                    ).start()

            else:

                state["sit_triggered"] = False

            # =====================================
            # DISPLAY
            # =====================================

            cv2.putText(
                frame,
                f"CAM {cam_index+1}",
                (20,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                f"POSTURE: {stable_posture}",
                (20,70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2
            )

            cv2.putText(
                frame,
                f"TORSO: {torso_angle:.1f}",
                (20,110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2
            )

            cv2.putText(
                frame,
                f"ASPECT: {aspect_ratio:.2f}",
                (20,150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2
            )

            cv2.putText(
                frame,
                f"DROP: {state['rapid_drop']}",
                (20,190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2
            )

            if state["fall_detected"]:

                cv2.putText(
                    frame,
                    "FALL DETECTED",
                    (20,250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3
                )

        frames.append(frame)

    # =====================================
    # FPS
    # =====================================

    fps = 1 / (time.time() - prev_time)

    prev_time = time.time()

    for frame in frames:

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20,330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    # =====================================
    # COMBINE DISPLAY
    # =====================================

    if len(frames) == 1:

        combined = frames[0]

    elif len(frames) == 2:

        combined = np.hstack(frames)

    else:

        rows = []

        for i in range(0, len(frames), 2):

            if i + 1 < len(frames):

                row = np.hstack(
                    [frames[i], frames[i+1]]
                )

            else:

                blank = np.zeros_like(frames[i])

                row = np.hstack(
                    [frames[i], blank]
                )

            rows.append(row)

        combined = np.vstack(rows)

    cv2.imshow(
        "Multi Camera Fall Detection",
        combined
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==================================================
# CLEANUP
# ==================================================

for cap in caps:
    cap.release()

cv2.destroyAllWindows()