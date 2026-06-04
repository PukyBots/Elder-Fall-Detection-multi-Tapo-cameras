import cv2
import numpy as np
import tensorflow as tf

# ==========================
# CONFIG
# ==========================

RTSP_URL = "rtsp://pulkitgarg:pulkitgarg@192.168.247.32:554/stream1"
MODEL_PATH = "movenet_singlepose_thunder.tflite"

# ==========================
# LOAD MODEL
# ==========================

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==========================
# KEYPOINT CONNECTIONS
# ==========================

EDGES = [
    (0,1),(0,2),
    (1,3),(2,4),
    (0,5),(0,6),
    (5,7),(7,9),
    (6,8),(8,10),
    (5,6),
    (5,11),(6,12),
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16)
]

# ==========================
# POSE ESTIMATION
# ==========================

def detect_pose(frame):

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_img = cv2.resize(img, (192, 192))

    input_img = np.expand_dims(input_img, axis=0)
    input_img = input_img.astype(np.uint8)

    interpreter.set_tensor(
        input_details[0]['index'],
        input_img
    )

    interpreter.invoke()

    keypoints = interpreter.get_tensor(
        output_details[0]['index']
    )

    return keypoints[0][0]

# ==========================
# DRAW
# ==========================

def draw_pose(frame, keypoints, threshold=0.3):

    h, w, _ = frame.shape

    points = []

    for kp in keypoints:

        y, x, conf = kp

        px = int(x * w)
        py = int(y * h)

        points.append((px, py, conf))

        if conf > threshold:
            cv2.circle(frame, (px, py), 5, (0,255,0), -1)

    for p1, p2 in EDGES:

        if points[p1][2] > threshold and points[p2][2] > threshold:

            cv2.line(
                frame,
                (points[p1][0], points[p1][1]),
                (points[p2][0], points[p2][1]),
                (255,0,0),
                2
            )

# ==========================
# VIDEO
# ==========================

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("Cannot open RTSP stream")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        break

    keypoints = detect_pose(frame)

    draw_pose(frame, keypoints)

    cv2.imshow("MoveNet - Tapo Camera", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()