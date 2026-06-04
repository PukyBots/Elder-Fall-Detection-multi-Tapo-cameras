# Elder Fall Detection using Multi Tapo Cameras

A real-time elder safety monitoring system that uses multiple TP-Link Tapo IP cameras, MediaPipe Pose Estimation, fall detection algorithms, local alarm triggering, and MQTT-based remote alert transmission.

<div align="left">
<img src="images/tapo.jpg" width="400">

<div align="right">
<img src="images/mediapipe.png" width="400">
</div>


## Overview

This project is designed to monitor elderly individuals using multiple Tapo cameras placed at different locations in a room or facility.

The system:

- Streams video from multiple Tapo IP cameras using RTSP.
- Performs real-time human pose estimation using MediaPipe.
- Detects posture states such as:
  - Standing
  - Sitting
  - Lying
- Detects potential falls using posture analysis and temporal tracking.
- Triggers a local alarm when a fall event is detected.
- Publishes alert messages over MQTT.
- Allows remote monitoring from another laptop, even on a different network.

---

## Features

### Multi-Camera Monitoring

- Supports multiple Tapo cameras simultaneously.
- Combines camera feeds into a single monitoring interface.
- Easily expandable to support additional cameras.

### Pose Detection

Uses MediaPipe Pose for:

- Human skeleton detection
- Landmark extraction
- Body posture analysis

### Fall Detection Logic

The system uses a combination of:

- Torso orientation
- Body aspect ratio
- Temporal posture smoothing
- Rapid posture transitions
- Lying duration analysis

to reduce false alarms and improve detection reliability.

### Alarm System

When a fall is detected:

- An alarm is triggered on the monitoring laptop.
- Visual warning is displayed on the screen.

### MQTT Alerts

The system publishes alert messages to an MQTT broker.

Example alert:

```json
{
  "event": "fall_detected",
  "camera": "Camera 1",
  "timestamp": "2026-06-04 14:20:32"
}
```

Remote clients can subscribe and receive alerts from anywhere.

---

## System Architecture

```text
+------------------+
| Tapo Camera 1    |
+--------+---------+
         |
         |
+--------v---------+
|                  |
| Monitoring PC    |
|                  |
| MediaPipe Pose   |
| Fall Detection   |
| Alarm Trigger    |
| MQTT Publisher   |
+--------+---------+
         |
         |
         v
   MQTT Broker
         |
         |
+--------v---------+
| Remote Laptop    |
| MQTT Subscriber  |
| Alert Dashboard  |
+------------------+
```

---

## Hardware Requirements

### Cameras

- TP-Link Tapo C200
- TP-Link Tapo C210
- Any RTSP-compatible IP camera

### Computer

- Windows/Linux PC
- Raspberry Pi 4/5 (for lightweight deployments)

### Network

- Cameras and monitoring PC connected to the same Wi-Fi network.
- MQTT broker accessible from remote clients.

---

## Software Requirements

- Python 3.10+
- OpenCV
- MediaPipe
- NumPy
- Paho MQTT

Install dependencies:

```bash
pip install opencv-python mediapipe numpy paho-mqtt
```

---

## Tapo Camera Setup

Enable RTSP:

1. Open Tapo App
2. Camera Settings
3. Advanced Settings
4. Camera Account
5. Create Username and Password

RTSP format:

```text
rtsp://username:password@camera_ip:554/stream1
```

Examples:

```text
rtsp://user:pass@192.168.1.100:554/stream1
rtsp://user:pass@192.168.1.101:554/stream1
```

---

## MQTT Configuration

Example broker:

```python
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "eldercare/fall_alert"
```

Publish alerts:

```python
client.publish(
    "eldercare/fall_alert",
    "Fall detected in Camera 1"
)
```

Subscribe from another laptop:

```python
client.subscribe("eldercare/fall_alert")
```

---

## Project Structure

```text
Elder-Fall-Detection-multi-Tapo-cameras/
│
├── multi_camera_fall_detection.py
├── mqtt_publisher.py
├── mqtt_subscriber.py
├── sound_check.py
├── requirements.txt
├── README.md
└── screenshots/
```

---

## Running the System

### Start Monitoring

```bash
python multi_camera_fall_detection.py
```

### Start MQTT Subscriber on Another Laptop

```bash
python mqtt_subscriber.py
```

---

## Example Workflow

1. Tapo cameras stream RTSP feeds.
2. MediaPipe detects human pose.
3. Posture is classified.
4. Fall detection algorithm evaluates posture changes.
5. Alarm is triggered locally.
6. MQTT alert is published.
7. Remote laptop receives notification.

---

## Future Improvements

- Web dashboard
- Mobile app notifications
- WhatsApp alerts
- Telegram bot integration
- Email alerts
- Person identification
- Multi-person tracking
- Cloud-based event storage
- Recording fall event clips
- Edge deployment on Raspberry Pi

---

## Author

**Pulkit Garg**

Robotics Engineer | ROS2 | Computer Vision | Autonomous Systems

GitHub:
https://github.com/PukyBots

LinkedIn:
(Add your LinkedIn profile link)

---

## License

This project is intended for educational, research, and elder-care assistance applications.

Feel free to modify and extend it for your own projects.
