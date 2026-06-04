import winsound
import time

print("Playing WAV alarm...")

winsound.PlaySound(
    "C:\\Windows\\Media\\Alarm01.wav",
    winsound.SND_FILENAME
)

print("Done")