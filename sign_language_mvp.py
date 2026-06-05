"""
Sign Language to Text + Emergency Message System (MVP)
MediaPipe 0.10.30+ compatible — no framework.formats import needed.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pyttsx3
import time
import threading
import urllib.request
import os

# ─── TTS ─────────────────────────────────────────────────────────────────────
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)

def speak(text):
    def _speak():
        tts_engine.say(text)
        tts_engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# ─── Download model ───────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task (~10MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

# ─── MediaPipe Tasks setup ────────────────────────────────────────────────────
options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# ─── Finger helpers ───────────────────────────────────────────────────────────
TIPS = [4, 8, 12, 16, 20]
PIPS = [3, 6, 10, 14, 18]

def finger_states(lm, hand_label):
    states = []
    if hand_label == "Right":
        states.append(lm[4].x < lm[3].x)
    else:
        states.append(lm[4].x > lm[3].x)
    for tip, pip in zip(TIPS[1:], PIPS[1:]):
        states.append(lm[tip].y < lm[pip].y)
    return states

def dist(a, b):
    return np.hypot(a.x - b.x, a.y - b.y)

# ─── Gesture recognition ──────────────────────────────────────────────────────
def recognize_gesture(lm, hand_label):
    s = finger_states(lm, hand_label)
    th, idx, mid, rng, pnk = s

    if all(s):                                              return "HELLO"
    if not any(s):                                          return "NO"
    if not th and idx and mid and not rng and not pnk:      return "YES"
    if th and not idx and not mid and not rng and not pnk:  return "GOOD"
    if not th and not idx and not mid and not rng and pnk:  return "HELP"
    if not th and idx and not mid and not rng and not pnk:  return "I NEED"
    if not th and idx and not mid and not rng and pnk:      return "EMERGENCY"
    if not th and idx and mid and rng and pnk:              return "STOP"
    if th and not idx and not mid and not rng and pnk:      return "PLEASE"
    if not th and not idx and mid and rng and pnk:          return "WATER"
    if dist(lm[4], lm[8]) < 0.05 and mid and rng and pnk:  return "OK"

    if hand_label == "Right":
        thumb_down = lm[4].x > lm[3].x and lm[4].y > lm[0].y
    else:
        thumb_down = lm[4].x < lm[3].x and lm[4].y > lm[0].y
    if not idx and not mid and not rng and not pnk and thumb_down:
        return "BAD"

    return None

# ─── Draw landmarks manually (no framework.formats needed) ───────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def draw_landmarks(frame, lm_list):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 100, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (0, 200, 255), -1)

# ─── Sentence state ───────────────────────────────────────────────────────────
sentence           = []
last_word          = None
WORD_HOLD_SEC      = 1.8
WORD_GAP_SEC       = 3.0
current_gesture    = None
gesture_start_time = 0
sentence_spoken    = False
last_gesture_time  = 0
EMERGENCY_WORDS    = {"HELP", "EMERGENCY"}

# ─── UI helpers ───────────────────────────────────────────────────────────────
def put_text(img, text, pos, scale=0.8, color=(255,255,255), thickness=2, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    if bg:
        cv2.rectangle(img, (x-4, y-th-6), (x+tw+4, y+6), (0,0,0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_emergency_overlay(img):
    overlay = img.copy()
    cv2.rectangle(overlay, (0,0), (img.shape[1], img.shape[0]), (0,0,200), -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    text = "! EMERGENCY ALERT !"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1.6, 4
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(img, text,
                ((img.shape[1]-tw)//2, (img.shape[0]+th)//2),
                font, scale, (255,255,255), thick, cv2.LINE_AA)

def draw_hold_bar(img, progress):
    bar_w = int(img.shape[1] * 0.6)
    x = (img.shape[1] - bar_w) // 2
    y = img.shape[0] - 40
    cv2.rectangle(img, (x, y), (x+bar_w, y+12), (60,60,60), -1)
    filled = int(bar_w * min(progress, 1.0))
    color  = (0,220,100) if progress < 0.99 else (0,255,0)
    cv2.rectangle(img, (x, y), (x+filled, y+12), color, -1)
    cv2.rectangle(img, (x, y), (x+bar_w, y+12), (180,180,180), 1)

# ─── Main loop ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Sign Language MVP started. Press Q to quit, C to clear sentence.")

timestamp_ms = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    now   = time.time()

    rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms += 33
    detection    = detector.detect_for_video(mp_image, timestamp_ms)

    detected_word = None

    if detection.hand_landmarks:
        lm         = detection.hand_landmarks[0]
        hand_label = detection.handedness[0][0].category_name
        draw_landmarks(frame, lm)
        detected_word = recognize_gesture(lm, hand_label)

    # Gesture hold logic
    if detected_word:
        last_gesture_time = now
        sentence_spoken   = False
        if detected_word != current_gesture:
            current_gesture    = detected_word
            gesture_start_time = now
        else:
            held_for = now - gesture_start_time
            draw_hold_bar(frame, held_for / WORD_HOLD_SEC)
            if held_for >= WORD_HOLD_SEC and detected_word != last_word:
                sentence.append(detected_word)
                last_word = detected_word
                speak("EMERGENCY ALERT! " + detected_word
                      if detected_word in EMERGENCY_WORDS else detected_word)
    else:
        current_gesture = None
        if (sentence and not sentence_spoken
                and (now - last_gesture_time) > WORD_GAP_SEC):
            speak(" ".join(sentence))
            sentence_spoken = True

    # Emergency overlay
    if any(word in EMERGENCY_WORDS for word in sentence[-3:]):
        draw_emergency_overlay(frame)

    # HUD
    if detected_word:
        color = (0,80,255) if detected_word in EMERGENCY_WORDS else (0,230,120)
        put_text(frame, f"Gesture: {detected_word}", (20, 50), 1.0, color)
    else:
        put_text(frame, "Gesture: ---", (20, 50), 1.0, (160,160,160))

    sentence_str = " ".join(sentence) if sentence else "(no sentence yet)"
    cv2.rectangle(frame, (0, h-90), (w, h), (20,20,20), -1)
    put_text(frame, f"Sentence: {sentence_str}", (16, h-55), 0.75, (255,220,80), bg=False)
    put_text(frame, "Hold gesture to add word  |  C = clear  |  Q = quit",
             (16, h-22), 0.5, (160,160,160), 1, bg=False)

    cv2.imshow("Sign Language MVP", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        sentence.clear()
        last_word       = None
        sentence_spoken = False

cap.release()
cv2.destroyAllWindows()
