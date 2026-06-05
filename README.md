# 🤟 Sign Language to Text ( for Emergency Cases )

A real-time hand gesture recognition system built with MediaPipe and OpenCV that converts hand gestures into text and speech — with an emergency alert feature.

> **Student MVP Project** — Uses custom finger-state logic (no ML training required)

---

## 📸 Demo

<img width="946" height="532" alt="Screenshot 2026-06-05 131446" src="https://github.com/user-attachments/assets/5a2bdc86-8583-4254-aea8-f4d5ee8636f5" />



---

## ✨ Features

- 🖐️ Real-time hand landmark detection via webcam
- 🤙 Recognizes 11 custom gestures and maps them to words
- 📝 Builds sentences from sequential gestures
- 🔊 Speaks words and full sentences using text-to-speech
- 🚨 Emergency alert overlay for HELP / EMERGENCY gestures
- ⌨️ Simple keyboard controls (C to clear, Q to quit)

---

## 🖐️ Gesture Reference

| Gesture | Word |
|---|---|
| All 5 fingers open | HELLO |
| Closed fist | NO |
| V / Peace sign | YES |
| Thumb up | GOOD |
| Thumb down | BAD |
| Pinky only up | HELP 🚨 |
| Index only up | I NEED |
| Index + Pinky (horns) | EMERGENCY 🚨 |
| All fingers except thumb | STOP |
| Thumb + Pinky (phone) | PLEASE |
| Middle + Ring + Pinky up | WATER |
| Thumb + Index pinch | OK |

> Hold any gesture steady for **~1.8 seconds** to confirm the word.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| MediaPipe 0.10.30+ | Hand landmark detection |
| OpenCV | Webcam feed + display |
| NumPy | Coordinate math |
| pyttsx3 | Text-to-speech |

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/sign-language-ai.git
cd sign-language-ai
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install opencv-python mediapipe numpy pyttsx3
```

### 4. Run the app
```bash
python sign_language_mvp.py
```

> **Note:** On first run, the app will automatically download the `hand_landmarker.task` model file (~10MB). Just wait for it to complete.

---

## 🎮 Controls

| Key | Action |
|---|---|
| Hold gesture ~1.8s | Add word to sentence |
| `C` | Clear current sentence |
| `Q` | Quit the app |

After **3 seconds** of no gesture, the full sentence is spoken aloud automatically.

---

## 📁 Project Structure

```
sign-language-ai/
├── sign_language_mvp.py      # Main application (single file)
├── hand_landmarker.task      # Auto-downloaded on first run
├── .gitignore
└── README.md
```

---

## ⚠️ Disclaimer

The gestures in this project are **custom finger-state shortcuts**, not standard ASL (American Sign Language). They were designed for easy detection using basic landmark logic without any machine learning training.

---

## 🚀 Future Improvements

- [ ] Add real ASL gesture support with motion tracking
- [ ] Build a web version using MediaPipe.js + Web Speech API
- [ ] Support two-hand detection
- [ ] Add more vocabulary words

---

## 👩‍💻 Author

**Sangini** — [github.com/sanginiv11](https://github.com/sanginiv11)
