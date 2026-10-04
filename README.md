# ✋ AI Finger – AirDraw

**Draw and solve maths in the air with just your index finger – no mouse, no touch, only a webcam.**

AirDraw tracks your hand in real time using **MediaPipe** and **OpenCV**. Raise one finger to draw on the screen, close your fist to open a menu, and turn on the built-in **air calculator** to write a sum like `2 + 2` and get the answer instantly.

<!-- Add a demo GIF or screenshot here -->
<!-- ![Demo](demo.gif) -->

---

## ✨ Features

- 🖐️ **Hand-gesture control** – real-time hand tracking (21 landmarks) via MediaPipe
- ✏️ **Air drawing** – 7 colours, adjustable brush size, smooth strokes
- 🧽 **Eraser**, **Undo / Redo**, **Clear** and **Save as PNG**
- 🧮 **Air calculator** – write `+ − × ÷` and decimals with your finger, the answer appears automatically
  - Handwritten digits recognised with a small SVM trained at start-up (scikit-learn)
  - Options: Solve, Auto ON/OFF, Clear Calc, Pen size, Exit Calc
- 🔀 **Draw ON / Paused toggle** – open hand → fist switches between drawing and menu selection
- 👆 **Touch-free buttons** – point at a button for ~0.4 s to click it

---

## 🎮 How to use

| Gesture | What it does |
|---|---|
| ☝️ **1 finger** (index) | Draw |
| ✌️ **2 fingers** (index + middle) | Lift the pen – move without drawing |
| 🖐️ → ✊ **Open hand, then fist** | Toggle **Draw ON** ⇄ **Draw Paused (menu)** |
| 👆 **Point at a button** (when paused) | Select colour / Eraser / Calc / Undo / Redo / Clear / Save / Size |

### 🧮 Using the calculator
1. Turn on **Calc** (button or press `K`).
2. Write a sum in the air, e.g. `2 + 2` (use 2 fingers to lift the pen between strokes).
3. Stop for a moment – the result appears at the top: `2 + 2 = 4`.

**Tips:** write big and clear, leave a small gap between symbols, and use good lighting.

### ⌨️ Keyboard shortcuts

| Key | Action |
|---|---|
| `1`–`7` | Colours |
| `E` | Eraser |
| `K` | Calculator on/off |
| `C` | Clear |
| `U` / `R` | Undo / Redo |
| `S` | Save as PNG |
| `M` | Draw ON / Paused |
| `+` / `-` | Size |
| `Q` | Quit |

---

## 🛠️ Installation

**Requirements:** Python 3.9+ and a webcam.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python airdraw.py
```

On Linux / macOS you can also run `bash setup_unix.sh` to set everything up.

> The hand-landmark model (`hand_landmarker.task`, ~7 MB) is downloaded automatically on the first run if it is not already in the folder.

### Dependencies
`opencv-python` · `numpy` · `mediapipe` · `scikit-learn`

---

## 📁 Project structure

```
AI Finger/
├── airdraw.py              # main application
├── hand_landmarker.task    # MediaPipe hand model
├── requirements.txt        # Python dependencies
├── setup_unix.sh           # setup script (Linux / macOS)
└── .gitignore
```

---

## ⚙️ Troubleshooting

- **Low FPS / laggy lines** – improve lighting, close other apps using the camera, or lower the resolution (`FRAME_W, FRAME_H` at the top of `airdraw.py`, e.g. `640, 480`).
- **Wrong camera** – change `CAM_INDEX` in `airdraw.py`.
- **Calculator not answering** – make sure **Calc is ON** (button shows "Calc ON"), write larger, and lift the pen with 2 fingers between strokes.

---

## 🚀 Future ideas

- Brackets, powers and square roots in the calculator
- Shape recognition (circle, square, line)
- Two-hand gestures (zoom / move canvas)
- Export drawing as video

---

## 🤝 Contributing

Issues and pull requests are welcome. Feel free to fork the project and improve it!

## 📄 License

Released under the [MIT License](LICENSE).

---

⭐ If you like this project, give it a star on GitHub!
