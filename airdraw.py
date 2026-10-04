"""
AirDraw - Hand Gesture Controlled Canvas  (v2)
===============================================
Draw on a virtual canvas using nothing but your webcam and your hand.

LAYOUT
    LEFT  panel : colour palette (7 colours)
    RIGHT panel : tools  (Eraser, Calc, Undo, Redo, Clear, Save, Size -, Size +)

GESTURES  (two states: DRAW ON  /  DRAW PAUSED = menu)
    DRAW ON      : show ONE finger (index)  -> it draws
    TOGGLE       : open your whole hand, then CLOSE it into a fist (once)
                   -> drawing stops, side panels + option bars become active
    DRAW PAUSED  : point your index finger at any button for ~0.4 s -> it clicks
                   (colour, Eraser, Calc, Undo, Redo, Clear, Save, sizes...)
    TOGGLE again : open hand, then fist -> drawing is back ON
    Calc options appear at the bottom-right while drawing is paused.
    2 fingers (index + middle) = "pen up": move your hand without drawing
    (needed between the strokes of a "+" or between digits).
    Keyboard [M] does the same toggle.

ONE-CLICK BUTTONS
    * hand   : hover the fingertip on a button for ~0.4 s (progress bar shows)
    * mouse  : a single left click on any button also works
    * keys   : shortcuts below

ERASER
    Medium size (4x the brush). Size -/+ buttons change it together with the brush.

CALCULATOR  (button "Calc" or key K)
    Turn Calc on, then write a sum in the air, e.g.  2 + 2
    Pause for a moment and the answer appears next to it:  2 + 2 = 4
    Supports + - x / and decimals (.), digits can be multi-digit (12 + 35).
    Operators:  +  (cross)   -  (line)   x  (two crossing diagonals)
                /  (slanted line)  or  the divide sign (dot, line, dot)
                =  (two lines, optional - the answer is shown anyway)
    Tips: write big, keep a small gap between symbols, don't let symbols
    overlap sideways. The small yellow letters above each symbol show what
    was recognised - if one is wrong, Undo (U) or erase and rewrite it.
    The Eraser works inside Calc mode as well.
    CALC OPTIONS (appear when Calc is ON and drawing is paused):
        Solve | Auto ON/OFF | Clear Calc | Pen - | Pen + | Exit Calc

KEYBOARD
    1-7 colours | E eraser | K calc | C clear | U undo | R redo | S save
    M draw on/pause | +/- size of the active tool | Q quit

Requirements:
    pip install opencv-python mediapipe numpy scikit-learn
    (scikit-learn is only needed for the calculator)
"""

import ast
import os
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

try:
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
WINDOW = "AirDraw - Hand Gesture Controlled Canvas"

PANEL_W = 120            # width of left / right side panels
BUTTON_MARGIN = 10
BUTTON_GAP = 8
MAX_BUTTON_H = 72
BOTTOM_BAR = 40          # help bar height

COLORS = [
    ("Blue",   (255, 100, 0)),
    ("Green",  (60, 200, 60)),
    ("Red",    (50, 50, 220)),
    ("Yellow", (0, 220, 235)),
    ("Purple", (200, 40, 190)),
    ("Cyan",   (235, 220, 0)),
    ("White",  (245, 245, 245)),
]

TOOLS = [
    ("Eraser", (110, 110, 110)),
    ("Calc",   (0, 130, 240)),
    ("Undo",   (90, 90, 90)),
    ("Redo",   (90, 90, 90)),
    ("Clear",  (60, 60, 170)),
    ("Save",   (60, 140, 60)),
    ("Size -", (80, 80, 80)),
    ("Size +", (80, 80, 80)),
]
# keep firing while you hover
REPEAT_BUTTONS = ("Size -", "Size +", "calc_pen_minus", "calc_pen_plus")

MIN_BRUSH, MAX_BRUSH = 2, 40
DEFAULT_BRUSH = 8
ERASER_SIZE_MULT = 4           # eraser = brush size x 4 (medium)
MIN_CALC_PEN, MAX_CALC_PEN = 3, 20
DEFAULT_CALC_PEN = 8

# open hand -> fist toggle (draw on / draw paused)
OPEN_HOLD_FRAMES = 4      # open palm must be seen this many frames
FIST_HOLD_FRAMES = 3      # then a fist this many frames
ARM_TIMEOUT = 1.5         # seconds allowed between open palm and fist
TOGGLE_COOLDOWN = 0.8     # ignore new toggles for this long

# option bars (bottom-right of the drawing area)
OPT_BTN_W, OPT_BTN_H, OPT_TITLE_H = 104, 50, 22

SMOOTHING = 0.55          # 0=no smoothing, closer to 1 = smoother but laggier
MAX_UNDO_STACK = 20
SELECT_HOVER_TIME = 0.4   # seconds hovering on a button to "click" it
REPEAT_INTERVAL = 0.25    # auto-repeat for Size +/- while hovering
CALC_DELAY = 0.8          # seconds after you stop writing before it calculates

# --------------------------------------------------------------------------
# HAND LANDMARK MODEL (auto-download on first run)
# --------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one-time, ~7 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded to", MODEL_PATH)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]


def fingers_up(landmarks, handedness_label):
    """Returns [thumb,index,middle,ring,pinky] booleans."""
    up = [False] * 5
    if handedness_label == "Right":
        up[0] = landmarks[4].x < landmarks[3].x
    else:
        up[0] = landmarks[4].x > landmarks[3].x
    for i in range(1, 5):
        up[i] = landmarks[FINGER_TIPS[i]].y < landmarks[FINGER_PIPS[i]].y
    return up


# --------------------------------------------------------------------------
# CALCULATOR: handwriting -> expression -> answer
# --------------------------------------------------------------------------
class MathRecognizer:
    """
    Reads digits and + - x / = drawn on the calculator layer.

    Pipeline:
      1. connected components on the ink  -> strokes
      2. strokes that overlap horizontally are merged into one symbol
         (so "=" , "+" , "x" and multi-stroke digits like 4 / 5 work)
      3. each symbol is classified:
            operators  -> simple geometry rules
            digits     -> small SVM trained at startup (no download needed)
      4. the expression is evaluated safely (no eval of raw text)
    """

    FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX,
             cv2.FONT_HERSHEY_PLAIN]

    def __init__(self):
        self.ready = False
        self.error = None
        if not HAS_SKLEARN:
            self.error = "scikit-learn missing: pip install scikit-learn"
            return
        self._train()
        self.ready = True

    # ---------- digit classifier ----------
    @staticmethod
    def _features(mask):
        """mask: 2-D bool/uint8 crop of one digit -> 64-dim vector (8x8, 0..16)."""
        m = (mask > 0).astype(np.uint8) * 255
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            return np.zeros(64, dtype=np.float32)
        m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = m.shape
        side = int(max(h, w) * 8 / 6) + 2
        sq = np.zeros((side, side), np.uint8)
        oy, ox = (side - h) // 2, (side - w) // 2
        sq[oy:oy + h, ox:ox + w] = m
        small = cv2.resize(sq, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
        if small.max() > 0:
            small = small / small.max() * 16.0
        return small.reshape(-1)

    def _synthetic(self):
        rng = np.random.default_rng(0)
        X, y = [], []
        for d in range(10):
            for font in self.FONTS:
                for thick in (5, 8, 11):
                    for _ in range(14):
                        scale = 2.6 if font != cv2.FONT_HERSHEY_PLAIN else 5.0
                        img = np.zeros((160, 160), np.uint8)
                        (tw, th), base = cv2.getTextSize(str(d), font, scale, thick)
                        cv2.putText(img, str(d), ((160 - tw) // 2, (160 + th) // 2),
                                    font, scale, 255, thick, cv2.LINE_AA)
                        ang = rng.uniform(-12, 12)
                        shear = rng.uniform(-0.2, 0.2)
                        sx = rng.uniform(0.8, 1.15)
                        M = cv2.getRotationMatrix2D((80, 80), ang, 1.0)
                        M[0, 1] += shear
                        M[0, 0] *= sx
                        img = cv2.warpAffine(img, M, (160, 160))
                        X.append(self._features(img > 60))
                        y.append(d)
        return np.array(X), np.array(y)

    def _train(self):
        from sklearn.datasets import load_digits
        from sklearn.svm import SVC
        Xs, ys = self._synthetic()
        d = load_digits()
        X = np.vstack([Xs, d.data.astype(np.float32)])
        y = np.concatenate([ys, d.target])
        self.clf = SVC(kernel="rbf", gamma=0.002, C=10).fit(X, y)

    def _classify_digit(self, mask):
        return str(int(self.clf.predict([self._features(mask)])[0]))

    # ---------- operator geometry ----------
    @staticmethod
    def _cross_kind(mask):
        """Return '+' , '*' or None for a roughly square, crossing shape."""
        h, w = mask.shape
        if h == 0 or w == 0 or not (0.55 < w / h < 1.8):
            return None
        g = cv2.resize(mask.astype(np.uint8) * 255, (32, 32),
                       interpolation=cv2.INTER_AREA) > 40
        band = slice(11, 21)
        vcov = g[:, band].any(axis=1).mean()
        hcov = g[band, :].any(axis=0).mean()
        d1 = np.mean([g[max(0, i - 3):i + 4, max(0, i - 3):i + 4].any() for i in range(32)])
        d2 = np.mean([g[max(0, i - 3):i + 4, max(0, 31 - i - 3):31 - i + 4].any()
                      for i in range(32)])
        corners = np.mean([g[:8, :8].mean(), g[:8, -8:].mean(),
                           g[-8:, :8].mean(), g[-8:, -8:].mean()])
        mids = np.mean([g[:8, 12:20].mean(), g[-8:, 12:20].mean(),
                        g[12:20, :8].mean(), g[12:20, -8:].mean()])
        cmax = max(g[:8, :8].mean(), g[:8, -8:].mean(),
                   g[-8:, :8].mean(), g[-8:, -8:].mean())
        # a real "+" has its bars through the middle (a "4" has them off-centre)
        cs, rs = g.sum(axis=0), g.sum(axis=1)
        pc = int(round(np.nonzero(cs >= 0.9 * cs.max())[0].mean()))
        pr = int(round(np.nonzero(rs >= 0.9 * rs.max())[0].mean()))
        centred = abs(pc - 15.5) < 6 and abs(pr - 15.5) < 6
        quads = [g[:max(pr - 5, 1), :max(pc - 5, 1)],
                 g[:max(pr - 5, 1), pc + 6:],
                 g[pr + 6:, :max(pc - 5, 1)],
                 g[pr + 6:, pc + 6:]]
        quad_ink = max((q.mean() for q in quads if q.size), default=0.0)
        centred = centred and quad_ink < 0.1
        if min(vcov, hcov) > 0.8 and corners < 0.12 and cmax < 0.2 and centred:
            return "+"
        if min(d1, d2) > 0.8 and mids < 0.15:
            return "*"
        return None

    @staticmethod
    def _line_angle(mask):
        ys, xs = np.nonzero(mask)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        pts -= pts.mean(axis=0)
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        lineness = np.sqrt(max(vals[0], 1e-6) / max(vals[1], 1e-6))
        vx, vy = vecs[:, 1]
        ang = abs(np.degrees(np.arctan2(vy, vx)))
        if ang > 90:
            ang = 180 - ang
        return lineness, ang          # ang: 0 = horizontal, 90 = vertical

    def _classify_symbol(self, comps, ref_h):
        """comps: list of dict(mask=bool crop in group coords, bbox=(x,y,w,h))."""
        # build the group mask
        gx1 = min(c["bbox"][0] for c in comps)
        gy1 = min(c["bbox"][1] for c in comps)
        gx2 = max(c["bbox"][0] + c["bbox"][2] for c in comps)
        gy2 = max(c["bbox"][1] + c["bbox"][3] for c in comps)
        gw, gh = gx2 - gx1, gy2 - gy1
        gm = np.zeros((gh, gw), bool)
        for c in comps:
            x, y, w, h = c["bbox"]
            gm[y - gy1:y - gy1 + h, x - gx1:x - gx1 + w] |= c["mask"]

        flat = [c for c in comps if c["bbox"][2] > 2.0 * c["bbox"][3]]
        n = len(comps)

        if n == 2 and len(flat) == 2:
            return "="
        if n == 3 and len(flat) == 1:
            return "/"                               # the divide sign
        if n == 1:
            if gw > 2.2 * gh:
                return "-"
            if gw < 0.3 * ref_h and gh < 0.3 * ref_h:
                return "."
        kind = self._cross_kind(gm)
        if kind:
            return kind
        if n == 1:
            lineness, ang = self._line_angle(gm)
            if lineness < 0.2:
                if ang < 20:
                    return "-"
                if ang > 74:
                    return "1"
                return "/"
        if n > 3:
            return "?"
        return self._classify_digit(gm)

    # ---------- segmentation ----------
    def read(self, ink):
        """ink: uint8 image (0 / 255). Returns dict or None."""
        if not self.ready:
            return None
        dil = cv2.dilate(ink, np.ones((7, 7), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
        comps = []
        for i in range(1, n):
            m = (labels == i) & (ink > 0)
            if m.sum() < 25:
                continue
            ys, xs = np.nonzero(m)
            x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            comps.append(dict(bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                              mask=m[y1:y2, x1:x2]))
        if not comps:
            return None
        comps.sort(key=lambda c: c["bbox"][0])

        # merge strokes that overlap horizontally
        groups = []
        for c in comps:
            x, y, w, h = c["bbox"]
            if groups:
                g = groups[-1]
                gx1 = min(k["bbox"][0] for k in g)
                gx2 = max(k["bbox"][0] + k["bbox"][2] for k in g)
                ov = min(gx2, x + w) - max(gx1, x)
                if ov > 0.35 * min(gx2 - gx1, w):
                    g.append(c)
                    continue
            groups.append([c])

        ref_h = max(max(k["bbox"][1] + k["bbox"][3] for k in g) -
                    min(k["bbox"][1] for k in g) for g in groups)
        symbols, boxes = [], []
        for g in groups:
            symbols.append(self._classify_symbol(g, ref_h))
            x1 = min(k["bbox"][0] for k in g)
            y1 = min(k["bbox"][1] for k in g)
            x2 = max(k["bbox"][0] + k["bbox"][2] for k in g)
            y2 = max(k["bbox"][1] + k["bbox"][3] for k in g)
            boxes.append((x1, y1, x2, y2))

        text = "".join(symbols)
        value, shown = self.evaluate(text)
        bx1 = min(b[0] for b in boxes)
        by1 = min(b[1] for b in boxes)
        bx2 = max(b[2] for b in boxes)
        by2 = max(b[3] for b in boxes)
        return dict(symbols=symbols, boxes=boxes, text=text, value=value,
                    shown=shown, bbox=(bx1, by1, bx2, by2))

    # ---------- evaluation ----------
    @staticmethod
    def pretty(text):
        return text.replace("*", "x")

    @staticmethod
    def _safe_eval(node):
        if isinstance(node, ast.Expression):
            return MathRecognizer._safe_eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = MathRecognizer._safe_eval(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a = MathRecognizer._safe_eval(node.left)
            b = MathRecognizer._safe_eval(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            return a / b
        raise ValueError("bad expression")

    def evaluate(self, text):
        """Returns (value_string_or_None, expression_part_that_was_evaluated)."""
        expr = text.split("=")[0]
        expr = expr.rstrip("+-*/.")
        if not expr or "?" in expr or not any(ch.isdigit() for ch in expr):
            return None, expr
        if not any(op in expr[1:] for op in "+-*/"):
            return None, expr                    # just a number, nothing to compute
        try:
            v = self._safe_eval(ast.parse(expr, mode="eval"))
        except ZeroDivisionError:
            return "Error (/0)", expr
        except Exception:
            return None, expr
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v))), expr
        return f"{v:.4f}".rstrip("0").rstrip("."), expr


# --------------------------------------------------------------------------
# UI LAYOUT  (colours LEFT, tools RIGHT)
# --------------------------------------------------------------------------
class SidePanels:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.left_w = PANEL_W
        self.right_w = PANEL_W
        self.buttons = []
        self._column([(n, c) for n, c in COLORS], "color", 0, self.left_w)
        self._column(TOOLS, "action", self.w - self.right_w, self.w)

    def _column(self, items, kind, px1, px2):
        n = len(items)
        top = BUTTON_MARGIN
        bottom = self.h - BOTTOM_BAR - BUTTON_MARGIN
        avail = bottom - top - BUTTON_GAP * (n - 1)
        bh = min(MAX_BUTTON_H, avail // n)
        y = top
        for label, color in items:
            self.buttons.append(dict(
                label=label, color=color, kind=kind,
                x1=px1 + BUTTON_MARGIN, x2=px2 - BUTTON_MARGIN,
                y1=y, y2=y + bh))
            y += bh + BUTTON_GAP

    # drawing area = the space between the two panels
    @property
    def draw_x1(self):
        return self.left_w

    @property
    def draw_x2(self):
        return self.w - self.right_w

    def in_side_panel(self, pt):
        return pt[0] < self.left_w or pt[0] > self.w - self.right_w

    def in_draw_area(self, pt):
        return self.draw_x1 <= pt[0] <= self.draw_x2 and pt[1] < self.h - BOTTOM_BAR

    def hit_test(self, px, py):
        for b in self.buttons:
            if b["x1"] <= px <= b["x2"] and b["y1"] <= py <= b["y2"]:
                return b
        return None

    def draw(self, img, active_color_name, eraser_on, calc_on,
             hover_label=None, hover_progress=0.0, alpha=1.0):
        base = img.copy() if alpha < 1.0 else None
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.left_w, self.h - BOTTOM_BAR), (25, 25, 25), -1)
        cv2.rectangle(overlay, (self.w - self.right_w, 0), (self.w, self.h - BOTTOM_BAR),
                      (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        for b in self.buttons:
            label = b["label"]
            if b["kind"] == "color":
                is_active = (label == active_color_name)
            else:
                is_active = (label == "Eraser" and eraser_on) or (label == "Calc" and calc_on)

            fill = b["color"]
            if label == "Calc" and calc_on:
                fill = (0, 200, 255)
            text = label
            if label == "Calc":
                text = "Calc ON" if calc_on else "Calc"
            if label == "Eraser" and eraser_on:
                text = "Eraser ON"
            dark_text = label in ("Yellow", "White", "Cyan") or (label == "Calc" and calc_on)
            draw_button(img, b, is_active, fill, text, dark_text, hover_label, hover_progress)

        if base is not None:      # faded look while drawing is ON (panels are inactive)
            cv2.addWeighted(img, alpha, base, 1 - alpha, 0, img)


def draw_button(img, b, active=False, fill=None, text=None, dark_text=False,
                hover_label=None, hover_progress=0.0):
    """Draws one button (side panel or option bar) incl. the hover progress bar."""
    label = b["label"]
    cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), fill or b["color"], -1)
    border = (255, 255, 255) if active else (30, 30, 30)
    cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), border, 3 if active else 1)

    text = text or label
    tcol = (0, 0, 0) if dark_text else (255, 255, 255)
    fs = 0.55
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    tx = b["x1"] + (b["x2"] - b["x1"] - tw) // 2
    ty = b["y1"] + (b["y2"] - b["y1"] + th) // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, tcol, 1, cv2.LINE_AA)

    if hover_label == label and hover_progress > 0:
        bar_w = int((b["x2"] - b["x1"]) * min(1.0, hover_progress))
        cv2.rectangle(img, (b["x1"], b["y2"] - 5), (b["x1"] + bar_w, b["y2"]),
                      (255, 255, 255), -1)


# --------------------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------------------
class AirDraw:
    def __init__(self):
        ensure_model()

        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check CAM_INDEX / camera permissions.")

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Could not read from webcam.")
        self.h, self.w = frame.shape[:2]

        # two layers: normal drawing + calculator ink
        self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.calc_canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.panels = SidePanels(self.w, self.h)

        self.color_name = "Cyan"
        self.color = dict(COLORS)[self.color_name]
        self.brush_size = DEFAULT_BRUSH
        self.eraser_on = False
        self.calc_on = False
        self.calc_pen = DEFAULT_CALC_PEN
        self.calc_auto = True
        self._ink_pt = None

        # draw ON / PAUSED (menu) state, switched by "open hand -> fist"
        self.draw_enabled = True
        self.open_frames = 0
        self.fist_frames = 0
        self.armed_until = 0.0
        self.last_toggle = 0.0
        self.flash_msg = ""
        self.flash_color = (0, 150, 0)

        self.mode = "IDLE"
        self.prev_point = None
        self.smoothed_point = None

        self.undo_stack = deque(maxlen=MAX_UNDO_STACK)
        self.redo_stack = deque(maxlen=MAX_UNDO_STACK)
        self._push_undo()

        # button hover / click state
        self.hover_label = None
        self.hover_start_time = None
        self.hover_fired = False
        self.last_fire_time = 0.0

        # calculator state
        print("Preparing calculator..." if HAS_SKLEARN else
              "scikit-learn not found - calculator disabled (pip install scikit-learn)")
        self.recognizer = MathRecognizer()
        self.calc_result = None
        self.calc_dirty = False
        self.last_ink_time = 0.0

        base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._frame_ts = 0

        self.prev_time = time.time()
        self.fps = 0.0
        self.last_hand_seen = time.time()
        self.saved_filename = None
        self.save_flash_until = 0

    # ---------------- layers / undo / redo ----------------
    @property
    def layer(self):
        return self.calc_canvas if self.calc_on else self.canvas

    def _snapshot(self):
        return (self.canvas.copy(), self.calc_canvas.copy())

    def _restore(self, snap):
        self.canvas = snap[0].copy()
        self.calc_canvas = snap[1].copy()
        self.calc_dirty = True
        self.calc_result = None
        self.last_ink_time = time.time()

    def _push_undo(self):
        self.undo_stack.append(self._snapshot())
        self.redo_stack.clear()

    def undo(self):
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            self._restore(self.undo_stack[-1])

    def redo(self):
        if self.redo_stack:
            state = self.redo_stack.pop()
            self.undo_stack.append(state)
            self._restore(state)

    def clear(self):
        """Clears the layer you are currently on (Calc layer when Calc is on)."""
        self.layer[:] = 0
        self.calc_dirty = True
        self.calc_result = None
        self._push_undo()

    def save(self):
        fname = f"airdraw_{time.strftime('%Y%m%d_%H%M%S')}.png"
        out = self.canvas.copy()
        m = np.any(self.calc_canvas != 0, axis=2)
        out[m] = self.calc_canvas[m]
        cv2.imwrite(fname, out)
        return fname

    # ---------------- tools ----------------
    def set_color(self, name):
        self.color_name = name
        self.color = dict(COLORS)[name]
        self.eraser_on = False

    def toggle_eraser(self):
        self.eraser_on = not self.eraser_on

    def toggle_calc(self):
        if not self.recognizer.ready:
            self.saved_filename = None
            self.flash(self.recognizer.error or "Calculator unavailable")
            return
        self.calc_on = not self.calc_on
        if self.calc_on:
            self.eraser_on = False      # otherwise the writing would be erased
        self.calc_result = None
        self.calc_dirty = self.calc_on
        self.last_ink_time = time.time()
        self.end_stroke()

    def flash(self, msg, secs=2.5, color=(0, 150, 0)):
        self.flash_msg = msg
        self.flash_color = color
        self.save_flash_until = time.time() + secs

    def change_size(self, delta):
        """Size -/+ adjusts the brush (eraser follows it) or the calc pen."""
        if self.calc_on and not self.eraser_on:
            self.change_calc_pen(delta)
        else:
            self.brush_size = int(np.clip(self.brush_size + delta, MIN_BRUSH, MAX_BRUSH))

    def change_calc_pen(self, delta):
        self.calc_pen = int(np.clip(self.calc_pen + delta, MIN_CALC_PEN, MAX_CALC_PEN))

    def solve_now(self):
        if not (self.calc_on and np.any(self.calc_canvas)):
            self.flash("Write a sum first", color=(0, 120, 220))
            return
        ink = (np.max(self.calc_canvas, axis=2) > 0).astype(np.uint8) * 255
        self.calc_result = self.recognizer.read(ink)
        self.calc_dirty = False

    def clear_calc(self):
        self.calc_canvas[:] = 0
        self.calc_result = None
        self.calc_dirty = False
        self._push_undo()

    # ---------------- draw ON / PAUSED ----------------
    def set_draw_enabled(self, enabled):
        if enabled == self.draw_enabled:
            return
        self.end_stroke()
        self.draw_enabled = enabled
        self.hover_label = None
        self.hover_start_time = None
        self.hover_fired = False
        if enabled:
            self.flash("DRAW ON - show 1 finger to draw", color=(0, 150, 0))
        else:
            self.flash("DRAW PAUSED - point at an option", color=(0, 120, 220))

    def update_toggle_gesture(self, open_palm, fist):
        """open hand, then fist (once)  ->  toggle draw ON / PAUSED."""
        now = time.time()
        if open_palm:
            self.open_frames += 1
            self.fist_frames = 0
            if self.open_frames >= OPEN_HOLD_FRAMES:
                self.armed_until = now + ARM_TIMEOUT
        elif fist:
            self.fist_frames += 1
            self.open_frames = 0
            if (self.fist_frames >= FIST_HOLD_FRAMES and now < self.armed_until
                    and now - self.last_toggle > TOGGLE_COOLDOWN):
                self.set_draw_enabled(not self.draw_enabled)
                self.armed_until = 0.0
                self.last_toggle = now
        else:
            self.open_frames = 0
            self.fist_frames = 0

    def end_stroke(self):
        if self.prev_point is not None:
            self._push_undo()
        self.prev_point = None
        self.smoothed_point = None
        self._ink_pt = None

    def draw_point(self, pt):
        if self.eraser_on:
            size = self.brush_size * ERASER_SIZE_MULT
        elif self.calc_on:
            size = self.calc_pen
        else:
            size = self.brush_size
        draw_color = (0, 0, 0) if self.eraser_on else self.color
        layer = self.layer
        if self.prev_point is None:
            cv2.circle(layer, pt, max(1, size // 2), draw_color, -1)
        else:
            cv2.line(layer, self.prev_point, pt, draw_color, size)
        self.prev_point = pt
        if self.calc_on:
            moved = (self._ink_pt is None or
                     (pt[0] - self._ink_pt[0]) ** 2 + (pt[1] - self._ink_pt[1]) ** 2 > 16)
            if moved:                      # a finger held still must not delay the answer
                self._ink_pt = pt
                self.calc_dirty = True
                self.calc_result = None
                self.last_ink_time = time.time()

    def handle_button(self, label):
        if label in dict(COLORS):
            self.set_color(label)
        elif label == "Eraser":
            self.toggle_eraser()
        elif label == "Calc":
            self.toggle_calc()
        elif label == "Clear":
            self.clear()
        elif label == "Undo":
            self.undo()
        elif label == "Redo":
            self.redo()
        elif label == "Save":
            self.flash("Saved: " + self.save())
        elif label == "Size -":
            self.change_size(-2)
        elif label == "Size +":
            self.change_size(2)
        # ---- calculator options ----
        elif label == "calc_solve":
            self.solve_now()
        elif label == "calc_auto":
            self.calc_auto = not self.calc_auto
            self.calc_dirty = True
            self.last_ink_time = time.time()
        elif label == "calc_clear":
            self.clear_calc()
        elif label == "calc_pen_minus":
            self.change_calc_pen(-1)
        elif label == "calc_pen_plus":
            self.change_calc_pen(1)
        elif label == "calc_exit":
            self.toggle_calc()

    # ---------------- option bars (eraser / calc) ----------------
    def build_options(self):
        """Returns (buttons, row_panels). Empty while drawing is ON."""
        if self.draw_enabled:
            return [], []
        rows = []
        if self.calc_on:
            rows.append(("calc", f"CALCULATOR OPTIONS  (pen {self.calc_pen}px)", [
                ("calc_solve", "Solve", (60, 140, 60)),
                ("calc_auto", f"Auto: {'ON' if self.calc_auto else 'OFF'}", (0, 130, 240)),
                ("calc_clear", "Clear Calc", (60, 60, 170)),
                ("calc_pen_minus", "Pen -", (80, 80, 80)),
                ("calc_pen_plus", "Pen +", (80, 80, 80)),
                ("calc_exit", "Exit Calc", (90, 90, 90))]))
        buttons, panels = [], []
        x_right = self.panels.draw_x2 - 24
        y_bottom = self.h - BOTTOM_BAR - 14
        for kind, title, items in rows:
            y2, y1 = y_bottom, y_bottom - OPT_BTN_H
            n = len(items)
            x = x_right - (n * OPT_BTN_W + (n - 1) * BUTTON_GAP)
            panels.append(dict(kind=kind, title=title, x=x, ty=y1 - 9,
                               bx1=x - 12,
                               bx2=x_right + 12, by1=y1 - OPT_TITLE_H - 6, by2=y2 + 8))
            for label, text, color in items:
                buttons.append(dict(label=label, text=text, color=color, kind="option",
                                    x1=x, x2=x + OPT_BTN_W, y1=y1, y2=y2))
                x += OPT_BTN_W + BUTTON_GAP
            y_bottom = y1 - OPT_TITLE_H - 22
        return buttons, panels

    def hit_test(self, px, py):
        for b in self.build_options()[0]:
            if b["x1"] <= px <= b["x2"] and b["y1"] <= py <= b["y2"]:
                return b
        return self.panels.hit_test(px, py)

    def draw_options(self, frame):
        buttons, panels = self.build_options()
        for p in panels:
            overlay = frame.copy()
            cv2.rectangle(overlay, (p["bx1"], p["by1"]), (p["bx2"], p["by2"]), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            cv2.rectangle(frame, (p["bx1"], p["by1"]), (p["bx2"], p["by2"]), (0, 200, 255), 1)
            cv2.putText(frame, p["title"], (p["x"], p["ty"]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 200, 255), 1, cv2.LINE_AA)
        hp = self.hover_progress()
        for b in buttons:
            lab = b["label"]
            active = (lab == "calc_auto" and self.calc_auto)
            draw_button(frame, b, active, None, b["text"], False, self.hover_label, hp)

    # ---------------- mouse: one click on any button ----------------
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            btn = self.hit_test(x, y)
            if btn:
                self.handle_button(btn["label"])

    # ---------------- hover-to-click with the fingertip ----------------
    def update_select(self, tip):
        now = time.time()
        btn = self.hit_test(*tip) if tip else None
        if btn is None:
            self.hover_label = None
            self.hover_start_time = None
            self.hover_fired = False
            return
        if btn["label"] != self.hover_label:
            self.hover_label = btn["label"]
            self.hover_start_time = now
            self.hover_fired = False
        if not self.hover_fired:
            if now - self.hover_start_time >= SELECT_HOVER_TIME:
                self.handle_button(btn["label"])
                self.hover_fired = True
                self.last_fire_time = now
        elif btn["label"] in REPEAT_BUTTONS and now - self.last_fire_time >= REPEAT_INTERVAL:
            self.handle_button(btn["label"])
            self.last_fire_time = now

    def hover_progress(self):
        if self.hover_start_time is None:
            return 0.0
        if self.hover_fired:
            return 1.0
        return (time.time() - self.hover_start_time) / SELECT_HOVER_TIME

    # ---------------- calculator ----------------
    def update_calc(self):
        if not (self.calc_on and self.calc_dirty and self.calc_auto):
            return
        if time.time() - self.last_ink_time < CALC_DELAY:
            return
        ink = (np.max(self.calc_canvas, axis=2) > 0).astype(np.uint8) * 255
        self.calc_result = self.recognizer.read(ink)
        self.calc_dirty = False

    @staticmethod
    def _outlined_text(img, text, org, scale, color, thick):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                    thick + 4, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    thick, cv2.LINE_AA)

    def draw_calc_overlay(self, frame):
        if not self.calc_on:
            return
        r = self.calc_result
        pw, ph = 520, 62
        cx = (self.panels.draw_x1 + self.panels.draw_x2) // 2
        x1, y1 = cx - pw // 2, 12
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + pw, y1 + ph), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x1 + pw, y1 + ph), (0, 200, 255), 2)

        if r is None:
            msg = "CALC: write a sum, e.g.  2 + 2"
            col = (200, 200, 200)
            if self.calc_dirty and np.any(self.calc_canvas):
                msg = "CALC: reading..." if self.calc_auto else "CALC: tap Solve"
            cv2.putText(frame, msg, (x1 + 18, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, col, 2, cv2.LINE_AA)
            return

        expr = r["shown"]
        for op in "+-*/":
            expr = expr.replace(op, f" {op} ")
        expr = expr.replace("*", "x").replace("  ", " ").strip()
        if r["value"] is not None:
            msg = f"{expr} = {r['value']}"
            col = (120, 255, 120)
        else:
            msg = f"{expr or r['text']}  (can't solve)"
            col = (80, 180, 255)
        cv2.putText(frame, msg, (x1 + 18, y1 + 42), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, col, 2, cv2.LINE_AA)

        # small labels above each recognised symbol
        for sym, (bx1, by1, bx2, by2) in zip(r["symbols"], r["boxes"]):
            label = "x" if sym == "*" else sym
            cv2.putText(frame, label, (bx1 + (bx2 - bx1) // 2 - 6, max(100, by1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # big answer next to the expression
        if r["value"] is not None:
            bx1, by1, bx2, by2 = r["bbox"]
            scale = float(np.clip((by2 - by1) / 45.0, 1.0, 3.0))
            thick = max(2, int(2 * scale))
            text = r["value"] if "=" in r["text"] else f"= {r['value']}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            tx = bx2 + 30
            ty = (by1 + by2) // 2 + th // 2
            if tx + tw > self.panels.draw_x2 - 10:          # no room on the right
                tx = max(self.panels.draw_x1 + 10, bx1)
                ty = min(self.h - BOTTOM_BAR - 10, by2 + th + 25)
            self._outlined_text(frame, text, (tx, ty), scale, (90, 255, 130), thick)

    # ---------------- main loop ----------------
    def run(self):
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            self._frame_ts += 1
            result = self.landmarker.detect_for_video(mp_image, self._frame_ts)

            self.mode = "IDLE"
            index_tip_px = None
            index_up = False

            if result.hand_landmarks:
                self.last_hand_seen = time.time()
                landmarks = result.hand_landmarks[0]
                handedness = "Right"
                if result.handedness and result.handedness[0]:
                    handedness = result.handedness[0][0].category_name

                up = fingers_up(landmarks, handedness)
                index_up, middle_up, ring_up, pinky_up = up[1], up[2], up[3], up[4]
                open_palm = index_up and middle_up and ring_up and pinky_up
                fist = not (index_up or middle_up or ring_up or pinky_up)
                single_finger = index_up and not (middle_up or ring_up or pinky_up)
                pointing = index_up and not (ring_up or pinky_up)

                # open hand -> fist  = toggle DRAW ON / PAUSED
                self.update_toggle_gesture(open_palm, fist)

                ix, iy = landmarks[8].x * self.w, landmarks[8].y * self.h
                index_tip_px = (int(ix), int(iy))

                pts = [(int(lm.x * self.w), int(lm.y * self.h)) for lm in landmarks]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
                for p in pts:
                    cv2.circle(frame, p, 4, (0, 140, 255), -1)
                cv2.circle(frame, index_tip_px, 10, (0, 255, 255), 2)
                cv2.circle(frame, index_tip_px, 4, (0, 255, 0), -1)

                if self.draw_enabled:
                    if single_finger:              # 1 finger -> draw
                        self.mode = "DRAW"
                elif pointing:                     # paused -> finger selects buttons
                    self.mode = "SELECT"
            else:
                self.open_frames = 0
                self.fist_frames = 0

            # ---- handle modes ----
            if self.mode == "DRAW" and index_tip_px and self.panels.in_draw_area(index_tip_px):
                if self.smoothed_point is None:
                    self.smoothed_point = index_tip_px
                else:
                    sx = int(SMOOTHING * self.smoothed_point[0] + (1 - SMOOTHING) * index_tip_px[0])
                    sy = int(SMOOTHING * self.smoothed_point[1] + (1 - SMOOTHING) * index_tip_px[1])
                    self.smoothed_point = (sx, sy)
                self.draw_point(self.smoothed_point)
                self.hover_label = None
                self.hover_start_time = None
                self.hover_fired = False
            else:
                self.end_stroke()

            if self.mode == "SELECT" and index_tip_px:
                self.update_select(index_tip_px)
            else:
                self.hover_label = None
                self.hover_start_time = None
                self.hover_fired = False

            self.update_calc()

            # ---- compose output frame ----
            for layer in (self.canvas, self.calc_canvas):
                mask = np.any(layer != 0, axis=2)
                frame[mask] = layer[mask]

            # eraser size ring follows the fingertip while drawing
            if self.mode == "DRAW" and self.eraser_on and index_tip_px \
                    and self.panels.in_draw_area(index_tip_px):
                ring_pt = self.smoothed_point or index_tip_px
                cv2.circle(frame, ring_pt, max(2, self.brush_size * ERASER_SIZE_MULT // 2), (255, 255, 255), 2)

            self.panels.draw(
                frame,
                active_color_name=None if self.eraser_on else self.color_name,
                eraser_on=self.eraser_on,
                calc_on=self.calc_on,
                hover_label=self.hover_label,
                hover_progress=self.hover_progress(),
                alpha=0.45 if self.draw_enabled else 1.0,
            )
            self.draw_options(frame)
            self.draw_calc_overlay(frame)
            self._draw_hud(frame)

            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif ord('1') <= key <= ord('7'):
                self.set_color(COLORS[key - ord('1')][0])
            elif key == ord('e'):
                self.toggle_eraser()
            elif key == ord('k'):
                self.toggle_calc()
            elif key == ord('m'):
                self.set_draw_enabled(not self.draw_enabled)
            elif key == ord('d'):
                self.eraser_on = False
            elif key == ord('c'):
                self.clear()
            elif key == ord('u'):
                self.undo()
            elif key == ord('r'):
                self.redo()
            elif key == ord('s'):
                self.handle_button("Save")
            elif key in (ord('+'), ord('=')):
                self.change_size(2)
            elif key == ord('-'):
                self.change_size(-2)

        self.cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

    # ---------------- HUD ----------------
    def _draw_hud(self, frame):
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        if dt > 0:
            inst_fps = 1.0 / dt
            self.fps = self.fps * 0.9 + inst_fps * 0.1 if self.fps else inst_fps

        dx1 = self.panels.draw_x1
        dx2 = self.panels.draw_x2

        # draw ON / PAUSED status (top-left of the drawing area, below the calc bar)
        panel_w, panel_h = 500, 96
        px1, py1 = dx1 + 12, 90
        overlay = frame.copy()
        cv2.rectangle(overlay, (px1, py1), (px1 + panel_w, py1 + panel_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        if self.draw_enabled:
            state_txt, state_col = "DRAW: ON", (0, 255, 0)
            hint = "1 finger = draw | 2 fingers = lift pen | open hand + fist = menu"
        else:
            state_txt, state_col = "DRAW: PAUSED (menu)", (0, 160, 255)
            hint = "point at a button | open hand + fist = draw"
        if time.time() < self.armed_until:
            hint, hint_col = "Open hand OK -> now CLOSE your fist", (0, 255, 255)
        else:
            hint_col = (210, 210, 210)
        cv2.putText(frame, state_txt, (px1 + 12, py1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_col, 2, cv2.LINE_AA)
        cv2.putText(frame, hint, (px1 + 12, py1 + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, hint_col, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS {self.fps:.0f}   gesture: {self.mode}", (px1 + 12, py1 + 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        # coloured frame around the drawing area: green = draw ON, orange = paused
        cv2.rectangle(frame, (dx1 + 2, 2), (dx2 - 2, self.h - BOTTOM_BAR - 2),
                      state_col, 3)

        # current tool chip (bottom-left of the drawing area)
        chip_w, chip_h = 300, 50
        cx1, cy1 = dx1 + 12, self.h - BOTTOM_BAR - chip_h - 12
        overlay = frame.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx1 + chip_w, cy1 + chip_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        swatch_color = (0, 0, 0) if self.eraser_on else self.color
        cv2.circle(frame, (cx1 + 28, cy1 + chip_h // 2), 17, swatch_color, -1)
        cv2.circle(frame, (cx1 + 28, cy1 + chip_h // 2), 17, (255, 255, 255), 2)
        label = "Eraser" if self.eraser_on else self.color_name
        extra = "  [CALC]" if self.calc_on else ""
        cv2.putText(frame, f"{label} | {self.brush_size * ERASER_SIZE_MULT if self.eraser_on else (self.calc_pen if self.calc_on else self.brush_size)}px{extra}", (cx1 + 55, cy1 + chip_h // 2 + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

        # bottom help bar
        help_text = ("[1-7] Colors | [E] Eraser | [K] Calc | [C] Clear | [U] Undo | [R] Redo | "
                     "[S] Save | [M] Draw/Menu | [+/-] Size | [Q] Quit")
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, self.h - BOTTOM_BAR), (self.w, self.h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, help_text, (16, self.h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # flash message (saved / errors)
        if getattr(self, "flash_msg", None) and time.time() < self.save_flash_until:
            msg = self.flash_msg
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx = (dx1 + dx2 - tw) // 2
            ty = 205 + th
            cv2.rectangle(frame, (tx - 15, ty - th - 12), (tx + tw + 15, ty + 12),
                          self.flash_color, -1)
            cv2.putText(frame, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if time.time() - self.last_hand_seen > 1.5:
            msg = "Show your hand to the camera..."
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx = (dx1 + dx2 - tw) // 2
            ty = self.h // 2
            cv2.putText(frame, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


if __name__ == "__main__":
    app = AirDraw()
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.cap.release()
        cv2.destroyAllWindows()
