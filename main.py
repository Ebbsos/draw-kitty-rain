import cv2
import numpy as np
import mediapipe as mp
import random
import math
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# CONFIGURACION
# =========================
CAMERA_INDEX = 0
MODEL_PATH = "hand_landmarker.task"

FRAME_WIDTH = 960
FRAME_HEIGHT = 720

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 980

WINDOW_NAME = "Magic Draw - Kitty Rain"

BRUSH_THICKNESS = 3
MAX_POINT_JUMP = 55
BUTTON_HOLD_FRAMES = 8

HEADER_HEIGHT = 95

# solo boton limpiar
BUTTON_Y1, BUTTON_Y2 = 20, 58
CLEAR_BTN = (20, 145)

INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10

KITTY_RAIN_FRAMES = 180
MAX_KITTIES = 7

VINTAGE_FRAME = (205, 215, 228)
VINTAGE_BAR = (185, 195, 210)
VINTAGE_TEXT = (95, 80, 70)

# scrapbook
SCRAP_TOP = 90
SCRAP_BOTTOM = 150
SCRAP_LEFT = 70
SCRAP_RIGHT = 70
POLAROID_TOP = 18
POLAROID_SIDE = 18
POLAROID_BOTTOM = 70

# =========================
# CARGAR GATITOS
# =========================
def load_kitty_images():
    kitty_images = []
    valid_ext = (".png", ".jpg", ".jpeg", ".webp")

    for filename in os.listdir("."):
        lower = filename.lower()

        if not lower.endswith(valid_ext):
            continue
        if not lower.startswith("kitty"):
            continue

        img = cv2.imread(filename)
        if img is not None:
            kitty_images.append(img)

    return kitty_images

def resize_img(img, scale):
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def overlay_img(bg, img, x, y):
    h, w = img.shape[:2]
    bg_h, bg_w = bg.shape[:2]

    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + w)
    y2 = min(bg_h, y + h)

    img_x1 = x1 - x
    img_y1 = y1 - y
    img_x2 = img_x1 + (x2 - x1)
    img_y2 = img_y1 + (y2 - y1)

    bg[y1:y2, x1:x2] = img[img_y1:img_y2, img_x1:img_x2]

# =========================
# UI
# =========================
def draw_soft_button(frame, x1, x2, text):
    cv2.rectangle(frame, (x1, BUTTON_Y1), (x2, BUTTON_Y2), (222, 210, 208), -1)
    cv2.rectangle(frame, (x1, BUTTON_Y1), (x2, BUTTON_Y2), (245, 240, 238), 1)

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0]
    tx = x1 + (x2 - x1 - text_size[0]) // 2
    ty = BUTTON_Y1 + (BUTTON_Y2 - BUTTON_Y1 + text_size[1]) // 2
    cv2.putText(frame, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, VINTAGE_TEXT, 1, cv2.LINE_AA)

def draw_ui(frame, hold_name=None, kitty_active=False):
    draw_soft_button(frame, CLEAR_BTN[0], CLEAR_BTN[1], "LIMPIAR")

    if hold_name:
        cv2.putText(frame, "Limpiando...", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 240), 1, cv2.LINE_AA)

    if kitty_active:
        cv2.putText(frame, "Kitty Rain!", (720, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (250, 238, 242), 2, cv2.LINE_AA)

# =========================
# EFECTOS MAGICOS
# =========================
def draw_star(img, x, y, color):
    size = random.randint(4, 7)
    cv2.line(img, (x - size, y), (x + size, y), color, 1)
    cv2.line(img, (x, y - size), (x, y + size), color, 1)
    cv2.line(img, (x - size + 2, y - size + 2), (x + size - 2, y + size - 2), color, 1)
    cv2.line(img, (x - size + 2, y + size - 2), (x + size - 2, y - size + 2), color, 1)

def draw_magic_effect(canvas, point):
    x, y = point
    if random.random() < 0.18:
        draw_star(canvas, x, y, (255, 250, 250))
    if random.random() < 0.14:
        px = x + random.randint(-7, 7)
        py = y + random.randint(-7, 7)
        cv2.circle(canvas, (px, py), 1, (235, 220, 230), -1)

def draw_wand(frame, fingertip):
    x, y = fingertip
    wand_start = (x - 22, y + 22)
    wand_end = (x - 4, y + 4)
    cv2.line(frame, wand_start, wand_end, (175, 205, 230), 4)
    cv2.line(frame, wand_start, wand_end, (120, 140, 165), 1)
    cv2.circle(frame, wand_end, 4, (255, 250, 248), -1)
    draw_star(frame, x + 3, y - 3, (255, 248, 250))

def draw_glow_ring(frame, point):
    x, y = point
    cv2.circle(frame, (x, y), 13, (240, 225, 235), 1)
    cv2.circle(frame, (x, y), 8, (250, 245, 245), 1)
    cv2.circle(frame, (x, y), 4, (255, 255, 250), -1)

# =========================
# GESTOS
# =========================
def detect_fingers_up(hand_landmarks):
    index_up = hand_landmarks[INDEX_PIP].y - hand_landmarks[INDEX_TIP].y > 0.02
    middle_up = hand_landmarks[MIDDLE_PIP].y - hand_landmarks[MIDDLE_TIP].y > 0.03
    return index_up, middle_up

def point_inside_button(x, y, button):
    return button[0] < x < button[1] and BUTTON_Y1 < y < BUTTON_Y2

# =========================
# DETECCION CORAZON
# =========================
def detect_heart(points):
    if len(points) < 20:
        return False

    if math.dist(points[0], points[-1]) > 80:
        return False

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    w = max(xs) - min(xs)
    h = max(ys) - min(ys)

    if w < 50 or h < 50:
        return False

    ratio = w / max(h, 1)
    return 0.6 <= ratio <= 1.6

# =========================
# GATITOS
# =========================
def create_kitty(kitty_images, frame_w):
    img = random.choice(kitty_images)
    img = resize_img(img, random.uniform(0.12, 0.18))

    max_x = max(1, frame_w - img.shape[1])

    return {
        "img": img,
        "x": random.randint(0, max_x),
        "y": random.randint(HEADER_HEIGHT + 5, HEADER_HEIGHT + 35),
        "speed": random.uniform(1.5, 2.4),
        "phase": random.uniform(0, 6.28),
        "drift": random.uniform(0.6, 1.2)
    }

def update_kitties(kitties):
    for k in kitties:
        k["y"] += k["speed"]
        k["x"] += math.sin(k["y"] * 0.03 + k["phase"]) * k["drift"]

def draw_kitties(frame, kitties):
    for k in kitties:
        overlay_img(frame, k["img"], int(k["x"]), int(k["y"]))

def respawn_kitties(kitties, kitty_images, frame_w, frame_h):
    for i, k in enumerate(kitties):
        if k["y"] > frame_h + 40:
            kitties[i] = create_kitty(kitty_images, frame_w)

# =========================
# EFECTO VINTAGE
# =========================
def apply_warm_tone(frame):
    warm = frame.astype(np.float32)
    warm[:, :, 2] *= 1.05
    warm[:, :, 1] *= 1.01
    warm[:, :, 0] *= 0.95
    return np.clip(warm, 0, 255).astype(np.uint8)

def apply_vignette(frame):
    rows, cols = frame.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols / 2.2)
    kernel_y = cv2.getGaussianKernel(rows, rows / 2.2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()

    vignette = np.empty_like(frame, dtype=np.float32)
    for i in range(3):
        vignette[:, :, i] = frame[:, :, i] * mask

    return np.clip(vignette, 0, 255).astype(np.uint8)

def add_film_grain(frame, intensity=5):
    noise = np.random.normal(0, intensity, frame.shape).astype(np.int16)
    grain = frame.astype(np.int16) + noise
    return np.clip(grain, 0, 255).astype(np.uint8)

# =========================
# SCRAPBOOK FRAME
# =========================
def add_paper_texture(board, intensity=5):
    noise = np.random.normal(0, intensity, board.shape).astype(np.int16)
    textured = board.astype(np.int16) + noise
    return np.clip(textured, 0, 255).astype(np.uint8)

def draw_tape(board, x, y, w, h, color=(210, 205, 190)):
    overlay = board.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    board[:] = cv2.addWeighted(overlay, 0.58, board, 0.42, 0)

    for i in range(0, h, 4):
        cv2.line(board, (x + 4, y + i), (x + w - 4, y + i), (225, 222, 214), 1)

def draw_photo_corners(img, x, y, w, h, color=(95, 75, 55), thickness=8, size=24):
    cv2.line(img, (x, y + size), (x, y), color, thickness)
    cv2.line(img, (x, y), (x + size, y), color, thickness)

    cv2.line(img, (x + w - size, y), (x + w, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + size), color, thickness)

    cv2.line(img, (x, y + h - size), (x, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x + size, y + h), color, thickness)

    cv2.line(img, (x + w - size, y + h), (x + w, y + h), color, thickness)
    cv2.line(img, (x + w, y + h - size), (x + w, y + h), color, thickness)

def create_scrapbook_frame(photo, title="", subtitle=""):
    ph, pw = photo.shape[:2]

    board_h = ph + SCRAP_TOP + SCRAP_BOTTOM
    board_w = pw + SCRAP_LEFT + SCRAP_RIGHT
    board = np.full((board_h, board_w, 3), (232, 223, 210), dtype=np.uint8)
    board = add_paper_texture(board, intensity=5)

    polaroid_h = ph + POLAROID_TOP + POLAROID_BOTTOM
    polaroid_w = pw + POLAROID_SIDE * 2
    polaroid = np.full((polaroid_h, polaroid_w, 3), 245, dtype=np.uint8)

    # sombra
    shadow_x = SCRAP_LEFT + 8
    shadow_y = SCRAP_TOP + 8
    cv2.rectangle(board, (shadow_x, shadow_y),
                  (shadow_x + polaroid_w, shadow_y + polaroid_h),
                  (170, 160, 150), -1)

    # foto dentro de polaroid
    polaroid[POLAROID_TOP:POLAROID_TOP + ph, POLAROID_SIDE:POLAROID_SIDE + pw] = photo

    # texto abajo
    cv2.putText(polaroid, title, (36, polaroid_h - 28),
                cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.4, (90, 70, 60), 2, cv2.LINE_AA)
    cv2.putText(polaroid, subtitle, (40, polaroid_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 105, 95), 1, cv2.LINE_AA)

    # pegar polaroid
    px = SCRAP_LEFT
    py = SCRAP_TOP
    board[py:py + polaroid_h, px:px + polaroid_w] = polaroid

    # esquinas tipo foto
    draw_photo_corners(board, px + 10, py + 10, polaroid_w - 20, polaroid_h - 78)

    # cintas adhesivas
    draw_tape(board, px + 25, py - 20, 110, 24, color=(214, 190, 195))
    draw_tape(board, px + polaroid_w - 140, py - 18, 110, 24, color=(220, 207, 184))
    draw_tape(board, 20, board_h - 72, 95, 24, color=(221, 208, 188))
    draw_tape(board, board_w - 150, board_h - 82, 105, 26, color=(213, 202, 187))

    # mini corazon y estrellita
    cv2.putText(board, "♡", (board_w - 65, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (175, 132, 145), 2, cv2.LINE_AA)
    cv2.putText(board, "✦", (board_w - 95, board_h - 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (185, 160, 125), 2, cv2.LINE_AA)

    # pequeños doodles
    cv2.putText(board, "scrapbook vibes", (28, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 105, 95), 1, cv2.LINE_AA)
    cv2.putText(board, "you are magic", (board_w - 175, board_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (105, 90, 82), 1, cv2.LINE_AA)

    # borde fino exterior
    cv2.rectangle(board, (10, 10), (board_w - 11, board_h - 11), (240, 236, 230), 1)

    return board

# =========================
# MEDIAPIPE
# =========================
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.25,
    min_hand_presence_confidence=0.25,
    min_tracking_confidence=0.25
)

landmarker = HandLandmarker.create_from_options(options)

# =========================
# INICIO
# =========================
kitty_images = load_kitty_images()
print("Gatitos cargados:", len(kitty_images))

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("No se pudo leer la camara.")

first_frame = cv2.flip(first_frame, 1)
actual_h, actual_w = first_frame.shape[:2]

canvas = np.zeros((actual_h, actual_w, 3), dtype=np.uint8)

drawing_points = []
last_point = None

kitty_rain = False
kitties = []
timer = 0

button_hold_name = None
button_hold_count = 0

frame_count = 0

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

# =========================
# LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, frame_count * 33)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        x = int(hand[INDEX_TIP].x * frame_w)
        y = int(hand[INDEX_TIP].y * frame_h)
        pt = (x, y)

        index, middle = detect_fingers_up(hand)

        if index and middle:
            if drawing_points:
                if detect_heart(drawing_points) and kitty_images:
                    kitty_rain = True
                    timer = KITTY_RAIN_FRAMES
                    kitties = [create_kitty(kitty_images, frame_w) for _ in range(MAX_KITTIES)]
                drawing_points = []

            last_point = None

            current_hover = None
            if point_inside_button(x, y, CLEAR_BTN):
                current_hover = "LIMPIAR"

            if current_hover is not None:
                if current_hover == button_hold_name:
                    button_hold_count += 1
                else:
                    button_hold_name = current_hover
                    button_hold_count = 1

                if button_hold_count >= BUTTON_HOLD_FRAMES:
                    canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                    button_hold_name = None
                    button_hold_count = 0
            else:
                button_hold_name = None
                button_hold_count = 0

        elif index and not middle:
            button_hold_name = None
            button_hold_count = 0

            if last_point is not None and math.dist(last_point, pt) > MAX_POINT_JUMP:
                pt = last_point

            if last_point is not None:
                cv2.line(canvas, last_point, pt, (245, 220, 230), BRUSH_THICKNESS)
                draw_magic_effect(canvas, pt)
                drawing_points.append(pt)
            else:
                drawing_points.append(pt)

            last_point = pt

        else:
            if drawing_points:
                if detect_heart(drawing_points) and kitty_images:
                    kitty_rain = True
                    timer = KITTY_RAIN_FRAMES
                    kitties = [create_kitty(kitty_images, frame_w) for _ in range(MAX_KITTIES)]
                drawing_points = []

            last_point = None
            button_hold_name = None
            button_hold_count = 0

        draw_glow_ring(frame, pt)
        draw_wand(frame, pt)

    else:
        if drawing_points:
            if detect_heart(drawing_points) and kitty_images:
                kitty_rain = True
                timer = KITTY_RAIN_FRAMES
                kitties = [create_kitty(kitty_images, frame_w) for _ in range(MAX_KITTIES)]
            drawing_points = []

        last_point = None
        button_hold_name = None
        button_hold_count = 0

    if canvas.shape[:2] != frame.shape[:2]:
        canvas = cv2.resize(canvas, (frame_w, frame_h))

    frame = cv2.add(frame, canvas)

    if kitty_rain and kitty_images:
        update_kitties(kitties)
        respawn_kitties(kitties, kitty_images, frame_w, frame_h)
        draw_kitties(frame, kitties)

        timer -= 1
        if timer <= 0:
            kitty_rain = False
            kitties = []

    frame = apply_warm_tone(frame)
    frame = apply_vignette(frame)
    frame = add_film_grain(frame, intensity=5)

    draw_ui(frame, button_hold_name, kitty_rain)

    cv2.putText(frame, "Indice: dibuja  |  Indice + medio: limpiar  |  Q: salir",
                (20, frame_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 230, 225), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("p"):
        cv2.imwrite("scrapbook_magic_draw.png", scrap_view)
        print("Guardado como scrapbook_magic_draw.png")

cap.release()
cv2.destroyAllWindows()