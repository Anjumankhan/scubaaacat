"""
Cat Gesture Mirror - Scuba Cat Edition (with Music)
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from PIL import Image
import pygame

# ---------------- MUSIC INIT ----------------
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

try:
    pygame.mixer.music.load("scuba.mp3")
    music_loaded = True
except:
    print("MP3 file not found. Audio disabled.")
    music_loaded = False

# ---------------- CONFIG ----------------
GIF_PATH           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cat.gif")
CAT_Y_FRACTION     = 0.05
SMOOTH_ALPHA       = 0.2
GIF_SPEED          = 1
SIZE_STEP          = 20
MOVEMENT_THRESHOLD = 5
MOVEMENT_HISTORY   = 8

GREEN_H_LOW,  GREEN_H_HIGH  = 35,  85
GREEN_S_LOW,  GREEN_S_HIGH  = 80, 255
GREEN_V_LOW,  GREEN_V_HIGH  = 80, 255
CHROMA_BLUR = 3

# ---------------- GREEN SCREEN ----------------
def remove_green_screen(bgra_frame):
    bgr = bgra_frame[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([GREEN_H_LOW, GREEN_S_LOW, GREEN_V_LOW], dtype=np.uint8)
    upper = np.array([GREEN_H_HIGH, GREEN_S_HIGH, GREEN_V_HIGH], dtype=np.uint8)

    green_mask = cv2.inRange(hsv, lower, upper)

    if CHROMA_BLUR > 1:
        green_mask = cv2.GaussianBlur(green_mask, (CHROMA_BLUR, CHROMA_BLUR), 0)
        _, green_mask = cv2.threshold(green_mask, 127, 255, cv2.THRESH_BINARY)

    result = bgra_frame.copy()
    result[:, :, 3] = np.where(green_mask == 255, 0, bgra_frame[:, :, 3])
    return result

# ---------------- GIF LOADER ----------------
def load_gif_frames(path, size):
    if not os.path.exists(path):
        print("GIF not found:", path)
        return None

    pil_gif = Image.open(path)
    frames = []

    try:
        while True:
            rgba = pil_gif.convert("RGBA")
            arr = np.array(rgba, dtype=np.uint8)

            bgra = arr.copy()
            bgra[:, :, 0] = arr[:, :, 2]
            bgra[:, :, 2] = arr[:, :, 0]

            bgra = remove_green_screen(bgra)
            bgra = cv2.resize(bgra, size)

            frames.append(bgra)

            pil_gif.seek(pil_gif.tell() + 1)
    except EOFError:
        pass

    print("Loaded", len(frames), "frames")
    return frames

# ---------------- OVERLAY ----------------
def overlay_bgra(background, overlay, x, y):
    oh, ow = overlay.shape[:2]
    bh, bw = background.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ow, bw), min(y + oh, bh)

    if x2 <= x1 or y2 <= y1:
        return background

    src = overlay[y1 - y:y2 - y, x1 - x:x2 - x]
    dst = background[y1:y2, x1:x2]

    alpha = src[:, :, 3:4] / 255.0
    dst[:] = src[:, :, :3] * alpha + dst * (1 - alpha)

    return background

# ---------------- HELPERS ----------------
class Smoother:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.val = None

    def update(self, new):
        self.val = new if self.val is None else self.alpha * new + (1 - self.alpha) * self.val
        return self.val


class MovementDetector:
    def __init__(self, history=8, threshold=5):
        self.history = []
        self.max_hist = history
        self.threshold = threshold
        self.moving_frames = 0
        self.still_frames = 0
        self.MOVING_CONFIRM = 2
        self.STILL_CONFIRM = 10
        self._is_moving = False

    def update(self, x):
        self.history.append(x)
        if len(self.history) > self.max_hist:
            self.history.pop(0)

        raw_moving = (max(self.history) - min(self.history)) > self.threshold if len(self.history) >= 2 else False

        if raw_moving:
            self.moving_frames += 1
            self.still_frames = 0
            if self.moving_frames >= self.MOVING_CONFIRM:
                self._is_moving = True
        else:
            self.still_frames += 1
            self.moving_frames = 0
            if self.still_frames >= self.STILL_CONFIRM:
                self._is_moving = False

    def is_moving(self):
        return self._is_moving


# ---------------- MAIN ----------------
def main():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Cat Mirror", cv2.WINDOW_NORMAL)

    gif_frames = None
    gif_idx = 0
    tick = 0
    cat_x = None

    mp_hands = mp.solutions.hands

    smoother = Smoother(SMOOTH_ALPHA)
    move_det = MovementDetector(MOVEMENT_HISTORY, MOVEMENT_THRESHOLD)

    with mp_hands.Hands(max_num_hands=2) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # ---------------- DYNAMIC CAT SIZE ----------------
            cat_scale = 0.25
            cat_size = (int(w * cat_scale), int(w * cat_scale))

            # Load GIF once size is known
            if gif_frames is None:
                gif_frames = load_gif_frames(GIF_PATH, cat_size)
                if not gif_frames:
                    print("No GIF found.")
                    sys.exit()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hr = hands.process(rgb)

            moving_hand = None

            if hr.multi_hand_landmarks:
                moving_hand = hr.multi_hand_landmarks[0]

            if moving_hand:
                wrist = moving_hand.landmark[0]
                x = int(wrist.x * w)

                move_det.update(x)
                smoothed = smoother.update(wrist.x)
                cat_x = int(smoothed * w)
            else:
                move_det.update(move_det.history[-1] if move_det.history else 0)

            hand_is_moving = move_det.is_moving()

            tick += 1
            if tick % (GIF_SPEED if hand_is_moving else 6) == 0:
                gif_idx = (gif_idx + 1) % len(gif_frames)

            # ---------------- MUSIC ----------------
            if music_loaded:
                if hand_is_moving:
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play(-1)
                        pygame.mixer.music.set_volume(1.0)
                else:
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.fadeout(500)

            # ---------------- OVERLAY ----------------
            frame = overlay_bgra(
                frame,
                gif_frames[gif_idx],
                (cat_x or w // 2) - cat_size[0] // 2,
                int(h * CAT_Y_FRACTION)
            )

            cv2.imshow("Cat Mirror", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if music_loaded:
        pygame.mixer.music.stop()
        pygame.mixer.quit()


if __name__ == "__main__":
    main()