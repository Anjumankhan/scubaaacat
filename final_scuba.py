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
pygame.mixer.init()
try:
    pygame.mixer.music.load("scuba.mp3")  # put your mp3 here
    music_loaded = True
except:
    print("MP3 file not found. Audio disabled.")
    music_loaded = False

# ---------------- CONFIG ----------------
GIF_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cat.gif")
CAT_SIZE        = (220, 220)
CAT_Y_FRACTION  = 0.05
SMOOTH_ALPHA    = 0.25
GIF_SPEED       = 2
SIZE_STEP       = 20
MOVEMENT_THRESHOLD = 8
MOVEMENT_HISTORY   = 6

GREEN_H_LOW,  GREEN_H_HIGH  = 35,  85
GREEN_S_LOW,  GREEN_S_HIGH  = 80, 255
GREEN_V_LOW,  GREEN_V_HIGH  = 80, 255
CHROMA_BLUR = 3

# ---------------- GREEN SCREEN ----------------
def remove_green_screen(bgra_frame):
    bgr  = bgra_frame[:, :, :3]
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([GREEN_H_LOW,  GREEN_S_LOW,  GREEN_V_LOW],  dtype=np.uint8)
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
    frames  = []

    try:
        while True:
            rgba = pil_gif.convert("RGBA")
            arr  = np.array(rgba, dtype=np.uint8)

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
    x2, y2 = min(x+ow, bw), min(y+oh, bh)

    if x2 <= x1 or y2 <= y1:
        return background

    src = overlay[y1-y:y2-y, x1-x:x2-x]
    dst = background[y1:y2, x1:x2]

    alpha = src[:, :, 3:4] / 255.0
    dst[:] = src[:, :, :3] * alpha + dst * (1 - alpha)

    return background

# ---------------- HELPERS ----------------
class Smoother:
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.val = None
    def update(self, new):
        self.val = new if self.val is None else self.alpha * new + (1 - self.alpha) * self.val
        return self.val

class MovementDetector:
    def __init__(self, history=6, threshold=8):
        self.history = []
        self.max_hist = history
        self.threshold = threshold
    def update(self, x):
        self.history.append(x)
        if len(self.history) > self.max_hist:
            self.history.pop(0)
    def is_moving(self):
        if len(self.history) < 2:
            return False
        return (max(self.history) - min(self.history)) > self.threshold

def is_near_nose(hand_lms, face_lms, threshold=0.13):    #chal jaa
    if face_lms is None:
        return False
    nose = face_lms.landmark[1]
    for i in [0,4,8]:
        lm = hand_lms.landmark[i]
        if ((lm.x-nose.x)**2 + (lm.y-nose.y)**2)**0.5 < threshold:
            return True
    return False

# ---------------- MAIN ----------------
def main():
    cap = cv2.VideoCapture(0)

    gif_frames = load_gif_frames(GIF_PATH, CAT_SIZE)
    if not gif_frames:
        print("No GIF found.")
        sys.exit()

    mp_hands = mp.solutions.hands
    mp_face  = mp.solutions.face_mesh

    smoother = Smoother(SMOOTH_ALPHA)
    move_det = MovementDetector(MOVEMENT_HISTORY, MOVEMENT_THRESHOLD)

    gif_idx = 0
    tick = 0
    cat_x = None

    with mp_hands.Hands(max_num_hands=2) as hands, \
         mp_face.FaceMesh(max_num_faces=1) as face:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hr = hands.process(rgb)
            fr = face.process(rgb)

            face_lms = fr.multi_face_landmarks[0] if fr.multi_face_landmarks else None

            moving_hand = None

            if hr.multi_hand_landmarks:
                moving_hand = hr.multi_hand_landmarks[0]

            hand_is_moving = False

            if moving_hand:
                wrist = moving_hand.landmark[0]
                x = int(wrist.x * w)
                move_det.update(x)

                hand_is_moving = move_det.is_moving()
                smoothed = smoother.update(wrist.x)
                cat_x = int(smoothed * w)

            tick += 1
            if hand_is_moving and tick % GIF_SPEED == 0:
                gif_idx = (gif_idx + 1) % len(gif_frames)

            # ---------------- MUSIC CONTROL ----------------
            if music_loaded:
                if hand_is_moving:
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play(-1)
                else:
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.pause()

            frame = overlay_bgra(
                frame,
                gif_frames[gif_idx],
                (cat_x or w//2) - CAT_SIZE[0]//2,
                int(h * CAT_Y_FRACTION)
            )

            cv2.imshow("Cat Mirror", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if music_loaded:
        pygame.mixer.music.stop()
        pygame.mixer.quit()

if __name__ == "__main__":
    main()