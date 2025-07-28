import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
import pygame
import os

sound_enabled = True
try:
    pygame.mixer.init()
    if os.path.exists("win.wav") and os.path.exists("lose.wav"):
        win_sound = pygame.mixer.Sound("win.wav")
        lose_sound = pygame.mixer.Sound("lose.wav")
    else:
        print("Sound files not found. Sound disabled.")
        sound_enabled = False
except pygame.error as e:
    print(f"Pygame sound error: {e}")
    sound_enabled = False

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def point_in_triangle(pt, v1, v2, v3):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    b1 = sign(pt, v1, v2) < 0.0
    b2 = sign(pt, v2, v3) < 0.0
    b3 = sign(pt, v3, v1) < 0.0
    return (b1 == b2) and (b2 == b3)

def animated_text(frame, text, position, base_color, time_passed):
    size = 2 + 0.3 * math.sin(2 * math.pi * time_passed)
    thickness = 2 + int(2 + 2 * math.sin(2 * math.pi * time_passed))
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_DUPLEX, size, base_color, thickness)

class Shape:
    def __init__(self, kind, center, size, color, filled=True):
        self.kind = kind
        self.center = center
        self.size = size
        self.color = color
        self.filled = filled
        self.locked = False

    def draw(self, img):
        x, y = self.center
        thickness = -1 if self.filled else 3
        if self.kind == "circle":
            cv2.circle(img, (int(x), int(y)), self.size, self.color, thickness)
        elif self.kind == "square":
            cv2.rectangle(img, (int(x - self.size), int(y - self.size)),
                          (int(x + self.size), int(y + self.size)), self.color, thickness)
        elif self.kind == "triangle":
            pts = self.vertices()
            cv2.drawContours(img, [np.array(pts, dtype=np.int32)], 0, self.color, thickness)

    def contains(self, pt):
        x, y = self.center
        if self.kind == "circle":
            return euclidean(pt, (x, y)) <= self.size
        elif self.kind == "square":
            return (x - self.size <= pt[0] <= x + self.size) and (y - self.size <= pt[1] <= y + self.size)
        elif self.kind == "triangle":
            v1, v2, v3 = self.vertices()
            return point_in_triangle(pt, v1, v2, v3)
        return False

    def vertices(self):
        x, y = self.center
        h = self.size * math.sqrt(3)
        p1 = (x, int(y - 2 / 3 * h))
        p2 = (int(x - self.size), int(y + 1 / 3 * h))
        p3 = (int(x + self.size), int(y + 1 / 3 * h))
        return p1, p2, p3


def main():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    with mp_hands.Hands(max_num_hands=1) as hands:
        w, h = 1280, 720
        cap.set(3, w)
        cap.set(4, h)

        game_start = time.time()
        score = 0
        won = False
        lost = False
        matched = 0
        grab_idx = None
        grab_offset = (0, 0)
        placed_shapes = []
        message_time = 0

        target_positions = [(300, 200), (500, 200), (700, 200)]
        kinds = ["circle", "square", "triangle"]
        target_shapes = [Shape(k, p, 40, (200, 200, 200), filled=False) for k, p in zip(kinds, target_positions)]

        drag_positions = [(300, 500), (500, 500), (700, 500)]
        random.shuffle(drag_positions)
        draggable_shapes = [Shape(k, p, 40, (0,255,0), filled=True) for k, p in zip(kinds, drag_positions)]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            pinch = False
            pinch_point = None

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                lm = hand.landmark
                x4, y4 = int(lm[4].x * w), int(lm[4].y * h)
                x8, y8 = int(lm[8].x * w), int(lm[8].y * h)

                pinch_point = ((x4 + x8) // 2, (y4 + y8) // 2)
                pinch = euclidean((x4, y4), (x8, y8)) < 40

                cv2.circle(frame, (x8, y8), 5, (255, 255, 255), -1)
                cv2.circle(frame, (x4, y4), 5, (255, 255, 255), -1)
                cv2.line(frame, (x4, y4), (x8, y8), (255, 255, 255), 1)
                cv2.circle(frame, pinch_point, 10, (0, 255, 255), 2)

            if pinch and pinch_point:
                if grab_idx is None:
                    for i in range(len(draggable_shapes)-1, -1, -1):
                        s = draggable_shapes[i]
                        if s.contains(pinch_point) and not s.locked:
                            grab_idx = i
                            cx, cy = s.center
                            grab_offset = (cx - pinch_point[0], cy - pinch_point[1])
                            break
                elif grab_idx is not None:
                    cx, cy = pinch_point[0] + grab_offset[0], pinch_point[1] + grab_offset[1]
                    draggable_shapes[grab_idx].center = (cx, cy)
            elif not pinch and grab_idx is not None:
                s = draggable_shapes[grab_idx]
                for target in target_shapes:
                    if s.kind == target.kind and target.contains(s.center):
                        s.center = target.center
                        s.locked = True
                        matched += 1
                        placed_shapes.append(s)
                        break
                grab_idx = None

                if matched == len(draggable_shapes):
                    correct = sum(1 for s in draggable_shapes if any(
                        s.kind == t.kind and s.center == t.center for t in target_shapes))
                    if correct == len(draggable_shapes):
                        won = True
                        message_time = time.time()
                        score += 1
                        if sound_enabled:
                            win_sound.play()
                    else:
                        lost = True
                        message_time = time.time()
                        if sound_enabled:
                            lose_sound.play()

            for t in target_shapes:
                t.draw(frame)
            for s in draggable_shapes:
                s.draw(frame)

            elapsed = int(time.time() - game_start)
            cv2.putText(frame, f"Time: {elapsed}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Score: {score}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if won:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0,0), (w,h), (0,255,0), -1)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                animated_text(frame, "YOU WON!", (400, 360), (0, 255, 0), time.time() - message_time)
            elif lost:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0,0), (w,h), (0,0,255), -1)
                frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
                animated_text(frame, "YOU LOST!", (400, 360), (0, 0, 255), time.time() - message_time)

            if (won or lost) and time.time() - message_time > 3:
                matched = 0
                won = False
                lost = False
                placed_shapes.clear()
                grab_idx = None
                drag_positions = [(300, 500), (500, 500), (700, 500)]
                random.shuffle(drag_positions)
                draggable_shapes = [Shape(k, p, 40, (0, 255, 255), filled=True) for k, p in zip(kinds, drag_positions)]
                game_start = time.time()

            cv2.imshow("Gesture Puzzle Game", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
