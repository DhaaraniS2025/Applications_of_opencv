import cv2
import mediapipe as mp
import pyautogui
import math
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = handLms.landmark
            # Thumb tip: 4, Index finger tip: 8
            x1, y1 = int(lm_list[4].x * w), int(lm_list[4].y * h)
            x2, y2 = int(lm_list[8].x * w), int(lm_list[8].y * h)

            cv2.circle(frame, (x1, y1), 8, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 8, (255, 0, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            distance = calculate_distance(x1, y1, x2, y2)

            # Volume control thresholds
            if distance > 100:
                pyautogui.press("volumeup")
                cv2.putText(frame, "Volume UP", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif distance < 40:
                pyautogui.press("volumedown")
                cv2.putText(frame, "Volume DOWN", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Volume Controller", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
