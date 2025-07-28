import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Start webcam
cap = cv2.VideoCapture(0)

clicking = False  # Flag to avoid multiple clicks per gesture

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Index finger tip
            x_index = int(hand_landmarks.landmark[8].x * frame_w)
            y_index = int(hand_landmarks.landmark[8].y * frame_h)

            # Thumb tip
            x_thumb = int(hand_landmarks.landmark[4].x * frame_w)
            y_thumb = int(hand_landmarks.landmark[4].y * frame_h)

            # Convert to screen coordinates
            screen_x = int(hand_landmarks.landmark[8].x * screen_w)
            screen_y = int(hand_landmarks.landmark[8].y * screen_h)

            # Move cursor
            pyautogui.moveTo(screen_x, screen_y)

            # Draw circles
            cv2.circle(frame, (x_index, y_index), 10, (0, 255, 0), -1)
            cv2.circle(frame, (x_thumb, y_thumb), 10, (0, 0, 255), -1)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate distance between index and thumb
            distance = math.hypot(x_thumb - x_index, y_thumb - y_index)

            # Click if fingers are close
            if distance < 30:
                if not clicking:
                    clicking = True
                    pyautogui.click()
                    cv2.putText(frame, "Click", (x_index, y_index - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                clicking = False

    cv2.imshow("Virtual Mouse - Click Enabled", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()
