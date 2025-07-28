import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import Canvas

root = tk.Tk()
root.title("Virtual Switchboard")
root.geometry("400x200")

canvas = Canvas(root, width=400, height=200)
canvas.pack()

light1 = canvas.create_oval(50, 50, 100, 100, fill="red")
light2 = canvas.create_oval(150, 50, 200, 100, fill="red")
light3 = canvas.create_oval(250, 50, 300, 100, fill="red")
canvas.create_text(75, 120, text="Light 1")
canvas.create_text(175, 120, text="Light 2")
canvas.create_text(275, 120, text="Light 3")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def update_lights(l1, l2, l3):
    canvas.itemconfig(light1, fill="green" if l1 else "red")
    canvas.itemconfig(light2, fill="green" if l2 else "red")
    canvas.itemconfig(light3, fill="green" if l3 else "red")

def process():
    ret, frame = cap.read()
    if not ret:
        root.after(10, process)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    l1 = l2 = l3 = False

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            lm = handLms.landmark
            h, w, _ = frame.shape

            index_open = lm[8].y < lm[6].y
            middle_open = lm[12].y < lm[10].y
            ring_open = lm[16].y < lm[14].y

            l1 = index_open
            l2 = middle_open
            l3 = ring_open

    update_lights(l1, l2, l3)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        root.destroy()
        return

    root.after(10, process)

root.after(10, process)
root.mainloop()