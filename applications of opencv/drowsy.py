import cv2
import mediapipe as mp
import time
import threading
import pyttsx3  
import tkinter as tk
from tkinter import Label
root = tk.Tk()
root.title("Driver Drowsiness Monitor")
root.geometry("300x150")
status_label = Label(root, text="Status: Awake", font=("Arial", 20), fg="green")
status_label.pack(pady=40)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_draw = mp.solutions.drawing_utils

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
frame_counter = 0
alarm_on = False

def calculate_ear(landmarks, img_w, img_h):
    from math import dist
    points = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in LEFT_EYE_IDX]
    A = dist(points[1], points[5])
    B = dist(points[2], points[4])
    C = dist(points[0], points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def play_alarm():
    global alarm_on
    engine = pyttsx3.init()
    engine.say("Drowsiness Alert! Please wake up.")
    engine.runAndWait()
    alarm_on = False

cap = cv2.VideoCapture(0)
def process():
    global frame_counter, alarm_on
    ret, frame = cap.read()
    if not ret:
        root.after(10, process)
        return

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            ear = calculate_ear(face_landmarks.landmark, w, h)

            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= CONSEC_FRAMES:
                    status_label.config(text="Status: Drowsy", fg="red")
                    if not alarm_on:
                        alarm_on = True
                        threading.Thread(target=play_alarm).start()
            else:
                frame_counter = 0
                status_label.config(text="Status: Awake", fg="green")

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        root.destroy()
        return

    root.after(10, process)

root.after(10, process)
root.mainloop()
