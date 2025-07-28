import cv2
import datetime
import time
import os
import pandas as pd
import pygame

attendance_file = "attendance.csv"
sound_file = "buzzer.wav"

cap = cv2.VideoCapture(0)
detector = cv2.QRCodeDetector()


pygame.mixer.init()
try:
    pygame.mixer.music.load(sound_file)
except:
    print(f"[ERROR] Could not load sound file: {sound_file}")

if not os.path.exists(attendance_file) or os.stat(attendance_file).st_size == 0:
    # File doesn't exist or empty, create with headers
    df = pd.DataFrame(columns=["Name", "Entry Time", "Exit Time"])
    df.to_csv(attendance_file, index=False)
else:
    df = pd.read_csv(attendance_file)
    required_cols = {"Name", "Entry Time", "Exit Time"}
    if not required_cols.issubset(df.columns):
        print("[ERROR] CSV file is corrupted. Deleting and recreating...")
        os.remove(attendance_file)
        df = pd.DataFrame(columns=["Name", "Entry Time", "Exit Time"])
        df.to_csv(attendance_file, index=False)

scan_count = {}

print("[INFO] Waiting for QR Code... Show QR to mark ENTRY / EXIT")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    data, bbox, _ = detector.detectAndDecode(frame)

    if data:
        name = data.strip()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Play buzzer sound
        try:
            pygame.mixer.music.play()
        except:
            print("[WARN] Failed to play sound")

        # Update scan count
        count = scan_count.get(name, 0) + 1
        scan_count[name] = count

        if count % 2 == 1:
            # Entry
            print(f"[ENTRY] {name} at {timestamp}")
            if name not in df["Name"].values:
                new_row = {"Name": name, "Entry Time": timestamp, "Exit Time": ""}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                print(f"[INFO] {name} already scanned before, awaiting exit...")
        else:
            # Exit
            print(f"[EXIT] {name} at {timestamp}")
            df.loc[df["Name"] == name, "Exit Time"] = timestamp

        df.to_csv(attendance_file, index=False)

        time.sleep(3)  

    cv2.imshow("QR Code Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
