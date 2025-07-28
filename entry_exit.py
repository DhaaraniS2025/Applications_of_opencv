import cv2
import time
from datetime import datetime

# === Initialize HOG + SVM detector for humans ===
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# === Laptop Camera (0 is default webcam) ===
cap = cv2.VideoCapture(0)

# === Line for entry/exit ===
line_position = 250  # vertical line position
offset = 30          # tolerance range

entry_count = 0
exit_count = 0

def center(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

print("[INFO] Starting human detection using laptop webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame from webcam.")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect people
    boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8))

    counted_entries = 0
    counted_exits = 0

    for (x, y, w, h) in boxes:
        cx, cy = center(x, y, w, h)

        # Draw rectangle and center
        color = (0, 255, 0) if cy < line_position else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (cx, cy), 5, color, -1)

        # Check crossing
        if abs(cy - line_position) < offset:
            if cy < line_position:
                counted_entries += 1
            else:
                counted_exits += 1

    if counted_entries > 0:
        entry_count += counted_entries
        print(f"[ENTRY] +{counted_entries} at {datetime.now().strftime('%H:%M:%S')}")

    if counted_exits > 0:
        exit_count += counted_exits
        print(f"[EXIT] +{counted_exits} at {datetime.now().strftime('%H:%M:%S')}")

    # Draw line and counts
    cv2.line(frame, (0, line_position), (640, line_position), (255, 255, 255), 2)
    cv2.putText(frame, f"Entry: {entry_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Exit: {exit_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Human Counter - Laptop Cam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
