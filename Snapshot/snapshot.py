import cv2
import time

# Open the default camera (0 for default webcam)
cap = cv2.VideoCapture(0)

# Check if the camera opened correctly
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Time reference for snapshots
last_capture_time = time.time()
snapshot_interval = 0.5  # seconds
snapshot_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break

    # Show the live video feed
    cv2.imshow('Live Feed', frame)

    # Take snapshot every 0.5 seconds
    current_time = time.time()
    if current_time - last_capture_time >= snapshot_interval:
        snapshot_count += 1
        filename = f"snapshot_{snapshot_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        last_capture_time = current_time

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
