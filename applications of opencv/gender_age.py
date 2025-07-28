import cv2
from deepface import DeepFace

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze the frame for emotion, age, gender
        results = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)

        for face_data in results:
            x, y, w, h = face_data['region']['x'], face_data['region']['y'], face_data['region']['w'], face_data['region']['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            # Show analysis results
            text = f"{face_data['dominant_emotion']}, {face_data['gender']}, Age: {face_data['age']}"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    except Exception as e:
        print(f"[Warning] {e}")

    cv2.imshow("DeepFace Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
